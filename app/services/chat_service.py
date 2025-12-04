from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.config import settings
from app.models import ChatRequest
import asyncpg
from typing import List


def get_chat_llm(provider: str | None = None):
    """LLM 프로바이더를 선택하여 반환합니다."""
    llm_provider = provider or settings.default_llm_provider
    
    if llm_provider == 'google':
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=settings.google_api_key,
            temperature=0.2
        )
    
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=settings.openai_api_key
    )


# 프롬프트 템플릿
prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 적성검사 결과 상담 전문 챗봇입니다. 

🚨 **절대 규칙** 🚨
1. **반드시 아래 [검사 결과]에 있는 정보만 사용하세요**
2. **[검사 결과]에 없는 내용은 절대 답변하지 마세요**
3. **일반적인 설명이나 추측을 하지 마세요**
4. **[검사 결과]가 비어있거나 관련 정보가 없으면 "검사 결과에서 해당 정보를 찾을 수 없습니다"라고 답변하세요**

📋 **용어 구분**
- "성향" = [성향] 태그가 붙은 내용만 (예: 진취형, 창조형, 제작형 등)
- "사고력" = [사고력] 태그가 붙은 내용만 (예: 창의적사고력, 수직적사고력 등)
- "역량" 또는 "재능" = [역량] 태그가 붙은 내용만
- "직업" = [직업] 태그가 붙은 내용만

✅ **답변 방법**
1. [검사 결과]에서 질문과 관련된 태그를 찾으세요
2. 해당 태그의 내용을 그대로 사용하여 답변하세요
3. 구체적인 이름, 점수, 순위 등을 정확히 언급하세요
4. 친근하고 공감적인 톤을 유지하세요

❌ **절대 하지 말 것**
- 일반적인 성향/사고력 설명을 만들어내지 마세요
- [검사 결과]에 없는 정보를 추가하지 마세요
- 다른 태그의 내용을 혼동하지 마세요"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """--- [검사 결과] ---
{context}

--- [질문] ---
{question}

위 [검사 결과]에 있는 정보만 사용하여 답변하세요. 결과에 없는 내용은 답변하지 마세요.""")
])


class SimpleChatMessageHistory(BaseChatMessageHistory):
    """간단한 PostgreSQL 기반 대화 기록 관리"""
    
    def __init__(self, session_id: str, anp_seq: int):
        self.session_id = session_id
        self.anp_seq = anp_seq
        self._messages: List[BaseMessage] = []
    
    async def aget_messages(self) -> List[BaseMessage]:
        """비동기로 대화 기록 가져오기"""
        conn = await asyncpg.connect(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
        try:
            rows = await conn.fetch(
                """
                SELECT message_type, content 
                FROM chat_history 
                WHERE session_id = $1 AND anp_seq = $2
                ORDER BY created_at ASC
                LIMIT 10
                """,
                self.session_id,
                self.anp_seq
            )
            
            messages = []
            for row in rows:
                if row['message_type'] == 'human':
                    messages.append(HumanMessage(content=row['content']))
                else:
                    messages.append(AIMessage(content=row['content']))
            
            return messages
        finally:
            await conn.close()
    
    async def aadd_message(self, message: BaseMessage) -> None:
        """비동기로 메시지 추가"""
        message_type = 'human' if isinstance(message, HumanMessage) else 'ai'
        
        conn = await asyncpg.connect(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
        try:
            await conn.execute(
                """
                INSERT INTO chat_history (session_id, anp_seq, message_type, content)
                VALUES ($1, $2, $3, $4)
                """,
                self.session_id,
                self.anp_seq,
                message_type,
                message.content
            )
        finally:
            await conn.close()
    
    def add_message(self, message: BaseMessage) -> None:
        """동기 메서드 (사용 안 함)"""
        raise NotImplementedError("Use aadd_message instead")
    
    def clear(self) -> None:
        """대화 기록 삭제 (사용 안 함)"""
        pass


class CustomRetriever:
    """사용자별 필터링된 커스텀 Retriever"""
    
    def __init__(self, anp_seq: int, language_code: str):
        self.anp_seq = anp_seq
        self.language_code = language_code
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.google_api_key
        )
    
    async def ainvoke(self, question: str) -> List[Document]:
        """비동기로 관련 문서 검색 - 키워드 기반 필터링 + 벡터 검색"""
        import json
        
        # 질문에서 키워드 추출
        question_lower = question.lower()
        filters = []
        
        # 키워드 매칭 (누적 적용)
        if '성향' in question or '유형' in question:
            filters.extend(['top_tendency', 'top_tendency_explain', 'bottom_tendency'])
            print(f"🔍 키워드 매칭: '성향' 관련")
            
        if '사고' in question or '사고력' in question or '사고유형' in question:
            filters.extend(['thinking_main', 'thinking_detail'])
            print(f"🔍 키워드 매칭: '사고력' 관련")
            
        if '재능' in question or '역량' in question or '능력' in question:
            filters.extend(['talent'])
            print(f"🔍 키워드 매칭: '역량' 관련")
            
        if '직업' in question or '진로' in question or '직무' in question:
            filters.extend(['suitable_job', 'competency_job', 'duty'])
            print(f"🔍 키워드 매칭: '직업' 관련")
            
        if '학습' in question or '공부' in question:
            filters.extend(['learning_style'])
            print(f"🔍 키워드 매칭: '학습' 관련")
            
        # 중복 제거
        chunk_type_filter = list(set(filters)) if filters else None
        
        # 임베딩 생성 (필터링 여부와 관계없이 정렬을 위해 필요)
        question_embedding = self.embeddings.embed_query(question)
        embedding_str = '[' + ','.join(map(str, question_embedding)) + ']'
        
        # PostgreSQL에서 검색
        conn = await asyncpg.connect(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
        try:
            if chunk_type_filter:
                # 키워드 필터링 + 벡터 유사도 정렬
                rows = await conn.fetch(
                    """
                    SELECT content, metadata, chunk_type
                    FROM report_chunks
                    WHERE anp_seq = $1 AND language_code = $2
                      AND chunk_type = ANY($3)
                    ORDER BY embedding <=> $4::vector
                    LIMIT 15
                    """,
                    self.anp_seq,
                    self.language_code,
                    chunk_type_filter,
                    embedding_str
                )
                print(f"📊 키워드 필터링+벡터 검색 결과: {len(rows)}개 청크 검색됨")
            else:
                # 순수 벡터 검색 (필터 없음)
                rows = await conn.fetch(
                    """
                    SELECT content, metadata, chunk_type
                    FROM report_chunks
                    WHERE anp_seq = $1 AND language_code = $2
                    ORDER BY embedding <=> $3::vector
                    LIMIT 15
                    """,
                    self.anp_seq,
                    self.language_code,
                    embedding_str
                )
                print(f"📊 전체 벡터 검색 결과: {len(rows)}개 청크 검색됨")
            
            # 디버깅: 검색된 청크 타입 출력
            print(f"\n{'='*60}")
            print(f"🔎 질문: {question}")
            print(f"🎯 적용된 필터: {chunk_type_filter if chunk_type_filter else '없음 (전체 검색)'}")
            print(f"{'='*60}")
            for i, row in enumerate(rows, 1):
                content_preview = row['content'][:100].replace('\n', ' ')
                print(f"{i}. [{row['chunk_type']}] {content_preview}...")
            print(f"{'='*60}\n")
            
            # Document 객체로 변환
            documents = []
            for row in rows:
                # metadata가 문자열이면 JSON 파싱
                metadata = row['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                elif metadata is None:
                    metadata = {}
                
                documents.append(
                    Document(page_content=row['content'], metadata=metadata)
                )
            
            return documents
        finally:
            await conn.close()
    
    def invoke(self, question: str) -> List[Document]:
        """동기 메서드 (사용 안 함)"""
        raise NotImplementedError("Use ainvoke instead")


def format_docs(docs: List[Document]) -> str:
    """문서 리스트를 문자열로 포맷"""
    return "\n\n".join(doc.page_content for doc in docs)


async def get_rag_chain_with_history(request: ChatRequest):
    """LangChain LCEL 스타일의 RAG 체인을 생성합니다."""
    from langchain_core.runnables import RunnableLambda, RunnableParallel
    
    llm = get_chat_llm(request.provider)
    retriever = CustomRetriever(request.anp_seq, request.language_code)
    history = SimpleChatMessageHistory(request.session_id, request.anp_seq)
    
    # 대화 기록 가져오기
    messages = await history.aget_messages()
    
    # LCEL 체인 구성
    # 1. 컨텍스트 검색
    retrieve_chain = RunnableLambda(lambda x: retriever.ainvoke(x["question"]))
    
    # 2. 문서 포맷팅
    format_chain = RunnableLambda(lambda docs: format_docs(docs))
    
    # 3. 프롬프트 + LLM
    async def generate_answer(inputs):
        # 컨텍스트 검색 및 포맷
        docs = await retriever.ainvoke(inputs["question"])
        context = format_docs(docs)
        
        # 프롬프트 생성
        final_prompt = prompt.format_messages(
            chat_history=messages,
            context=context,
            question=inputs["question"]
        )
        
        # LLM 호출
        response = await llm.ainvoke(final_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    
    # 체인 실행
    answer = await generate_answer({"question": request.question})
    
    # 대화 기록 저장
    await history.aadd_message(HumanMessage(content=request.question))
    await history.aadd_message(AIMessage(content=answer))
    
    return answer
