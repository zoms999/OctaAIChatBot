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
    ("system", """당신은 친절하고 명확한 적성검사 결과 상담 챗봇입니다. 
주어진 [검사 결과]와 [대화 기록]을 바탕으로 사용자의 [질문]에 대해 답변해주세요. 

⚠️ 중요: 용어를 정확히 구분하세요!
- "성향" 또는 "성향 유형" = [성향] 태그가 붙은 내용 (예: 진취형, 창조형, 제작형, 복합형 등)
- "사고력" 또는 "사고 유형" = [사고력] 태그가 붙은 내용 (예: 창의적사고력, 수직적사고력 등)
- "역량" 또는 "재능" = [역량] 태그가 붙은 내용 (예: 문서능력, 음악감각 등)
- "직업" = [직업] 태그가 붙은 내용

답변 규칙:
1. 사용자가 "성향"을 물어보면 반드시 [성향] 태그가 붙은 내용만 답변하세요.
2. 사용자가 "사고력"을 물어보면 [사고력] 태그가 붙은 내용만 답변하세요.
3. 검사 결과의 태그를 확인하고, 질문에 맞는 태그의 내용만 사용하세요.
4. 검사 결과에 기반하여 구체적이고 명확하게 답변하세요.
5. 친근하고 공감적인 톤으로 답변하세요."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """--- [검사 결과] ---
{context}

--- [질문] ---
{question}

답변 시 검사 결과의 태그([성향], [사고력], [역량] 등)를 확인하고, 질문에 맞는 내용만 답변하세요.""")
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
        chunk_type_filter = None
        
        if '성향' in question:
            chunk_type_filter = ['top_tendency', 'top_tendency_explain', 'bottom_tendency']
        elif '사고' in question or '사고력' in question:
            chunk_type_filter = ['thinking_main', 'thinking_detail']
        elif '재능' in question or '역량' in question:
            chunk_type_filter = ['talent']
        elif '직업' in question:
            chunk_type_filter = ['suitable_job', 'competency_job', 'duty']
        
        # PostgreSQL에서 검색
        conn = await asyncpg.connect(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
        try:
            if chunk_type_filter:
                # 키워드 필터링 적용
                rows = await conn.fetch(
                    """
                    SELECT content, metadata, chunk_type
                    FROM report_chunks
                    WHERE anp_seq = $1 AND language_code = $2
                      AND chunk_type = ANY($3)
                    LIMIT 10
                    """,
                    self.anp_seq,
                    self.language_code,
                    chunk_type_filter
                )
            else:
                # 벡터 검색
                question_embedding = self.embeddings.embed_query(question)
                embedding_str = '[' + ','.join(map(str, question_embedding)) + ']'
                
                rows = await conn.fetch(
                    """
                    SELECT content, metadata, chunk_type
                    FROM report_chunks
                    WHERE anp_seq = $1 AND language_code = $2
                    ORDER BY embedding <=> $3::vector
                    LIMIT 10
                    """,
                    self.anp_seq,
                    self.language_code,
                    embedding_str
                )
            
            # 디버깅: 검색된 청크 타입 출력
            print(f"\n=== 검색된 청크 (질문: {question}, 필터: {chunk_type_filter}) ===")
            for i, row in enumerate(rows, 1):
                print(f"{i}. {row['chunk_type']}: {row['content'][:100]}...")
            print("=" * 50 + "\n")
            
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
