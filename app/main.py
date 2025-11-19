from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.models import IndexRequest, IndexResponse, ChatRequest, ChatResponse
from app.services.indexing_service import index_report_data_for_anp_seq
from app.services.chat_service import get_rag_chain_with_history

app = FastAPI(
    title="Personalized Aptitude Test RAG Chatbot",
    description="개인화된 적성검사 결과 기반 RAG 챗봇 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js 개발 서버
        "http://localhost:3002",  # Next.js 개발 서버 (대체 포트)
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3002",
        # 프로덕션 도메인 추가 필요
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "Personalized Aptitude Test RAG Chatbot API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/index-report", response_model=IndexResponse)
async def create_index_for_report(request: IndexRequest):
    """
    보고서 데이터를 벡터 DB에 인덱싱합니다.
    
    - **anp_seq**: 인덱싱할 사용자의 anp_seq
    - **language_code**: 언어 코드 (기본값: ko-KR)
    """
    try:
        _, total_chunks = await index_report_data_for_anp_seq(request.anp_seq, request.language_code)
        return IndexResponse(
            success=True,
            message=f"Indexing completed for anp_seq={request.anp_seq}, language={request.language_code}.",
            anp_seq=request.anp_seq,
            total_chunks_created=total_chunks
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing error: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    """
    사용자 질문에 답변하고 대화 기록을 관리합니다.
    
    - **session_id**: 대화 세션 ID
    - **anp_seq**: 사용자의 anp_seq
    - **question**: 사용자 질문
    - **language_code**: 언어 코드 (기본값: ko-KR)
    - **provider**: LLM 프로바이더 (openai 또는 google, 기본값: google)
    """
    try:
        answer = await get_rag_chain_with_history(request)
        return ChatResponse(
            answer=answer,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing chat request: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
