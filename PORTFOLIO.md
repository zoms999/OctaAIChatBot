# 📋 포트폴리오: 개인화된 적성검사 RAG 챗봇 시스템

## 🎯 프로젝트 개요

**프로젝트명**: 개인화된 적성검사 결과 상담 RAG 챗봇  
**개발 기간**: 2024  
**역할**: Full-Stack Developer  
**프로젝트 유형**: AI 기반 상담 챗봇 시스템

### 프로젝트 설명

사용자의 적성검사 결과를 기반으로 개인화된 상담을 제공하는 AI 챗봇 시스템입니다. RAG(Retrieval-Augmented Generation) 기술을 활용하여 각 사용자의 검사 결과 데이터를 벡터 데이터베이스에 저장하고, 질문에 대해 관련성 높은 정보를 검색하여 정확하고 맥락에 맞는 답변을 생성합니다.

---

## 🛠️ 기술 스택

### Backend & API
- **FastAPI** - 고성능 비동기 웹 프레임워크
- **Python 3.x** - 주 개발 언어
- **Uvicorn** - ASGI 서버

### AI & Machine Learning
- **LangChain** - RAG 파이프라인 구축 프레임워크
- **Google Gemini 2.0 Flash** - 주 LLM (비용 효율적)
- **OpenAI GPT-4o** - 선택적 LLM
- **Google text-embedding-004** - 임베딩 모델 (768차원)

### Database & Vector Store
- **PostgreSQL** - 관계형 데이터베이스
- **pgvector** - 벡터 유사도 검색 확장
- **asyncpg** - 비동기 PostgreSQL 드라이버

### Development Tools
- **Pydantic** - 데이터 검증 및 설정 관리
- **python-dotenv** - 환경 변수 관리

---

## 🏗️ 시스템 아키텍처

### 전체 구조

```
┌─────────────┐
│   Client    │
│  (Frontend) │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────────────────┐
│      FastAPI Backend            │
│  ┌──────────────────────────┐  │
│  │   API Endpoints          │  │
│  │  - /index-report         │  │
│  │  - /chat                 │  │
│  │  - /health               │  │
│  └──────────┬───────────────┘  │
│             │                   │
│  ┌──────────▼───────────────┐  │
│  │   Services Layer         │  │
│  │  - Indexing Service      │  │
│  │  - Chat Service          │  │
│  └──────────┬───────────────┘  │
└─────────────┼───────────────────┘
              │
    ┌─────────┴──────────┐
    ▼                    ▼
┌─────────┐      ┌──────────────┐
│ LangChain│      │ PostgreSQL   │
│  RAG     │◄────►│  + pgvector  │
│ Pipeline │      │              │
└────┬────┘      └──────────────┘
     │
     ▼
┌──────────────┐
│  LLM APIs    │
│ - Google AI  │
│ - OpenAI     │
└──────────────┘
```

### RAG 파이프라인

1. **데이터 인덱싱**
   - 적성검사 결과 데이터를 의미 단위로 청크 분할
   - Google text-embedding-004로 벡터화
   - PostgreSQL + pgvector에 저장

2. **질의 처리**
   - 사용자 질문을 벡터로 변환
   - 코사인 유사도 기반 관련 문서 검색
   - 대화 기록과 함께 LLM에 전달

3. **답변 생성**
   - 검색된 컨텍스트와 대화 기록 활용
   - LLM이 개인화된 답변 생성
   - 대화 기록 저장 및 세션 관리

---

## 💡 주요 기능 및 구현 내용

### 1. 개인화된 데이터 인덱싱

**엔드포인트**: `POST /index-report`

```python
{
  "anp_seq": 12345,
  "language_code": "ko-KR"
}
```

**구현 특징**:
- 사용자별 고유 식별자(anp_seq)로 데이터 격리
- 다국어 지원 (language_code)
- 기존 데이터 자동 삭제 후 재인덱싱
- 청크 단위 벡터화로 검색 정확도 향상

### 2. 컨텍스트 기반 대화 시스템

**엔드포인트**: `POST /chat`

```python
{
  "session_id": "user1-session-abc-123",
  "anp_seq": 12345,
  "question": "저의 가장 중요한 성향은 무엇인가요?",
  "language_code": "ko-KR",
  "provider": "google"
}
```

**구현 특징**:
- Session ID 기반 대화 기록 관리
- 사용자별 데이터만 검색 (데이터 격리)
- LLM 프로바이더 선택 가능 (Google/OpenAI)
- 비동기 처리로 높은 동시성 지원

### 3. 벡터 유사도 검색

**기술적 구현**:
- **임베딩 모델**: Google text-embedding-004 (768차원)
- **검색 알고리즘**: 코사인 유사도
- **필터링**: anp_seq + language_code 복합 조건
- **최적화**: pgvector 인덱스 활용

### 4. 대화 기록 관리

**데이터베이스 스키마**:

```sql
-- 벡터 저장소
CREATE TABLE report_chunks (
    id SERIAL PRIMARY KEY,
    anp_seq INTEGER NOT NULL,
    language_code VARCHAR(10) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 대화 기록
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    anp_seq INTEGER NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🎨 프로젝트 구조

```
chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI 앱 및 라우팅
│   ├── config.py                  # 환경 설정
│   ├── models.py                  # Pydantic 데이터 모델
│   └── services/
│       ├── __init__.py
│       ├── indexing_service.py    # 데이터 인덱싱 로직
│       └── chat_service.py        # RAG 체인 및 대화 처리
├── .env                           # 환경 변수
├── requirements.txt               # 의존성 관리
├── create_tables.sql              # DB 스키마
├── init_db.py                     # DB 초기화 스크립트
├── test_api.py                    # API 테스트
├── check_*.py                     # 데이터 검증 스크립트
└── README.md                      # 프로젝트 문서
```

---

## 🔧 핵심 기술 구현

### 1. 비동기 프로그래밍

```python
# FastAPI의 비동기 엔드포인트
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    answer = await get_rag_chain_with_history(request)
    return ChatResponse(answer=answer, session_id=request.session_id)
```

**장점**:
- 높은 동시 요청 처리 능력
- I/O 대기 시간 최소화
- 효율적인 리소스 활용

### 2. RAG 파이프라인 구축

**LangChain을 활용한 구현**:
- Vector Store: PostgreSQL + pgvector
- Retriever: 유사도 기반 문서 검색
- Memory: 세션별 대화 기록 관리
- LLM: Google Gemini / OpenAI GPT-4o

### 3. 데이터 격리 및 보안

- 사용자별 데이터 완전 격리 (anp_seq 필터링)
- 세션 기반 대화 관리
- CORS 설정으로 허용된 도메인만 접근
- 환경 변수로 API 키 관리

### 4. 에러 핸들링

```python
try:
    answer = await get_rag_chain_with_history(request)
    return ChatResponse(answer=answer, session_id=request.session_id)
except Exception as e:
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Error processing chat request: {str(e)}"
    )
```

---

## 📊 성능 및 최적화

### 벡터 검색 최적화
- pgvector 인덱스 활용으로 빠른 유사도 검색
- 768차원 임베딩으로 정확도와 성능 균형

### 비용 최적화
- 기본 LLM으로 Google Gemini 1.5 Flash 사용
- OpenAI GPT-4o는 선택적 사용
- 효율적인 청크 분할로 토큰 사용량 최소화

### 확장성
- 비동기 처리로 높은 동시성 지원
- 데이터베이스 연결 풀링
- 수평적 확장 가능한 아키텍처

---

## 🧪 테스트 및 검증

### API 테스트
- `test_api.py`: 엔드포인트 기능 검증
- Swagger UI (`/docs`): 대화형 API 문서 및 테스트

### 데이터 검증
- `check_chunks.py`: 벡터 청크 검증
- `check_embeddings.py`: 임베딩 차원 확인
- `check_data.py`: 데이터 무결성 검사

### 헬스 체크
- `/health` 엔드포인트로 서비스 상태 모니터링

---

## 🚀 배포 및 운영

### 로컬 개발 환경

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
# .env 파일에 DATABASE_URL, GOOGLE_API_KEY 등 설정

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

### 프로덕션 고려사항
- CORS 설정에 프로덕션 도메인 추가
- 환경 변수 보안 관리
- 로깅 및 모니터링 시스템 구축
- 데이터베이스 백업 전략

---

## 📈 프로젝트 성과

### 기술적 성과
✅ RAG 기술을 활용한 정확한 개인화 답변 생성  
✅ 비동기 프로그래밍으로 높은 동시성 처리  
✅ 벡터 데이터베이스를 활용한 효율적인 검색  
✅ 다국어 지원 및 확장 가능한 아키텍처  
✅ LLM 프로바이더 선택으로 비용 최적화  

### 학습 및 성장
- **AI/ML**: LangChain, RAG 파이프라인, 임베딩 모델 활용
- **Backend**: FastAPI, 비동기 프로그래밍, RESTful API 설계
- **Database**: PostgreSQL, pgvector, 벡터 검색 최적화
- **DevOps**: 환경 설정 관리, API 문서화, 테스트 자동화

---

## 🔮 향후 개선 계획

1. **기능 확장**
   - 음성 인터페이스 추가
   - 멀티모달 입력 지원 (이미지, 파일)
   - 실시간 스트리밍 응답

2. **성능 개선**
   - 캐싱 시스템 도입
   - 응답 속도 최적화
   - 벡터 인덱스 튜닝

3. **사용자 경험**
   - 대화 요약 기능
   - 추천 질문 제시
   - 감정 분석 및 맞춤형 응답 톤

4. **운영 고도화**
   - 모니터링 대시보드
   - A/B 테스트 프레임워크
   - 사용자 피드백 수집 시스템

---

## 📞 연락처

**GitHub**: [프로젝트 저장소 링크]  
**이메일**: [이메일 주소]  
**포트폴리오**: [포트폴리오 웹사이트]

---

## 📝 라이선스

이 프로젝트는 개인 포트폴리오 목적으로 작성되었습니다.

---

*최종 업데이트: 2024년 12월*
