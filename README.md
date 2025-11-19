# 개인화된 적성검사 RAG 챗봇

Google의 임베딩 모델을 기반으로 한 개인화된 적성검사 결과 상담 챗봇 시스템입니다.

## 기술 스택

- **백엔드 API**: FastAPI
- **데이터베이스**: PostgreSQL + pgvector
- **RAG 프레임워크**: LangChain
- **임베딩 모델**: Google text-embedding-004
- **LLM**: Google Gemini 1.5 Flash (기본) / OpenAI GPT-4o (선택)

## 설치 및 실행

### 1. 의존성 설치

```bash
cd chatbot
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 수정하여 필요한 API 키와 데이터베이스 정보를 입력합니다:

```env
DATABASE_URL="postgresql+asyncpg://postgres:1111@127.0.0.1:5433/myway2"
GOOGLE_API_KEY="your-google-api-key"
OPENAI_API_KEY="your-openai-api-key"  # 선택사항
DEFAULT_LLM_PROVIDER="google"
```

### 3. 서버 실행

```bash
uvicorn app.main:app --reload --port 8000
```

서버가 실행되면 다음 주소에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 사용법

### 1. 보고서 데이터 인덱싱

사용자의 적성검사 결과를 벡터 DB에 인덱싱합니다.

```bash
POST http://localhost:8000/index-report
Content-Type: application/json

{
  "anp_seq": 12345
}
```

**응답 예시:**
```json
{
  "success": true,
  "message": "Indexing completed for anp_seq=12345.",
  "anp_seq": 12345,
  "total_chunks_created": 45
}
```

### 2. 챗봇과 대화

인덱싱된 데이터를 기반으로 사용자 질문에 답변합니다.

```bash
POST http://localhost:8000/chat
Content-Type: application/json

{
  "session_id": "user1-session-abc-123",
  "anp_seq": 12345,
  "question": "저의 가장 중요한 성향은 무엇인가요?",
  "language_code": "ko-KR",
  "provider": "google"
}
```

**응답 예시:**
```json
{
  "answer": "당신님의 가장 중요한 성향은 '창의성'입니다. 이 성향은...",
  "session_id": "user1-session-abc-123"
}
```

## 주요 기능

1. **개인화된 검색**: 각 사용자의 anp_seq를 기반으로 해당 사용자의 데이터만 검색
2. **대화 기록 관리**: session_id를 통해 대화 맥락 유지
3. **다국어 지원**: language_code를 통한 언어별 데이터 처리
4. **LLM 선택**: Google Gemini 또는 OpenAI GPT-4o 중 선택 가능
5. **의미 기반 검색**: Google의 text-embedding-004 모델을 사용한 고품질 임베딩

## 프로젝트 구조

```
chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 앱 및 엔드포인트
│   ├── config.py            # 환경변수 설정
│   ├── models.py            # Pydantic 모델
│   └── services/
│       ├── __init__.py
│       ├── indexing_service.py  # 데이터 인덱싱 로직
│       └── chat_service.py      # 챗봇 응답 생성 로직
├── .env                     # 환경변수
├── requirements.txt         # 의존성
└── README.md
```

## 데이터베이스 스키마

시스템은 다음 두 개의 테이블을 사용합니다:

1. **report_chunks**: 벡터화된 보고서 청크 저장
2. **chat_history**: 대화 기록 저장

(테이블은 이미 생성되어 있음)

## 개발 노트

- 모든 개발은 `chatbot` 폴더 내에서만 진행
- DB 테이블은 이미 생성되어 있음
- Google API 키는 `.env` 파일에서 관리
- 기본 LLM은 Google Gemini (비용 효율적)
