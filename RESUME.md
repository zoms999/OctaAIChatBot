# 이력서: 개인화된 적성검사 RAG 챗봇 프로젝트

## 프로젝트 정보

| 항목 | 내용 |
|------|------|
| **프로젝트명** | 개인화된 적성검사 결과 상담 RAG 챗봇 시스템 |
| **개발 기간** | 2024년 |
| **팀 구성** | 1인 개발 (Full-Stack) |
| **프로젝트 규모** | 중규모 AI 챗봇 시스템 |
| **배포 환경** | 로컬/클라우드 (확장 가능) |

---

## 프로젝트 개요

적성검사 결과를 기반으로 사용자에게 개인화된 상담을 제공하는 AI 챗봇 시스템입니다. RAG(Retrieval-Augmented Generation) 기술을 활용하여 사용자별 검사 결과를 벡터 데이터베이스에 저장하고, 질문에 대해 맥락에 맞는 정확한 답변을 생성합니다.

---

## 담당 업무 및 역할

### 1. 시스템 설계 및 아키텍처 구축
- RAG 파이프라인 설계 및 구현
- FastAPI 기반 RESTful API 서버 구축
- PostgreSQL + pgvector 벡터 데이터베이스 설계
- 비동기 처리 아키텍처 설계

### 2. Backend 개발
- FastAPI를 활용한 고성능 API 서버 개발
- 비동기 프로그래밍으로 높은 동시성 처리 구현
- Pydantic을 활용한 데이터 검증 및 모델 정의
- CORS 설정 및 보안 구현

### 3. AI/ML 통합
- LangChain을 활용한 RAG 파이프라인 구축
- Google Gemini 1.5 Flash 및 OpenAI GPT-4o 통합
- Google text-embedding-004 임베딩 모델 적용
- 대화 기록 관리 및 컨텍스트 유지 시스템 구현

### 4. 데이터베이스 설계 및 최적화
- PostgreSQL 스키마 설계
- pgvector 확장을 활용한 벡터 검색 구현
- 사용자별 데이터 격리 및 인덱싱 전략 수립
- 비동기 데이터베이스 연결 구현 (asyncpg)

### 5. 테스트 및 검증
- API 테스트 스크립트 작성
- 데이터 검증 도구 개발
- Swagger UI를 통한 API 문서화
- 헬스 체크 엔드포인트 구현

---

## 사용 기술

### Backend & Framework
- **Python 3.x** - 주 개발 언어
- **FastAPI** - 고성능 비동기 웹 프레임워크
- **Uvicorn** - ASGI 서버
- **Pydantic** - 데이터 검증 및 설정 관리

### AI & Machine Learning
- **LangChain** - RAG 파이프라인 프레임워크
- **Google Gemini 2.0 Flash** - 주 LLM (비용 효율적)
- **OpenAI GPT-4o** - 선택적 LLM
- **Google text-embedding-004** - 임베딩 모델 (768차원)
- **langchain-google-genai** - Google AI 통합
- **langchain-openai** - OpenAI 통합

### Database
- **PostgreSQL** - 관계형 데이터베이스
- **pgvector** - 벡터 유사도 검색 확장
- **asyncpg** - 비동기 PostgreSQL 드라이버
- **psycopg** - PostgreSQL 어댑터

### Development Tools
- **python-dotenv** - 환경 변수 관리
- **Git** - 버전 관리

---

## 주요 성과 및 기여

### 1. 기술적 성과

#### RAG 시스템 구현
- 벡터 데이터베이스를 활용한 의미 기반 검색 구현
- 검색된 컨텍스트와 대화 기록을 결합한 정확한 답변 생성
- 사용자별 데이터 완전 격리로 개인정보 보호

#### 성능 최적화
- 비동기 프로그래밍으로 높은 동시 요청 처리 능력 확보
- pgvector 인덱스 활용으로 빠른 벡터 검색 구현
- 효율적인 청크 분할로 토큰 사용량 최소화

#### 비용 효율성
- Google Gemini를 기본 LLM으로 사용하여 비용 절감
- OpenAI GPT-4o를 선택적으로 사용 가능한 유연한 구조
- 임베딩 캐싱 및 재사용 전략

### 2. 기능 구현

#### 데이터 인덱싱 시스템
- 사용자별 적성검사 결과 자동 벡터화
- 다국어 지원 (language_code 기반)
- 기존 데이터 자동 삭제 및 재인덱싱

#### 대화 시스템
- Session ID 기반 대화 기록 관리
- 컨텍스트를 유지하는 연속적인 대화 지원
- 사용자별 맞춤형 답변 생성

#### API 설계
- RESTful API 설계 원칙 준수
- 명확한 엔드포인트 구조 (`/index-report`, `/chat`, `/health`)
- Swagger UI를 통한 자동 API 문서화

### 3. 품질 관리

#### 에러 핸들링
- 체계적인 예외 처리 및 에러 메시지
- HTTP 상태 코드 적절한 활용
- 사용자 친화적인 에러 응답

#### 테스트 및 검증
- API 기능 테스트 스크립트
- 데이터 무결성 검증 도구
- 임베딩 차원 확인 도구

#### 코드 품질
- 모듈화된 서비스 레이어 구조
- 타입 힌팅 및 Pydantic 모델 활용
- 명확한 함수 및 변수 네이밍

---

## 핵심 구현 내용

### 1. RAG 파이프라인

```
사용자 질문 입력
    ↓
질문 벡터화 (Google text-embedding-004)
    ↓
벡터 유사도 검색 (PostgreSQL + pgvector)
    ↓
관련 문서 검색 (사용자별 필터링)
    ↓
대화 기록 + 검색 결과 결합
    ↓
LLM 답변 생성 (Google Gemini / OpenAI GPT-4o)
    ↓
답변 반환 및 대화 기록 저장
```

### 2. 데이터베이스 스키마

**report_chunks 테이블**
- 벡터화된 적성검사 결과 저장
- anp_seq 및 language_code로 사용자 데이터 격리
- 768차원 벡터 임베딩 저장

**chat_history 테이블**
- 세션별 대화 기록 저장
- 사용자 질문 및 AI 답변 기록
- 시간순 정렬 및 컨텍스트 관리

### 3. API 엔드포인트

#### POST /index-report
- 적성검사 결과 데이터 인덱싱
- 기존 데이터 삭제 및 새로운 벡터 생성
- 청크 수 반환

#### POST /chat
- 사용자 질문 처리
- 관련 문서 검색 및 답변 생성
- 대화 기록 저장

#### GET /health
- 서비스 상태 확인
- 모니터링 및 헬스 체크

---

## 프로젝트를 통해 배운 점

### 기술적 학습

1. **AI/ML 기술**
   - RAG(Retrieval-Augmented Generation) 아키텍처 이해 및 구현
   - 임베딩 모델 활용 및 벡터 검색 최적화
   - LLM API 통합 및 프롬프트 엔지니어링

2. **Backend 개발**
   - FastAPI를 활용한 고성능 비동기 API 개발
   - Python 비동기 프로그래밍 (async/await)
   - RESTful API 설계 및 문서화

3. **데이터베이스**
   - PostgreSQL 고급 기능 활용
   - 벡터 데이터베이스 설계 및 최적화
   - 비동기 데이터베이스 연결 관리

4. **시스템 설계**
   - 확장 가능한 아키텍처 설계
   - 데이터 격리 및 보안 고려
   - 비용 효율적인 시스템 구축

### 문제 해결 경험

1. **벡터 검색 최적화**
   - 문제: 대량의 벡터 데이터에서 검색 속도 저하
   - 해결: pgvector 인덱스 활용 및 쿼리 최적화

2. **대화 컨텍스트 관리**
   - 문제: 긴 대화에서 컨텍스트 유지 어려움
   - 해결: 세션 기반 대화 기록 저장 및 관리

3. **비용 최적화**
   - 문제: OpenAI API 비용 부담
   - 해결: Google Gemini를 기본 LLM으로 사용, 선택적 OpenAI 활용

---

## 향후 발전 방향

1. **기능 확장**
   - 실시간 스트리밍 응답 구현
   - 멀티모달 입력 지원 (이미지, 파일)
   - 대화 요약 및 추천 질문 기능

2. **성능 개선**
   - Redis 캐싱 시스템 도입
   - 응답 속도 최적화
   - 벡터 인덱스 튜닝

3. **운영 고도화**
   - 모니터링 및 로깅 시스템 구축
   - CI/CD 파이프라인 구축
   - 프로덕션 배포 및 운영

---

## 프로젝트 링크

- **GitHub Repository**: [저장소 링크]
- **API Documentation**: http://localhost:8000/docs
- **프로젝트 문서**: [README.md](./README.md)

---

*작성일: 2024년 12월*
