-- pgvector 확장 활성화 (이미 있으면 무시됨)
CREATE EXTENSION IF NOT EXISTS vector;

-- report_chunks 테이블 생성
CREATE TABLE IF NOT EXISTS report_chunks (
    id SERIAL PRIMARY KEY,
    anp_seq INTEGER NOT NULL,
    language_code VARCHAR(10) NOT NULL,
    chunk_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(768),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_anp_seq (anp_seq),
    INDEX idx_language_code (language_code),
    INDEX idx_embedding_vector ON report_chunks USING ivfflat (embedding vector_cosine_ops)
);

-- chat_history 테이블 생성
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    anp_seq INTEGER NOT NULL,
    message_type VARCHAR(10) NOT NULL CHECK (message_type IN ('human', 'ai')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_session_anp (session_id, anp_seq),
    INDEX idx_created_at (created_at)
);

-- 인덱스 생성 확인
CREATE INDEX IF NOT EXISTS idx_report_chunks_anp_seq ON report_chunks(anp_seq);
CREATE INDEX IF NOT EXISTS idx_report_chunks_language ON report_chunks(language_code);
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id, anp_seq);
