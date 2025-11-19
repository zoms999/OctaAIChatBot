import asyncio
import asyncpg

async def create_tables():
    """데이터베이스 테이블 생성"""
    conn = await asyncpg.connect("postgresql://hongsam:1111@127.0.0.1:5432/testoc")
    
    try:
        print("Creating pgvector extension...")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        print("Creating report_chunks table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS report_chunks (
                id SERIAL PRIMARY KEY,
                anp_seq INTEGER NOT NULL,
                language_code VARCHAR(10) NOT NULL,
                chunk_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                embedding vector(768),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        print("Creating chat_history table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                anp_seq INTEGER NOT NULL,
                message_type VARCHAR(10) NOT NULL CHECK (message_type IN ('human', 'ai')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        print("Creating indexes...")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_report_chunks_anp_seq ON report_chunks(anp_seq);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_report_chunks_language ON report_chunks(language_code);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id, anp_seq);")
        
        print("✅ All tables created successfully!")
        
        # 테이블 확인
        tables = await conn.fetch("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('report_chunks', 'chat_history')
        """)
        
        print(f"\n📋 Created tables: {[t['table_name'] for t in tables]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(create_tables())
