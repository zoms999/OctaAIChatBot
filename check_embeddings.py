import asyncio
import asyncpg

async def check_embeddings():
    """임베딩 벡터 확인"""
    conn = await asyncpg.connect("postgresql://hongsam:1111@127.0.0.1:5432/testoc")
    try:
        # 임베딩이 NULL인 청크 확인
        null_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM report_chunks 
            WHERE anp_seq = 19719 AND embedding IS NULL
        """)
        print(f"임베딩이 NULL인 청크: {null_count}개")
        
        # 전체 청크 수
        total_count = await conn.fetchval("""
            SELECT COUNT(*) 
            FROM report_chunks 
            WHERE anp_seq = 19719
        """)
        print(f"전체 청크: {total_count}개")
        
        # top_tendency 청크의 임베딩 확인
        rows = await conn.fetch("""
            SELECT chunk_type, 
                   CASE WHEN embedding IS NULL THEN 'NULL' ELSE 'EXISTS' END as emb_status,
                   content
            FROM report_chunks 
            WHERE anp_seq = 19719 AND chunk_type = 'top_tendency'
        """)
        
        print("\n=== TOP_TENDENCY 청크 임베딩 상태 ===")
        for row in rows:
            print(f"Type: {row['chunk_type']}, Embedding: {row['emb_status']}")
            print(f"Content: {row['content'][:100]}...")
            print()
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_embeddings())
