import asyncio
import asyncpg

async def check_chunks():
    """청크 내용 확인"""
    conn = await asyncpg.connect("postgresql://hongsam:1111@127.0.0.1:5432/testoc")
    try:
        # 청크 개수 확인
        count = await conn.fetchval("SELECT COUNT(*) FROM report_chunks WHERE anp_seq = 19719")
        print(f"Total chunks for anp_seq=19719: {count}")
        
        # 성향 관련 청크 확인
        tendency_chunks = await conn.fetch("""
            SELECT chunk_type, content 
            FROM report_chunks 
            WHERE anp_seq = 19719 AND chunk_type LIKE '%tendency%'
            LIMIT 5
        """)
        
        print("\n=== 성향 관련 청크 ===")
        for chunk in tendency_chunks:
            print(f"Type: {chunk['chunk_type']}")
            print(f"Content: {chunk['content'][:200]}...")
            print()
        
        # 사고력 관련 청크 확인
        thinking_chunks = await conn.fetch("""
            SELECT chunk_type, content 
            FROM report_chunks 
            WHERE anp_seq = 19719 AND chunk_type LIKE '%thinking%'
            LIMIT 3
        """)
        
        print("\n=== 사고력 관련 청크 ===")
        for chunk in thinking_chunks:
            print(f"Type: {chunk['chunk_type']}")
            print(f"Content: {chunk['content'][:200]}...")
            print()
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_chunks())
