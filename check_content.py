import asyncio
import asyncpg

async def check_content():
    """청크의 실제 content 확인"""
    conn = await asyncpg.connect("postgresql://hongsam:1111@127.0.0.1:5432/testoc")
    try:
        # top_tendency 청크 확인
        rows = await conn.fetch("""
            SELECT chunk_type, content 
            FROM report_chunks 
            WHERE anp_seq = 19719 AND chunk_type = 'top_tendency'
            ORDER BY created_at
            LIMIT 3
        """)
        
        print("=== TOP TENDENCY 청크 ===")
        for row in rows:
            print(f"Type: {row['chunk_type']}")
            print(f"Content: {row['content']}")
            print()
        
        # thinking_main 청크 확인
        rows = await conn.fetch("""
            SELECT chunk_type, content 
            FROM report_chunks 
            WHERE anp_seq = 19719 AND chunk_type = 'thinking_main'
            LIMIT 1
        """)
        
        print("=== THINKING MAIN 청크 ===")
        for row in rows:
            print(f"Type: {row['chunk_type']}")
            print(f"Content: {row['content']}")
            print()
            
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_content())
