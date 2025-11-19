import asyncio
import asyncpg

async def clear_chunks(anp_seq: int):
    """특정 anp_seq의 청크 삭제"""
    conn = await asyncpg.connect("postgresql://hongsam:1111@127.0.0.1:5432/testoc")
    try:
        result = await conn.execute("DELETE FROM report_chunks WHERE anp_seq = $1", anp_seq)
        print(f"✅ Deleted chunks for anp_seq={anp_seq}: {result}")
    finally:
        await conn.close()

if __name__ == "__main__":
    anp_seq = 19719  # 삭제할 anp_seq
    asyncio.run(clear_chunks(anp_seq))
    print(f"\n이제 다음 API를 호출하여 다시 인덱싱하세요:")
    print(f"curl -X 'POST' 'http://127.0.0.1:8000/index-report' -H 'Content-Type: application/json' -d '{{\"anp_seq\": {anp_seq}}}'")
