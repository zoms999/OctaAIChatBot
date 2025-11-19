import asyncio
import asyncpg
import json

async def check_data():
    conn = await asyncpg.connect("postgresql://hongsam:1111@127.0.0.1:5432/testoc")
    try:
        # Check if data exists for anp_seq=19937
        query = """
            SELECT anp_seq, language_code, tendency_view, competency_view, 
                   LENGTH(report_data::text) as data_length,
                   report_data
            FROM mwd_report_data 
            WHERE anp_seq = 19937
        """
        records = await conn.fetch(query)
        
        print(f"Found {len(records)} records for anp_seq=19937:")
        for record in records:
            print(f"\n  anp_seq: {record['anp_seq']}")
            print(f"  language_code: {record['language_code']}")
            print(f"  tendency_view: {record['tendency_view']}")
            print(f"  competency_view: {record['competency_view']}")
            print(f"  data_length: {record['data_length']}")
            
            # Parse report_data
            report_data = record['report_data']
            if isinstance(report_data, str):
                report_data = json.loads(report_data)
            
            print(f"\n  Report data keys: {list(report_data.keys())}")
            print(f"  topTendencies count: {len(report_data.get('topTendencies', []))}")
            print(f"  talentDetails count: {len(report_data.get('talentDetails', []))}")
            print(f"  suitableJobsDetail count: {len(report_data.get('suitableJobsDetail', []))}")
        
        # Check report_chunks table
        chunks_query = "SELECT COUNT(*) as count FROM report_chunks WHERE anp_seq = 19937"
        chunks_count = await conn.fetchval(chunks_query)
        print(f"\n\nChunks in report_chunks table: {chunks_count}")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_data())
