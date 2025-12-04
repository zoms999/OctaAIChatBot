"""
사용자의 챗봇 데이터를 재인덱싱하는 스크립트
업데이트된 태그 형식을 적용하기 위해 실행하세요
"""
import asyncio
import sys
from app.services.indexing_service import index_report_data_for_anp_seq

async def reindex_user(anp_seq: int, language_code: str = 'ko-KR'):
    """특정 사용자의 데이터를 재인덱싱합니다"""
    print(f"🔄 anp_seq={anp_seq}, language={language_code} 재인덱싱 시작...")
    
    try:
        records_count, chunks_count = await index_report_data_for_anp_seq(anp_seq, language_code)
        print(f"✅ 재인덱싱 완료!")
        print(f"   - 처리된 레코드: {records_count}개")
        print(f"   - 생성된 청크: {chunks_count}개")
        return True
    except Exception as e:
        print(f"❌ 재인덱싱 실패: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python reindex_user.py <anp_seq> [language_code]")
        print("예시: python reindex_user.py 19719 ko-KR")
        sys.exit(1)
    
    anp_seq = int(sys.argv[1])
    language_code = sys.argv[2] if len(sys.argv) > 2 else 'ko-KR'
    
    asyncio.run(reindex_user(anp_seq, language_code))
