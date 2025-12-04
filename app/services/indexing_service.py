import asyncpg
import json
from typing import Dict, Any, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from app.config import settings


class ReportChunk:
    def __init__(self, anp_seq: int, language_code: str, chunk_type: str, content: str, metadata: Dict[str, Any]):
        self.anp_seq = anp_seq
        self.language_code = language_code
        self.chunk_type = chunk_type
        self.content = content
        self.metadata = metadata


def create_chunks_from_report(report_record: Dict[str, Any]) -> List[ReportChunk]:
    """보고서 데이터를 의미 있는 청크로 분할합니다."""
    report_data = report_record.get('report_data', {})
    anp_seq = report_record.get('anp_seq')
    lang = report_record.get('language_code')
    
    # report_data가 문자열인 경우 JSON 파싱
    if isinstance(report_data, str):
        try:
            report_data = json.loads(report_data)
        except json.JSONDecodeError:
            print(f"Failed to parse report_data for anp_seq={anp_seq}")
            return []
    
    # report_data 안의 'data' 키에 실제 데이터가 있음
    if 'data' in report_data:
        report_data = report_data['data']
    
    chunks = []
    
    # 개인 정보
    personal_info = report_data.get('personalInfo', {})
    if personal_info:
        pname = personal_info.get('pname', '당신')
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='personal_info',
            content=f"{pname}님의 기본 정보: 성별 {personal_info.get('sex', '')}, 나이 {personal_info.get('age', '')}세",
            metadata={'source': 'personalInfo'}
        ))
    else:
        pname = '당신'
    
    # 주요 성향 (상위 3개)
    for tendency in report_data.get('topTendencies', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='top_tendency',
            content=f"[성향] {pname}님의 {tendency.get('rank', '')}순위 주요 성향: {tendency.get('tendency_name', '')}형",
            metadata={'source': 'topTendencies', 'rank': tendency.get('rank'), 'code': tendency.get('code')}
        ))
    
    # 주요 성향 상세 설명
    for tendency_explain in report_data.get('topTendencyExplains', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='top_tendency_explain',
            content=f"[성향] {pname}님의 {tendency_explain.get('rank', '')}순위 성향 '{tendency_explain.get('tendency_name', '')}형' 설명:\n{tendency_explain.get('explanation', '')}",
            metadata={'source': 'topTendencyExplains', 'rank': tendency_explain.get('rank')}
        ))
    
    # 하위 성향
    for tendency in report_data.get('bottomTendencies', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='bottom_tendency',
            content=f"[성향] {pname}님의 하위 성향 (순위: {tendency.get('rank', '')}): {tendency.get('tendency_name', '')}형",
            metadata={'source': 'bottomTendencies', 'rank': tendency.get('rank')}
        ))
    
    # 사고 유형
    thinking_main = report_data.get('thinkingMain', {})
    if thinking_main:
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='thinking_main',
            content=f"[사고력] {pname}님의 사고 유형:\n- 주요 사고: {thinking_main.get('thkm', '')}\n- 보조 사고: {thinking_main.get('thks', '')}",
            metadata={'source': 'thinkingMain'}
        ))
    
    # 사고 유형 상세
    for thinking in report_data.get('thinkingDetails', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='thinking_detail',
            content=f"[사고력] {pname}님의 '{thinking.get('qua_name', '')}' (점수: {thinking.get('score', 0)}점):\n{thinking.get('explain', '')}",
            metadata={'source': 'thinkingDetails'}
        ))
    
    # 적합 직업 (성향 기반)
    for job in report_data.get('suitableJobsDetail', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='suitable_job',
            content=f"[직업] {pname}님께 추천하는 직업: {job.get('jo_name', '')}\n개요: {job.get('jo_outline', '')}\n주요 업무: {job.get('jo_mainbusiness', '')}",
            metadata={'source': 'suitableJobsDetail'}
        ))
    
    # 역량 (재능)
    for talent in report_data.get('talentDetails', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='talent',
            content=f"[역량] {pname}님의 {talent.get('rank', '')}순위 역량: '{talent.get('qua_name', '')}' (점수: {talent.get('tscore', 0)}점, 백분위: {talent.get('percentile_rank', 0)}%)\n{talent.get('explain', '')}",
            metadata={'source': 'talentDetails', 'rank': talent.get('rank')}
        ))
    
    # 역량 기반 직업
    for job in report_data.get('competencyJobs', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='competency_job',
            content=f"{pname}님의 역량에 맞는 직업: {job.get('jo_name', '')}. 개요: {job.get('jo_outline', '')}. 관련 학과: {job.get('majors', '')}",
            metadata={'source': 'competencyJobs'}
        ))
    
    # IT 직무
    for duty in report_data.get('duties', []):
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='duty',
            content=f"{pname}님께 추천하는 IT 직무: {duty.get('du_name', '')}. 설명: {duty.get('du_content', '')}. 관련 학과: {duty.get('majors', '')}. 적합도: {duty.get('match_rate', 0)}%",
            metadata={'source': 'duties'}
        ))
    
    # 학습 스타일
    learning_style = report_data.get('learningStyle', {})
    if learning_style:
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='learning_style',
            content=f"{pname}님의 학습 스타일: {learning_style.get('tnd1', '')} - {learning_style.get('tnd1_study', '')}. 학습 방법: {learning_style.get('tnd1_way', '')}",
            metadata={'source': 'learningStyle'}
        ))
    
    # 선호도 반응 검사 결과
    preference_data = report_data.get('preferenceData', {})
    if preference_data:
        # 선호도 요약
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='preference_summary',
            content=f"[선호도] {pname}님의 선호도 반응 검사 결과 요약:\n전체 반응률: {preference_data.get('total_response_rate', 0)}%\n(전체 {preference_data.get('total_count', 0)}개 중 {preference_data.get('response_count', 0)}개 이미지에 반응)",
            metadata={'source': 'preferenceData'}
        ))

        # 1순위 선호 성향
        chunks.append(ReportChunk(
            anp_seq=anp_seq, language_code=lang,
            chunk_type='preference_top',
            content=f"[선호도] {pname}님의 1순위 선호 성향: {preference_data.get('tdname1', '')} (선호도: {preference_data.get('rrate1', 0)}%)\n설명: {preference_data.get('exp1', '')}",
            metadata={'source': 'preferenceData', 'rank': 1}
        ))

        # 2순위 선호 성향
        if preference_data.get('tdname2'):
            chunks.append(ReportChunk(
                anp_seq=anp_seq, language_code=lang,
                chunk_type='preference_top',
                content=f"[선호도] {pname}님의 2순위 선호 성향: {preference_data.get('tdname2', '')} (선호도: {preference_data.get('rrate2', 0)}%)\n설명: {preference_data.get('exp2', '')}",
                metadata={'source': 'preferenceData', 'rank': 2}
            ))

        # 3순위 선호 성향
        if preference_data.get('tdname3'):
            chunks.append(ReportChunk(
                anp_seq=anp_seq, language_code=lang,
                chunk_type='preference_top',
                content=f"[선호도] {pname}님의 3순위 선호 성향: {preference_data.get('tdname3', '')} (선호도: {preference_data.get('rrate3', 0)}%)\n설명: {preference_data.get('exp3', '')}",
                metadata={'source': 'preferenceData', 'rank': 3}
            ))
            
        # 선호 성향 기반 추천 직업 (데이터에 있는 경우)
        if preference_data.get('recommend_jobs'):
             chunks.append(ReportChunk(
                anp_seq=anp_seq, language_code=lang,
                chunk_type='preference_job',
                content=f"[선호도] {pname}님의 선호 성향 기반 추천 직업:\n{preference_data.get('recommend_jobs', '')}",
                metadata={'source': 'preferenceData'}
            ))
    
    return chunks


async def index_report_data_for_anp_seq(anp_seq: int, language_code: str = 'ko-KR') -> tuple[int, int]:
    """특정 anp_seq의 보고서 데이터를 벡터 DB에 인덱싱합니다."""
    # PostgreSQL에서 보고서 데이터 가져오기 (특정 언어만)
    conn = await asyncpg.connect(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
    try:
        query = """
            SELECT anp_seq, language_code, report_data 
            FROM mwd_report_data 
            WHERE anp_seq = $1 
              AND language_code = $2
              AND tendency_view = 'full' 
              AND competency_view = 'full'
        """
        report_records = await conn.fetch(query, anp_seq, language_code)
    finally:
        await conn.close()
    
    if not report_records:
        raise ValueError(f"anp_seq={anp_seq}, language_code={language_code}에 해당하는 'full'/'full' 뷰 타입의 리포트 데이터를 찾을 수 없습니다.")
    
    # 모든 레코드를 청크로 변환
    all_chunks: List[ReportChunk] = []
    for record in report_records:
        all_chunks.extend(create_chunks_from_report(dict(record)))
    
    if not all_chunks:
        return len(report_records), 0
    
    # Google 임베딩 모델 초기화
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.google_api_key
    )
    
    # 기존 데이터 삭제
    conn = await asyncpg.connect(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
    try:
        await conn.execute("DELETE FROM report_chunks WHERE anp_seq = $1", anp_seq)
    finally:
        await conn.close()
    
    # 배치로 임베딩 생성 (Google API 부하 분산)
    import time
    batch_size = 10
    successful_chunks = 0
    
    conn = await asyncpg.connect(settings.database_url.replace('postgresql+asyncpg://', 'postgresql://'))
    try:
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            for chunk in batch:
                try:
                    # 임베딩 생성
                    embedding = embeddings.embed_query(chunk.content)
                    
                    # PostgreSQL vector 형식으로 변환: [1.0, 2.0, 3.0]
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # PostgreSQL에 직접 삽입
                    await conn.execute(
                        """
                        INSERT INTO report_chunks (anp_seq, language_code, chunk_type, content, embedding, metadata)
                        VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb)
                        """,
                        chunk.anp_seq,
                        chunk.language_code,
                        chunk.chunk_type,
                        chunk.content,
                        embedding_str,  # vector 형식 문자열
                        json.dumps({**chunk.metadata})
                    )
                    successful_chunks += 1
                    
                except Exception as e:
                    print(f"Error embedding chunk: {str(e)[:100]}")
                    # 개별 청크 실패는 건너뛰고 계속 진행
                    continue
            
            # 배치 간 짧은 대기 (API rate limit 방지)
            if i + batch_size < len(all_chunks):
                time.sleep(0.5)
    finally:
        await conn.close()
    
    return len(report_records), successful_chunks
