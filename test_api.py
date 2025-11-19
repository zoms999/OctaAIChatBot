"""
API 테스트 스크립트
FastAPI 서버가 실행 중일 때 이 스크립트로 API를 테스트할 수 있습니다.
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """헬스 체크 테스트"""
    print("\n=== Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


def test_index_report(anp_seq: int):
    """보고서 인덱싱 테스트"""
    print(f"\n=== Indexing Report for anp_seq={anp_seq} ===")
    response = requests.post(
        f"{BASE_URL}/index-report",
        json={"anp_seq": anp_seq}
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_chat(session_id: str, anp_seq: int, question: str, provider: str = "google"):
    """챗봇 대화 테스트"""
    print(f"\n=== Chat Test ===")
    print(f"Question: {question}")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={
            "session_id": session_id,
            "anp_seq": anp_seq,
            "question": question,
            "language_code": "ko-KR",
            "provider": provider
        }
    )
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Answer: {response.json()['answer']}")
    else:
        print(f"Error: {response.json()}")


if __name__ == "__main__":
    # 테스트할 anp_seq (실제 데이터가 있는 값으로 변경)
    TEST_ANP_SEQ = 12345
    TEST_SESSION_ID = "test-session-001"
    
    # 1. 헬스 체크
    test_health()
    
    # 2. 보고서 인덱싱
    # test_index_report(TEST_ANP_SEQ)
    
    # 3. 챗봇 대화 테스트
    # test_chat(TEST_SESSION_ID, TEST_ANP_SEQ, "저의 가장 중요한 성향은 무엇인가요?")
    # test_chat(TEST_SESSION_ID, TEST_ANP_SEQ, "이 성향을 바탕으로 추천할 만한 직업을 알려주세요.")
    # test_chat(TEST_SESSION_ID, TEST_ANP_SEQ, "제 역량 중 가장 높은 것은 무엇인가요?")
    
    print("\n=== Test Complete ===")
    print("주석을 해제하여 다른 테스트를 실행하세요.")
