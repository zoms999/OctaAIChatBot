from pydantic import BaseModel, Field
from typing import Literal, Optional


# --- Indexing API Models ---
class IndexRequest(BaseModel):
    anp_seq: int
    language_code: str = 'ko-KR'


class IndexResponse(BaseModel):
    success: bool
    message: str
    anp_seq: int
    total_chunks_created: int


# --- Chat API Models ---
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="각 대화방을 구분하는 고유 ID")
    anp_seq: int
    question: str
    language_code: str = 'ko-KR'
    provider: Optional[Literal['openai', 'google']] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
