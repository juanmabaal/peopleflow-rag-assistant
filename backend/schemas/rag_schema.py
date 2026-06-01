from typing import Literal
from pydantic import BaseModel, Field

class ChunkRelated(BaseModel):
    chunk_id: str
    source_type: str
    source_name: str | None = None
    page: float | int | None = None
    score: float
    content_preview: str

class RAGResponse(BaseModel):
    user_question: str
    system_answer: str
    retrieval_mode: Literal[
        "internal_rag",
        "web_fallback_required",
        "hybrid_web_fallback",
    ]
    chunks_related: list[ChunkRelated] = Field(default_factory=list)

