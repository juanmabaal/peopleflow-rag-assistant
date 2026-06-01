import json
from typing import Any

from backend.generation.llm_factory import get_chat_model
from backend.generation.prompt_builder import build_rag_prompt
from backend.schemas.rag_schema import RAGResponse

model = get_chat_model()

def build_chunks_related(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "source_type": chunk.get("source_type", "unknown"),
            "source_name": chunk.get("source_name"),
            "page": chunk.get("page"),
            "score": chunk.get("score", 0.0),
            "content_preview": chunk.get("content_preview", ""),
        }
        for chunk in chunks
    ]

def generate_internal_answer(
        user_question: str,
        retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    retrieval_mode = retrieval_result["retrieval_mode"]
    chunks = retrieval_result["chunks"]

    prompt = build_rag_prompt(
        user_question= user_question,
        chunks=chunks,
        retrieval_mode=retrieval_mode,
    )

    response = model.invoke(prompt)
    content = response.content

    try:
        parsed_response = json.loads(content)
    except json.JSONDecodeError:
        parsed_response = {
            "user_question": user_question,
            "system_answer": content,
            "retrieval_mode": retrieval_mode,
            "chunks_related": build_chunks_related(chunks),
        }
    
    parsed_response["chunks_related"] = build_chunks_related(chunks)

    validate_response = RAGResponse.model_validate(parsed_response)

    return validate_response.model_dump()

def generate_fallback_required_answer(
        user_question: str,
        retrieval_result: dict[str, Any]
) -> dict[str, Any]:
    chunks = retrieval_result["chunks"]

    response = {
        "user_question": user_question,
        "system_answer": (
            "The internal PeopleFlow documentation does not contain enough relevant "
            "information to answer this question with confidence. A web fallback search "
            "is required to enrich the response."
        ),
        "retrieval_mode": "web_fallback_required",
        "chunks_related": build_chunks_related(chunks),
    }

    validated_response = RAGResponse.model_validate(response)

    return validated_response.model_dump()
