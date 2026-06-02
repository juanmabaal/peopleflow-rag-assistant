import json
from typing import Any

from langsmith import traceable

from backend.generation.llm_factory import get_chat_model
from backend.generation.prompt_builder import build_rag_prompt
from backend.schemas.rag_schema import RAGResponse


def build_chunks_related(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.get("chunk_id", "unknown"),
            "source_type": chunk.get("source_type", "unknown"),
            "source_name": chunk.get("source_name"),
            "provider": chunk.get("provider"),
            "category": chunk.get("category"),
            "source_url": chunk.get("source_url"),
            "page": chunk.get("page"),
            "score": chunk.get("score", 0.0),
            "content_preview": chunk.get("content_preview", ""),
        }
        for chunk in chunks
    ]


@traceable(
    name="generate_structured_rag_answer",
    run_type="llm",
    tags=["peopleflow", "generation", "structured_json"],
    metadata={
        "component": "answer_generator",
        "output_format": "rag_json",
    },
)
def generate_rag_answer(
    user_question: str,
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    retrieval_mode = retrieval_result["retrieval_mode"]
    chunks = retrieval_result["chunks"]

    prompt = build_rag_prompt(
        user_question=user_question,
        chunks=chunks,
        retrieval_mode=retrieval_mode,
    )

    model = get_chat_model()
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

    validated_response = RAGResponse.model_validate(parsed_response)

    return validated_response.model_dump()


@traceable(
    name="generate_controlled_fallback_answer",
    run_type="chain",
    tags=["peopleflow", "generation", "fallback"],
    metadata={
        "component": "answer_generator",
        "reason": "insufficient_internal_and_web_context",
    },
)
def generate_fallback_required_answer(
    user_question: str,
    retrieval_result: dict[str, Any],
) -> dict[str, Any]:
    chunks = retrieval_result["chunks"]

    response = {
        "user_question": user_question,
        "system_answer": (
            "The available PeopleFlow internal documentation and curated external HR SaaS "
            "references do not contain enough relevant context to answer this question with "
            "confidence. Please contact a human support specialist or provide more details."
        ),
        "retrieval_mode": "web_fallback_required",
        "chunks_related": build_chunks_related(chunks),
    }

    validated_response = RAGResponse.model_validate(response)

    return validated_response.model_dump()


# Backward-compatible alias, in case another file still imports the old name.
generate_internal_answer = generate_rag_answer