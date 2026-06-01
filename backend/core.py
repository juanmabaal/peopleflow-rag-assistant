from typing import Any

from backend.retrieval.hybrid_retriever import retrieve_hybrid_context
from backend.generation.answer_generator import (
    generate_internal_answer,
    generate_fallback_required_answer,
)


def run_rag_pipeline(user_question: str) -> dict[str, Any]:
    retrieval_result = retrieve_hybrid_context(
        query=user_question,
        top_k=4,
    )

    retrieval_mode = retrieval_result["retrieval_mode"]

    if retrieval_mode == "internal_rag":
        return generate_internal_answer(
            user_question=user_question,
            retrieval_result=retrieval_result,
        )

    return generate_fallback_required_answer(
        user_question=user_question,
        retrieval_result=retrieval_result,
    )


def run_llm(query: str) -> dict[str, Any]:
    """
    Backward-compatible wrapper for the Streamlit UI or old imports.
    Internally, this now runs the PeopleFlow RAG pipeline.
    """
    return run_rag_pipeline(query)


if __name__ == "__main__":
    result = run_rag_pipeline(
        "How can an employee request vacation days?"
    )

    print(result)