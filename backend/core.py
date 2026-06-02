from typing import Any

from langsmith import traceable

from backend.retrieval.hybrid_retriever import retrieve_hybrid_context
from backend.generation.answer_generator import (
    generate_rag_answer,
    generate_fallback_required_answer,
)


DEFAULT_RETRIEVAL_TOP_K = 2


@traceable(
    name="peopleflow_hybrid_rag_pipeline",
    run_type="chain",
    tags=["peopleflow", "rag", "core", "hybrid"],
    metadata={
        "component": "core",
        "pipeline": "hybrid_rag",
        "strategy": "internal_first_web_fallback",
        "top_k": DEFAULT_RETRIEVAL_TOP_K,
    },
)
def run_rag_pipeline(user_question: str) -> dict[str, Any]:
    """
    Main PeopleFlow Hybrid RAG pipeline.

    Flow:
    1. Retrieve internal PeopleFlow PDF context first.
    2. If internal score is strong enough, answer using internal documentation.
    3. If internal score is weak, retrieve curated web-ingested HR SaaS context.
    4. If neither internal nor web context is useful, return a controlled fallback.
    """

    retrieval_result = retrieve_hybrid_context(
        query=user_question,
        top_k=DEFAULT_RETRIEVAL_TOP_K,
    )

    retrieval_mode = retrieval_result["retrieval_mode"]

    if retrieval_mode in ["internal_rag", "hybrid_web_fallback"]:
        return generate_rag_answer(
            user_question=user_question,
            retrieval_result=retrieval_result,
        )

    return generate_fallback_required_answer(
        user_question=user_question,
        retrieval_result=retrieval_result,
    )


def run_llm(query: str) -> dict[str, Any]:
    """
    Backward-compatible wrapper for Streamlit or old imports.
    """
    return run_rag_pipeline(query)


if __name__ == "__main__":
    test_questions = [
        "How can an employee request vacation days?",
        "What are common HR policies for vacation leave?",
        "What is the best programming language for video games?",
    ]

    for question in test_questions:
        print("=" * 100)
        print(f"QUESTION: {question}")

        result = run_rag_pipeline(question)

        print("Retrieval mode:", result.get("retrieval_mode"))
        print("Answer:", result.get("system_answer"))
        print("Chunks related:", len(result.get("chunks_related", [])))