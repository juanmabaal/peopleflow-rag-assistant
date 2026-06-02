from typing import Any

from langsmith import traceable

from backend.retrieval.internal_retriever import retrieve_internal_context
from backend.retrieval.web_retriever import retrieve_web_context


MIN_INTERNAL_SCORE = 0.50


def get_best_score(chunks: list[dict[str, Any]]) -> float:
    """
    Return the highest similarity score from a list of retrieved chunks.
    If there are no chunks, return 0.0.
    """
    if not chunks:
        return 0.0

    return max(chunk.get("score", 0.0) for chunk in chunks)


def is_internal_context_sufficient(
    chunks: list[dict[str, Any]],
    min_score: float = MIN_INTERNAL_SCORE,
) -> bool:
    """
    Decide if the internal PeopleFlow documentation is good enough
    to answer the user question without using web fallback.
    """
    if not chunks:
        return False

    best_score = get_best_score(chunks)

    return best_score >= min_score


@traceable(
    name="hybrid_retrieval_decision",
    run_type="chain",
    tags=["peopleflow", "retrieval", "hybrid_decision"],
    metadata={
        "component": "hybrid_retriever",
        "strategy": "internal_first_then_web_fallback",
        "internal_threshold": MIN_INTERNAL_SCORE,
    },
)
def retrieve_hybrid_context(
    query: str,
    top_k: int = 2,
) -> dict[str, Any]:
    """
    Hybrid retrieval strategy:

    1. Search internal PeopleFlow documentation first.
    2. If internal confidence is high enough, use internal RAG.
    3. If internal confidence is low, retrieve curated web-ingested chunks.
    4. If web retrieval also returns no useful chunks, keep fallback as required.
    """

    internal_chunks = retrieve_internal_context(
        query=query,
        top_k=top_k,
    )

    internal_best_score = get_best_score(internal_chunks)

    if is_internal_context_sufficient(internal_chunks):
        return {
            "retrieval_mode": "internal_rag",
            "web_used": False,
            "best_score": internal_best_score,
            "internal_best_score": internal_best_score,
            "web_best_score": None,
            "threshold": MIN_INTERNAL_SCORE,
            "chunks": internal_chunks,
            "internal_chunks": internal_chunks,
            "web_chunks": [],
        }

    web_chunks = retrieve_web_context(
        query=query,
        top_k=top_k,
    )

    web_best_score = get_best_score(web_chunks)

    if not web_chunks:
        return {
            "retrieval_mode": "web_fallback_required",
            "web_used": True,
            "best_score": internal_best_score,
            "internal_best_score": internal_best_score,
            "web_best_score": 0.0,
            "threshold": MIN_INTERNAL_SCORE,
            "chunks": internal_chunks,
            "internal_chunks": internal_chunks,
            "web_chunks": [],
        }

    combined_chunks = web_chunks + internal_chunks

    return {
        "retrieval_mode": "hybrid_web_fallback",
        "web_used": True,
        "best_score": web_best_score,
        "internal_best_score": internal_best_score,
        "web_best_score": web_best_score,
        "threshold": MIN_INTERNAL_SCORE,
        "chunks": combined_chunks,
        "internal_chunks": internal_chunks,
        "web_chunks": web_chunks,
    }