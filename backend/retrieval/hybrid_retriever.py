from typing import Any

from backend.retrieval.internal_retriever import retrieve_internal_context


MIN_INTERNAL_SCORE = 0.3

def get_best_score (chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    
    return max(chunk["score"] for chunk in chunks)

def is_internal_context_sufficient(
        chunks: list[dict[str, Any]],
        min_score: float = MIN_INTERNAL_SCORE,
) -> bool:
    
    if not chunks:
        return False
    
    best_score = get_best_score(chunks)

    return best_score >= min_score

def retrieve_hybrid_context(
        query:str,
        top_k: int = 4,
) -> dict[str, Any]:
    
    internal_chunks = retrieve_internal_context(
        query=query,
        top_k=top_k,
    )

    best_score = get_best_score(internal_chunks)

    if is_internal_context_sufficient(internal_chunks):
        return {
            "retrieval_mode": "internal_rag",
            "web_used": False,
            "best_score": best_score,
            "threshold": MIN_INTERNAL_SCORE,
            "chunks": internal_chunks,
        }
    return {
        "retrieval_mode": "web_fallback_required",
        "web_used": True,
        "best_score": best_score,
        "threshold": MIN_INTERNAL_SCORE,
        "chunks": internal_chunks,
    }