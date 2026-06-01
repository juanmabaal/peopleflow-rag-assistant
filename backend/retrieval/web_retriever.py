from typing import Any

from langsmith import traceable

from backend.vectorstore.pinecone_store import get_pinecone_vector_store


WEB_SOURCE_TYPES = [
    "web_hr_saas_docs",
    "web_hr_policy_reference",
]

MIN_WEB_SCORE = 0.30


@traceable(
    name="retrieve_curated_web_context",
    run_type="retriever",
    tags=["peopleflow", "retrieval", "web", "pinecone"],
    metadata={
        "component": "web_retriever",
        "source": "curated_web",
        "vectorstore": "pinecone",
        "min_web_score": MIN_WEB_SCORE,
    },
)
def retrieve_web_context(
    query: str,
    top_k: int = 4,
    min_score: float = MIN_WEB_SCORE,
) -> list[dict[str, Any]]:
    vector_store = get_pinecone_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
        filter={
            "source_type": {
                "$in": WEB_SOURCE_TYPES,
            }
        },
    )

    chunks = []

    for document, score in results:
        score = float(score)

        if score < min_score:
            continue

        chunks.append(
            {
                "chunk_id": document.metadata.get("chunk_id", "unknown"),
                "source_type": document.metadata.get("source_type", "web"),
                "source_name": document.metadata.get("source_name", "Web Source"),
                "provider": document.metadata.get("provider", "Unknown Provider"),
                "category": document.metadata.get("category", "general"),
                "source_url": document.metadata.get(
                    "source_url",
                    document.metadata.get("source"),
                ),
                "page": document.metadata.get("page"),
                "score": score,
                "content": document.page_content,
                "content_preview": document.page_content[:350],
            }
        )

    return chunks