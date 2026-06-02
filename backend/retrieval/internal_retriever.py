from typing import Any

from langsmith import traceable

from backend.vectorstore.pinecone_store import get_internal_vector_store

@traceable(
    name="retrieve_internal_peopleflow_context",
    run_type="retriever",
    tags=["peopleflow", "retrieval", "internal", "pinecone"],
    metadata={
        "component": "internal_retriever",
        "source": "peopleflow_pdf",
        "vectorstore": "pinecone",
        "index_type": "internal_pdf",
    },
)
def retrieve_internal_context(
        query:str,
        top_k: int=4,
) -> list[dict[str,Any]]:
    vector_store = get_internal_vector_store()

    results = vector_store.similarity_search_with_score(
    query=query,
    k=top_k,
    )

    chunks = []


    for document, score in results:
        chunks.append(
            {
                "chunk_id": document.metadata.get("chunk_id", "unknown"),
                "source_type": document.metadata.get("source_type", "internal_pdf"),
                "source_name": document.metadata.get(
                    "source_name",
                    "PeopleFlow HR Knowledge Base",
                ),
                "document_version": document.metadata.get("document_version"),
                "page": document.metadata.get("page"),
                "score": float(score),
                "content": document.page_content,
                "content_preview": document.page_content[:600],
            }
        )
    
    return chunks
    