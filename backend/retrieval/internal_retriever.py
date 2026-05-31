from typing import Any

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.config.settings import EMBEDDING_MODEL, PINECONE_INDEX_NAME

def get_internal_vectore_store() -> PineconeVectorStore:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


def retrieve_internal_context(
        query:str,
        top_k: int=4,
) -> list[dict[str,Any]]:
    vector_store = get_internal_vectore_store()

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
                "content_preview": document.page_content[:300],
            }
        )
    
    return chunks
    