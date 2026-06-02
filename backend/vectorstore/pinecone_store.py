from typing import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.config.settings import (
    EMBEDDING_MODEL,
    PINECONE_INDEX_NAME_INTERNAL,
    PINECONE_INDEX_NAME_WEB,
)


DEFAULT_BATCH_SIZE = 50


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_pinecone_vector_store(index_name: str) -> PineconeVectorStore:
    """
    Create a Pinecone vector store for the provided index name.

    This function is intentionally generic so it can be reused by:
    - internal PDF retriever
    - web retriever
    - PDF ingestion
    - web ingestion
    """
    embeddings = get_embeddings()

    return PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )

def get_internal_vector_store() -> PineconeVectorStore:
    """
    Vector store for the internal PeopleFlow PDF knowledge base.
    """
    return get_pinecone_vector_store(PINECONE_INDEX_NAME_INTERNAL)


def get_web_vector_store() -> PineconeVectorStore:
    """
    Vector store for curated web-ingested HR SaaS references.
    """
    return get_pinecone_vector_store(PINECONE_INDEX_NAME_WEB)



def batch_documents(
    documents: Sequence[Document],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[Sequence[Document]]:
    return [
        documents[index : index + batch_size]
        for index in range(0, len(documents), batch_size)
    ]


def add_documents_to_pinecone(
    documents: Sequence[Document],
    index_name: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """
    Upload documents to a specific Pinecone index in batches.

    The index_name must be explicitly provided to avoid accidentally
    mixing internal PDF knowledge and curated web knowledge.
    """
    if not documents:
        raise ValueError("No documents provided to upload to Pinecone.")

    vector_store = get_pinecone_vector_store(index_name=index_name)

    document_batches = batch_documents(
        documents=documents,
        batch_size=batch_size,
    )

    print(f"📦 Uploading {len(documents)} documents to Pinecone index: {index_name}")
    print(f"📚 Total batches: {len(document_batches)}")
    print(f"📏 Batch size: {batch_size}")

    for batch_index, document_batch in enumerate(document_batches, start=1):
        print(
            f"⬆️ Uploading batch {batch_index}/{len(document_batches)} "
            f"with {len(document_batch)} documents..."
        )

        vector_store.add_documents(list(document_batch))

    print("✅ All document batches uploaded to Pinecone successfully.")