from typing import Sequence

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from backend.config.settings import EMBEDDING_MODEL, PINECONE_INDEX_NAME


DEFAULT_BATCH_SIZE = 50


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def get_pinecone_vector_store() -> PineconeVectorStore:
    embeddings = get_embeddings()

    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


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
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    if not documents:
        raise ValueError("No documents provided to upload to Pinecone.")

    vector_store = get_pinecone_vector_store()
    document_batches = batch_documents(
        documents=documents,
        batch_size=batch_size,
    )

    print(f"📦 Uploading {len(documents)} documents to Pinecone...")
    print(f"📚 Total batches: {len(document_batches)}")
    print(f"📏 Batch size: {batch_size}")

    for batch_index, document_batch in enumerate(document_batches, start=1):
        print(
            f"⬆️ Uploading batch {batch_index}/{len(document_batches)} "
            f"with {len(document_batch)} documents..."
        )

        vector_store.add_documents(list(document_batch))

    print("✅ All document batches uploaded to Pinecone successfully.")