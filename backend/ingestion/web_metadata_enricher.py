from langchain_core.documents import Document


def enrich_web_metadata(chunks: list[Document]) -> list[Document]:
    enriched_chunks = []

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"web_chunk_{index + 1:04d}"
        chunk.metadata["source_type"] = chunk.metadata.get("source_type", "web")
        chunk.metadata["source_name"] = chunk.metadata.get("source_name", "Web Source")
        chunk.metadata["source_url"] = chunk.metadata.get(
            "source_url",
            chunk.metadata.get("source"),
        )
        chunk.metadata["provider"] = chunk.metadata.get("provider", "Unknown Provider")
        chunk.metadata["category"] = chunk.metadata.get("category", "general")
        chunk.metadata["document_version"] = "web_v1.0"

        enriched_chunks.append(chunk)

    return enriched_chunks