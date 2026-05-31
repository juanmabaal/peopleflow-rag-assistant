def enrich_metadata(chunks):
    enriched_chunks = []

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"peopleflow_chunk_{index + 1:04d}"
        chunk.metadata["source_type"] = "internal_pdf"
        chunk.metadata["source_name"] = "PeopleFlow HR Knowledge Base"
        chunk.metadata["document_version"] = "v1.0"

        enriched_chunks.append(chunk)

    return enriched_chunks