from logger import (
    Colors,
    log_error,
    log_header,
    log_info,
    log_success,
    log_warning,
)

from backend.ingestion.text_splitter import split_documents
from backend.ingestion.web_loader import load_web_documents
from backend.ingestion.web_metadata_enricher import enrich_web_metadata
from backend.vectorestore.pinecone_store import add_documents_to_pinecone


MIN_WEB_CHUNKS = 10
WEB_PINECONE_BATCH_SIZE = 50


def validate_loaded_documents(documents: list) -> None:
    if not documents:
        raise RuntimeError(
            "No web documents were loaded. Check Tavily API key, source URLs, "
            "mapping depth, seed URLs, or keyword filters."
        )


def validate_created_chunks(chunks: list) -> None:
    if not chunks:
        raise RuntimeError("No web chunks were created from the loaded documents.")

    if len(chunks) < MIN_WEB_CHUNKS:
        log_warning(
            f"Only {len(chunks)} web chunks were created. "
            f"Expected at least {MIN_WEB_CHUNKS}. "
            "Consider increasing max_urls, max_depth, seed_urls, or adjusting keywords."
        )


def main() -> None:
    try:
        log_header("PEOPLEFLOW WEB KNOWLEDGE INGESTION PIPELINE")

        log_info("Loading curated HR SaaS web documents using Tavily Map + Extract...")
        documents = load_web_documents()

        validate_loaded_documents(documents)

        log_success(f"Loaded {len(documents)} web documents successfully.")

        log_info("Splitting web documents into chunks...")
        chunks = split_documents(
            documents,
            chunk_size=1800,
            chunk_overlap=250,
        )

        validate_created_chunks(chunks)

        log_success(f"Created {len(chunks)} web chunks successfully.")

        log_info("Enriching web chunk metadata...")
        enriched_chunks = enrich_web_metadata(chunks)

        log_success("Web metadata enrichment completed.")

        log_info("Uploading web chunks to Pinecone...")
        add_documents_to_pinecone(
            enriched_chunks,
            batch_size=WEB_PINECONE_BATCH_SIZE,
        )

        log_success("Web chunks uploaded to Pinecone successfully.")

        log_header("WEB KNOWLEDGE INGESTION COMPLETED")
        log_info(f"{Colors.CYAN}Documents loaded:{Colors.RESET} {len(documents)}")
        log_info(f"{Colors.CYAN}Chunks indexed:{Colors.RESET} {len(enriched_chunks)}")

    except Exception as error:
        log_error(f"Web ingestion pipeline failed: {error}")
        raise


if __name__ == "__main__":
    main()