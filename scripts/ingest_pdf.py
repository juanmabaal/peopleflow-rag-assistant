import os
from pathlib import Path

from backend.config.settings import PDF_PATH
from backend.ingestion.pdf_loader import load_pdf_document
from backend.ingestion.text_splitter import split_documents
from backend.ingestion.metadata_enricher import enrich_metadata
from backend.vectorestore.pinecone_store import add_documents_to_pinecone
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)


def main():
    if not PDF_PATH.exists():
        log_error(f"Error: The PDF file was not found in {PDF_PATH}")
        print('the path doesn´t exist')
        exit(1)

    log_info("📖 Loading PeopleFlow HR Pdf ...")
    documents =load_pdf_document(PDF_PATH)

    log_success(f"✅ Loaded {len(documents)} pages")
    
    log_info("✂️ Splitting documents into chunks...")
    chunks = split_documents(documents)

    log_success(f"✅ Created {len(chunks)} chunks")

    if len(chunks) < 20:
        raise ValueError("The document must generate al least 20 chunks.")
    
    log_info("🏷️ Enriching metadata...")
    enriched_chunks  = enrich_metadata(chunks)

    log_info("📦 Uploading chunks to Pinecone...")
    add_documents_to_pinecone(enriched_chunks)

    log_success("🎉 PeopleFlow HR ingestion completed successfully.")

if __name__ == "__main__":
    main()
