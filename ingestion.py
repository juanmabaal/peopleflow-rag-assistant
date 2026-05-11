import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)


load_dotenv()

#configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where()) 
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", show_progress_bar=False, chunk_size=50, retry_max_seconds=10
)

#chorama = Chorama(persist_directory = "chorama_db", embedding_function = embeddings)
vectorStore = PineconeVectorStore(index_name="langchain-docs-2026", embedding= embeddings)
tavility_extract = TavilyExtract()
tavility_map = TavilyMap(max_depth=5, max_breadth=20, max_pages = 1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i: i + chunk_size]
        chunks.append(chunk)
    return chunks 

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch URLs."""
    try:
        log_info(
            f"🔩 TavilyExtract: Processing batch {batch_num} with {len(urls)} URLs",
            Colors.BLUE
        )
        docs = await tavility_extract.ainvoke(input={"urls": urls})
        log_success(
            f"TavilyExtract: Commpleted batch {batch_num} - extracted {len(docs.get('results', []))} documents"
        )
        return docs
    except Exception as e:
        log_error(f"TavilyExtract: Failed to extract batch {batch_num} - {e}")
        return []
    
async def async_extract(url_batches: List[List[str]]):
    log_header("DOCUMENT EXTRACTION PHASE")
    log_info(
        f"🔨 TavilyExtract: Starting Concurrent extraction of {len(url_batches)} batches",
        Colors.DARKCYAN,
    )

    task = [extract_batch(batch, i +1) for i, batch in enumerate(url_batches)]

    results = await asyncio.gather(*task, return_exceptions=True)

    # Filter out exceptions and flatten results
    all_pages = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            log_error(f"TavilyExtract: Batch failed with exception - {result}")
            failed_batches += 1
        else:
            for extracted_page in result["results"]: #type: ignore
                document = Document(
                    page_content=extracted_page["raw_content"],
                    metdata={"source": extracted_page["url"]},
                )
                all_pages.append(document)
    log_success(
        f"TavilyExtract: Extraction complete! Total pages  extracted: {len(all_pages)}"
    )
    if failed_batches > 0:
        log_warning(f"TavilyExtract: {failed_batches} batches failed during extraction")
    
    return all_pages

async def main():
    """Main async function to orchastate the entire process."""
    log_header("DOCUMENTATION INGEST PIPELINE")

    # log_info(
    #     """ TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/""",
    #         Colors.PURPLE
    # )

    log_info(
        """ TavilyCrawl: Starting to map documentation structure from https://python.langchain.com/""",
            Colors.PURPLE
    )
    site_map = tavility_map.invoke("https://python.langchain.com/")


    log_success(
        f"TavilyCrawl: Succesfully mapped {len(site_map['results'])} URLs from documentation site"
    )

    #Split URLS into batches of 20 
    urls_batches = chunk_urls(list(site_map["results"]), chunk_size=20)

    log_info(
        f"📓 URL Processing: Split {len(site_map['results'])} URLs into {len(urls_batches)} batches",
        Colors.BLUE,
    )

    all_docs = await async_extract(urls_batches)

    #crawl the documentation site

    # res = tavily_crawl.invoke({
    #     "url": "https://python.langchain.com/",
    #     "max_depth": 1,
    #     "extract_depth": "advanced",
    # })
    # docs=res['results']
    # all_docs = [Document(page_content=result['raw_content'], metadata = {"source": result['url']}) for result in res['results']]

    # log_success(
    #     f"TavilyCrawl: Succesfully crawled {len(all_docs)} URLs from documentation site"
    # )


if __name__ == "__main__":
    asyncio.run (main())
