from typing import Any
import os
import re
import ssl

import certifi
from langchain_core.documents import Document
from langchain_tavily import TavilyExtract, TavilyMap

from backend.retrieval.web_sources import WEB_INGESTION_SOURCES


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


def normalize_text(value: str) -> str:
    return value.lower().replace("-", " ").replace("_", " ").strip()


def url_matches_keywords(url: str, keywords: list[str]) -> bool:
    normalized_url = normalize_text(url)

    return any(
        normalize_text(keyword) in normalized_url
        for keyword in keywords
    )


def extract_url_from_item(item: Any) -> str | None:
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        return (
            item.get("url")
            or item.get("source_url")
            or item.get("href")
            or item.get("link")
        )

    return None


def extract_urls_from_map_result(result: Any) -> list[str]:
    urls: list[str] = []

    if isinstance(result, dict):
        raw_results = (
            result.get("results")
            or result.get("urls")
            or result.get("data")
            or []
        )

        for item in raw_results:
            url = extract_url_from_item(item)

            if url:
                urls.append(url)

        return urls

    if isinstance(result, list):
        for item in result:
            url = extract_url_from_item(item)

            if url:
                urls.append(url)

        return urls

    if isinstance(result, str):
        return re.findall(r"https?://[^\s\]\)\"']+", result)

    return urls


def deduplicate_urls(urls: list[str]) -> list[str]:
    unique_urls = []
    seen_urls = set()

    for url in urls:
        if not isinstance(url, str):
            continue

        clean_url = url.strip()

        if not clean_url:
            continue

        if clean_url in seen_urls:
            continue

        unique_urls.append(clean_url)
        seen_urls.add(clean_url)

    return unique_urls


def filter_urls_by_keywords(
    urls: list[str],
    keywords: list[str],
    max_urls: int,
) -> list[str]:
    selected_urls = []
    unique_urls = deduplicate_urls(urls)

    for url in unique_urls:
        if url_matches_keywords(url, keywords):
            selected_urls.append(url)

        if len(selected_urls) >= max_urls:
            break

    return selected_urls


def fallback_select_urls(
    urls: list[str],
    max_urls: int,
) -> list[str]:
    unique_urls = deduplicate_urls(urls)
    return unique_urls[:max_urls]


def merge_urls(
    mapped_urls: list[str],
    seed_urls: list[str],
    max_urls: int,
) -> list[str]:
    merged_urls = []
    seen_urls = set()

    for url in mapped_urls + seed_urls:
        if not isinstance(url, str):
            continue

        clean_url = url.strip()

        if not clean_url:
            continue

        if clean_url in seen_urls:
            continue

        merged_urls.append(clean_url)
        seen_urls.add(clean_url)

        if len(merged_urls) >= max_urls:
            break

    return merged_urls


def map_source_urls(source: dict[str, Any]) -> list[str]:
    tavily_map = TavilyMap()

    print(f"🗺️ Mapping source: {source['source_name']}")

    try:
        result = tavily_map.invoke(
            {
                "url": source["root_url"],
                "max_depth": source.get("max_depth", 2),
            }
        )

    except Exception as error:
        print(f"❌ Mapping failed for {source['source_name']}: {error}")
        return []

    urls = extract_urls_from_map_result(result)

    if not urls:
        print(f"⚠️ No URLs returned for {source['source_name']}")
        return []

    filtered_urls = filter_urls_by_keywords(
        urls=urls,
        keywords=source.get("keywords", []),
        max_urls=source.get("max_urls", 10),
    )

    if not filtered_urls:
        print(
            f"⚠️ No URLs matched keywords for {source['source_name']}. "
            "Using fallback URL selection."
        )

        filtered_urls = fallback_select_urls(
            urls=urls,
            max_urls=source.get("max_urls", 10),
        )

    print(f"✅ Selected {len(filtered_urls)} URLs from {source['source_name']}")

    return filtered_urls


def build_document_from_extract_result(
    item: dict[str, Any],
    source: dict[str, Any],
) -> Document | None:
    raw_content = item.get("raw_content") or item.get("content")

    if not raw_content:
        return None

    url = item.get("url") or item.get("source_url") or source["root_url"]

    return Document(
        page_content=raw_content,
        metadata={
            "source": url,
            "source_url": url,
            "provider": source["provider"],
            "source_name": source["source_name"],
            "source_type": source["source_type"],
            "category": source["category"],
            "root_url": source["root_url"],
        },
    )


def extract_urls(
    urls: list[str],
    source: dict[str, Any],
) -> list[Document]:
    if not urls:
        print(f"⚠️ No URLs to extract for {source['source_name']}")
        return []

    tavily_extract = TavilyExtract()

    print(f"📥 Extracting {len(urls)} URLs from {source['source_name']}")

    try:
        result = tavily_extract.invoke(
            {
                "urls": urls,
                "extract_depth": source.get("extract_depth", "advanced"),
            }
        )

    except Exception as error:
        print(f"❌ Extraction failed for {source['source_name']}: {error}")
        return []

    if not isinstance(result, dict):
        print(
            f"⚠️ Unexpected TavilyExtract response type for "
            f"{source['source_name']}: {type(result)}"
        )
        return []

    documents = []

    for item in result.get("results", []):
        if not isinstance(item, dict):
            continue

        document = build_document_from_extract_result(
            item=item,
            source=source,
        )

        if document is not None:
            documents.append(document)

    print(f"✅ Extracted {len(documents)} documents from {source['source_name']}")

    return documents


def load_documents_from_source(source: dict[str, Any]) -> list[Document]:
    seed_urls = source.get("seed_urls", [])
    max_urls = source.get("max_urls", 10)
    min_urls = source.get("min_urls", 3)

    if source.get("use_map", True):
        mapped_urls = map_source_urls(source)
    else:
        mapped_urls = [source["root_url"]]

    if len(mapped_urls) < min_urls:
        print(
            f"⚠️ Only {len(mapped_urls)} mapped URLs found for "
            f"{source['source_name']}. Adding curated seed URLs."
        )

    urls = merge_urls(
        mapped_urls=mapped_urls,
        seed_urls=seed_urls,
        max_urls=max_urls,
    )

    print(f"🔗 Final URL count for {source['source_name']}: {len(urls)}")

    return extract_urls(
        urls=urls,
        source=source,
    )


def load_web_documents() -> list[Document]:
    all_documents = []

    for source in WEB_INGESTION_SOURCES:
        print("=" * 80)
        print(f"🌐 Processing web source: {source['source_name']}")

        try:
            documents = load_documents_from_source(source)
            all_documents.extend(documents)

        except Exception as error:
            print(f"❌ Failed processing source {source['source_name']}: {error}")
            continue

    print("=" * 80)
    print(f"📚 Total web documents loaded: {len(all_documents)}")

    return all_documents