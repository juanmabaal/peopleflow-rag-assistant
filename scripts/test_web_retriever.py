from backend.retrieval.web_retriever import retrieve_web_context


TEST_QUERIES = [
    "What are common HR policies for vacation leave?",
    "How should payroll admins manage time off policies?",
    "What are common onboarding steps for new employees?",
    "What is the best programming language for video games?",
]


def main():
    for query in TEST_QUERIES:
        print("\n" + "=" * 100)
        print("QUERY:", query)

        chunks = retrieve_web_context(query=query, top_k=4)

        print("Chunks retrieved:", len(chunks))

        if not chunks:
            continue

        scores = [chunk["score"] for chunk in chunks]

        print("Best score:", max(scores))
        print("Scores:", scores)

        top_chunk = chunks[0]

        print("Top chunk:", top_chunk["chunk_id"])
        print("Provider:", top_chunk.get("provider"))
        print("Source:", top_chunk.get("source_name"))
        print("URL:", top_chunk.get("source_url"))
        print("Preview:", top_chunk["content_preview"])


if __name__ == "__main__":
    main()