from backend.retrieval.internal_retriever import retrieve_internal_context
from backend.retrieval.hybrid_retriever import retrieve_hybrid_context

TEST_QUERIES = [
    "How can an employee request vacation days?",
    "How do I submit a time off request?",
    "What should I do if I forgot my password?",
    "How can a manager approve leave requests?",
    "What happens during employee offboarding?",
    "What is the best programming language for video games?",
    "How do I cook pasta?",
]

def main():
    # query = "quiero pedir vacaciones, como lo hago"

    # chunks = retrieve_internal_context(query, top_k=4)

    # for chunk in chunks:
    #     print("=" * 80)
    #     print("Chunk ID:", chunk["chunk_id"])
    #     print("Score:", chunk["score"])
    #     print("Source:", chunk["source_name"])
    #     print("Page:", chunk["page"])
    #     print("Preview:", chunk["content_preview"])

    # print("\n" + "=" * 80)

    # if not chunks:
    #     print("No chunks retrieved.")
    #     return

    # best_score = max(chunk["score"] for chunk in chunks)

    # threshold = 0.70
    # is_sufficient = best_score >= threshold

    # print("Best score:", best_score)
    # print("Threshold:", threshold)
    # print("Internal context sufficient:", is_sufficient)

    # if is_sufficient:
    #     print("Decision: Use internal PeopleFlow RAG")
    # else:
    #     print("Decision: Use web fallback")

    for query in TEST_QUERIES:
        result = retrieve_hybrid_context(query=query, top_k=4)

        print("\n" + "=" * 100)
        print("QUERY:", query)
        print("Retrieval mode:", result["retrieval_mode"])
        print("Web used:", result["web_used"])
        print("Best score:", result["best_score"])
        print("Threshold:", result["threshold"])
        print("Chunks retrieved:", len(result["chunks"]))

        if result["chunks"]:
            top_chunk = result["chunks"][0]
            print("Top chunk:", top_chunk["chunk_id"])
            print("Top preview:", top_chunk["content_preview"])
        
        



if __name__ == "__main__":
    main()