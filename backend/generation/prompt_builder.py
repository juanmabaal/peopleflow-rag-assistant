from typing import Any

def build_context_from_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant internal context was retrieved."
    
    context_blocks = []

    for index, chunk in enumerate(chunks, start = 1):
        context_blocks.append(
            f"""
            [Chunk {index}]
            chunk_id: {chunk.get("chunk_id")}
            source_type: {chunk.get("source_type")}
            source_name: {chunk.get("source_name")}
            page: {chunk.get("page")}
            score: {chunk.get("score")}

            content:
            {chunk.get("content")}
            """.strip()
        )
    
    return "\n\n".join(context_blocks)

def build_rag_prompt(
        user_question: str,
        chunks: list[dict[str, Any]],
        retrieval_mode: str,
) -> str:
    context = build_context_from_chunks(chunks)

    return f"""
    You are PeopleFlow AI Support Assistant, an HR SaaS support chatbot.

    Your task is to answer the user's question using ONLY the provided retrieved context.

    Rules:
    - Do not invent policies, procedures, or product features.
    - If the internal documentation does not contain enough information, say that clearly.
    - Keep the answer professional, concise, and helpful.
    - Return only valid JSON.
    - The JSON must include:
    - user_question
    - system_answer
    - retrieval_mode
    - chunks_related

    User question:
    {user_question}

    Retrieval mode:
    {retrieval_mode}

    Retrieved context:
    {context}

    Return the response using this exact JSON structure:

    {{
    "user_question": "{user_question}",
    "system_answer": "...",
    "retrieval_mode": "{retrieval_mode}",
    "chunks_related": [
        {{
        "chunk_id": "...",
        "source_type": "...",
        "source_name": "...",
        "page": 0,
        "score": 0.0,
        "content_preview": "..."
        }}
    ]
    }}
    """.strip()