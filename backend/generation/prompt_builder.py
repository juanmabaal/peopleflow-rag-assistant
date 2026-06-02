from typing import Any


def build_context_from_chunks(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "No relevant context was retrieved."

    context_blocks = []

    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            f"""
[Chunk {index}]
chunk_id: {chunk.get("chunk_id")}
source_type: {chunk.get("source_type")}
source_name: {chunk.get("source_name")}
provider: {chunk.get("provider")}
category: {chunk.get("category")}
source_url: {chunk.get("source_url")}
document_version: {chunk.get("document_version")}
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

Your task is to answer the user's question using ONLY the retrieved context.

Important rules:
- Do not invent PeopleFlow internal policies, procedures, or product features.
- If retrieval_mode is "internal_rag", answer as an internal PeopleFlow documentation answer.
- If retrieval_mode is "hybrid_web_fallback", clearly explain that the answer is supported by curated external HR SaaS or HR policy references, not by internal PeopleFlow documentation alone.
- If the context is not enough, say that clearly.
- Keep the answer professional, concise, and helpful.
- Return only valid JSON.
- Do not include markdown outside the JSON.
- The final answer must be useful for a customer support scenario.
- Preserve traceability by keeping chunks_related aligned with the retrieved chunks.

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
      "provider": "...",
      "category": "...",
      "source_url": "...",
      "page": 0,
      "score": 0.0,
      "content_preview": "..."
    }}
  ]
}}
""".strip()