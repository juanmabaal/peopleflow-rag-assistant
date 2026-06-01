from typing import Any, Dict, List

import streamlit as st

from backend.core import run_rag_pipeline


def format_sources(chunks_related: List[Dict[str, Any]]) -> List[str]:
    sources = []

    for chunk in chunks_related or []:
        chunk_id = chunk.get("chunk_id", "unknown")
        source_name = chunk.get("source_name", "Unknown source")
        source_type = chunk.get("source_type", "unknown")
        page = chunk.get("page", "N/A")
        score = chunk.get("score", "N/A")

        sources.append(
            f"{chunk_id} | {source_name} | {source_type} | page: {page} | score: {score}"
        )

    return sources


def render_chunks(chunks_related: List[Dict[str, Any]]) -> None:
    if not chunks_related:
        st.info("No related chunks were returned.")
        return

    for chunk in chunks_related:
        chunk_id = chunk.get("chunk_id", "unknown")
        score = chunk.get("score", "N/A")

        with st.expander(f"{chunk_id} | score: {score}"):
            st.markdown(f"**Source:** {chunk.get('source_name', 'Unknown')}")
            st.markdown(f"**Source type:** {chunk.get('source_type', 'Unknown')}")
            st.markdown(f"**Page:** {chunk.get('page', 'N/A')}")
            st.markdown("**Content preview:**")
            st.write(chunk.get("content_preview", ""))


def render_assistant_response(result: Dict[str, Any]) -> None:
    answer = str(result.get("system_answer", "")).strip() or "(No answer returned.)"
    retrieval_mode = result.get("retrieval_mode", "unknown")
    chunks_related = result.get("chunks_related", [])
    sources = format_sources(chunks_related)

    st.markdown(answer)

    st.caption(f"Retrieval mode: `{retrieval_mode}`")

    if retrieval_mode == "internal_rag":
        st.success("Answer generated using internal PeopleFlow documentation.")

    elif retrieval_mode == "web_fallback_required":
        st.warning(
            "Internal documentation did not provide enough confidence. "
            "Web fallback is required."
        )

    elif retrieval_mode == "hybrid_web_fallback":
        st.info("Answer generated using internal documentation plus web fallback.")

    if sources:
        with st.expander("Sources"):
            for source in sources:
                st.markdown(f"- {source}")

    if chunks_related:
        with st.expander("Retrieved chunks"):
            render_chunks(chunks_related)


st.set_page_config(
    page_title="PeopleFlow RAG Assistant",
    layout="centered",
)

st.title("PeopleFlow RAG Assistant")
st.caption(
    "Hybrid RAG assistant for HR SaaS support using internal PeopleFlow documentation "
    "and web fallback logic."
)

with st.sidebar:
    st.subheader("Session")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None)
        st.rerun()

    st.markdown("---")
    st.markdown("### Retrieval Logic")
    st.markdown(
        """
        1. Search internal PeopleFlow documentation  
        2. Evaluate retrieval confidence  
        3. Use internal RAG if confidence is enough  
        4. Require web fallback if confidence is low  
        """
    )

    st.markdown("---")
    st.markdown("### Example questions")
    st.markdown(
        """
        - How can an employee request vacation days?
        - What happens during employee offboarding?
        - How can a manager approve leave requests?
        - What should I do if I forgot my password?
        """
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I am the PeopleFlow RAG Assistant. "
                "Ask me questions about HR policies, payroll, onboarding, offboarding, "
                "time off, benefits, security, or PeopleFlow procedures."
            ),
            "sources": [],
            "retrieval_mode": None,
            "chunks_related": [],
        }
    ]


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("retrieval_mode"):
            st.caption(f"Retrieval mode: `{msg['retrieval_mode']}`")

        if msg.get("sources"):
            with st.expander("Sources"):
                for source in msg["sources"]:
                    st.markdown(f"- {source}")

        if msg.get("chunks_related"):
            with st.expander("Retrieved chunks"):
                render_chunks(msg["chunks_related"])


prompt = st.chat_input("Ask a question about PeopleFlow HR...")

if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
            "sources": [],
            "retrieval_mode": None,
            "chunks_related": [],
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving PeopleFlow context and generating answer..."):
                result: Dict[str, Any] = run_rag_pipeline(prompt)

                answer = (
                    str(result.get("system_answer", "")).strip()
                    or "(No answer returned.)"
                )
                retrieval_mode = result.get("retrieval_mode", "unknown")
                chunks_related = result.get("chunks_related", [])
                sources = format_sources(chunks_related)

            render_assistant_response(result)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "retrieval_mode": retrieval_mode,
                    "chunks_related": chunks_related,
                }
            )

        except Exception as error:
            st.error("Failed to generate a response.")
            st.exception(error)