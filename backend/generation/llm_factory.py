from langchain_openai import ChatOpenAI

from backend.config.settings import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    XAI_API_KEY,
    XAI_MODEL,
    XAI_BASE_URL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    DEEPSEEK_BASE_URL,
)


def get_chat_model():
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=None,
        )

    if LLM_PROVIDER == "xai":
        if not XAI_API_KEY:
            raise RuntimeError("XAI_API_KEY is not configured.")

        return ChatOpenAI(
            model=XAI_MODEL,
            api_key=XAI_API_KEY,
            base_url=XAI_BASE_URL,
            temperature=0.2,
        )

    if LLM_PROVIDER == "deepseek":
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")

        return ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.2,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}. "
        "Use one of: openai, xai, deepseek."
    )