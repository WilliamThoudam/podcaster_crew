import os
from functools import lru_cache

from langchain_openai import ChatOpenAI


@lru_cache(maxsize=1)
def chat_model() -> ChatOpenAI:
    """OpenAI-compatible chat model (same env contract as crew OpenAI LLM)."""
    model = os.getenv("OPENAI_MODEL") or os.getenv("MODEL") or "gpt-4.1-mini-2025-04-14"
    base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip() or "dummy-key-for-local"
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0.7,
    )
