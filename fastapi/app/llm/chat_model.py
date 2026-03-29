from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """
    OpenAI-compatible chat model (OpenAI, OpenRouter, LiteLLM gateway, vLLM, etc.).
    Settings.openai_base_url should be the API root, e.g. https://api.openai.com/v1 or http://host:8000/v1.
    """
    base = (settings.openai_base_url or "").strip().rstrip("/")
    if not base:
        raise ValueError("openai_base_url is required")
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=base,
        temperature=settings.openai_temperature,
        timeout=settings.openai_timeout_seconds,
    )
