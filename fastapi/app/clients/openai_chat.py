from __future__ import annotations

from typing import Any

import httpx


def _chat_completions_url(base_url: str) -> str:
    """
    Accept either:
    - https://host/v1
    - https://host
    and normalize to the Chat Completions endpoint.
    """
    b = (base_url or "").strip().rstrip("/")
    if not b:
        raise ValueError("OPENAI_BASE_URL is required")
    if b.endswith("/v1"):
        return b + "/chat/completions"
    return b + "/v1/chat/completions"


async def chat_complete_json(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = _chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()


def extract_assistant_text(body: dict[str, Any]) -> str:
    try:
        choices = body.get("choices") or []
        if not choices:
            raise KeyError("choices")
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise TypeError("content")
        return content
    except Exception:
        # Keep response short; callers may include details in logs / HTTP errors
        raise ValueError("OpenAI-compatible response missing choices[0].message.content")

