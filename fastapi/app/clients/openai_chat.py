from __future__ import annotations

import json
from typing import Any, AsyncIterator

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
        if resp.status_code >= 400:
            body_snippet = resp.text[:1000].replace("\n", " ")
            raise httpx.HTTPStatusError(
                f"OpenAI-compatible upstream HTTP {resp.status_code} at {url}. Body: {body_snippet}",
                request=resp.request,
                response=resp,
            )
        return resp.json()


async def chat_complete_stream_text(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout_seconds: float,
) -> AsyncIterator[str]:
    """
    True upstream token streaming from an OpenAI-compatible endpoint.
    Yields assistant text deltas only.
    """
    url = _chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", errors="replace")
                body_snippet = text[:1000].replace("\n", " ")
                raise httpx.HTTPStatusError(
                    f"OpenAI-compatible upstream HTTP {resp.status_code} at {url}. Body: {body_snippet}",
                    request=resp.request,
                    response=resp,
                )

            async for raw in resp.aiter_lines():
                line = (raw or "").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = (((obj.get("choices") or [{}])[0].get("delta") or {}).get("content"))
                    if isinstance(delta, str) and delta:
                        yield delta
                except json.JSONDecodeError:
                    continue


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

