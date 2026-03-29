"""Serper.dev Google Search API — used after user approves Web Crawler HITL."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import Settings

SERPER_SEARCH_URL = "https://google.serper.dev/search"


async def serper_google_search(
    *,
    settings: Settings,
    query: str,
    num: int = 8,
) -> list[dict[str, Any]]:
    """Return organic results as {title, link, snippet} dicts."""
    key = (settings.serper_api_key or "").strip()
    if not key:
        raise ValueError("serper_api_key is not configured")

    q = (query or "").strip()
    if not q:
        raise ValueError("search query is empty")

    n = max(1, min(10, num))
    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        resp = await client.post(
            SERPER_SEARCH_URL,
            json={"q": q, "num": n},
            headers={
                "X-API-KEY": key,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    organic = data.get("organic") if isinstance(data, dict) else None
    if not isinstance(organic, list):
        return []

    out: list[dict[str, Any]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        link = item.get("link")
        snippet = item.get("snippet")
        out.append(
            {
                "title": str(title) if title is not None else "",
                "link": str(link) if link is not None else "",
                "snippet": str(snippet) if snippet is not None else "",
            }
        )
    return out
