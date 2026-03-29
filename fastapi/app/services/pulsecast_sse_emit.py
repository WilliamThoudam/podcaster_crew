"""Shared SSE progress emission for Pulsecast (avoids circular imports with pulsecast_llm_agents)."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Union

ProgressCallback = Callable[[dict[str, Any]], Union[Awaitable[None], None]]

STREAM_CHUNK_SIZE = 10
STREAM_DELAY_S = 0.001


async def emit_progress(cb: ProgressCallback | None, event: dict[str, Any]) -> None:
    if cb is None:
        return
    maybe = cb(event)
    if asyncio.iscoroutine(maybe):
        await maybe


async def emit_text_chunks(
    *,
    on_progress: ProgressCallback | None,
    base_event: dict[str, Any],
    text: str,
    chunk_size: int = STREAM_CHUNK_SIZE,
    event_type: str = "tts_sql_chunk",
    delay_s: float = STREAM_DELAY_S,
) -> None:
    s = text or ""
    for i in range(0, len(s), chunk_size):
        await emit_progress(
            on_progress,
            {**base_event, "type": event_type, "chunk": s[i : i + chunk_size]},
        )
        await asyncio.sleep(delay_s)
