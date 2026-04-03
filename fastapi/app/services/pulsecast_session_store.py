"""In-memory Pulsecast session (checkpointed pipeline state per OpenAI `user` / client session_id).

Single-process until a Redis-backed SessionStore replaces InMemorySessionStore.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.schemas import (
    ExecuteSqlResponse,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)

PulsecastSessionPhase = Literal[
    "idle",
    "after_planning",
    "after_sub_questions",
    "after_web",
    "after_agents",
]


class PulsecastSession(BaseModel):
    """Server-side thread state for merge + refine; JSON-serializable for future Redis."""

    model_config = {"extra": "ignore"}

    session_id: str
    raw_user_turns: list[str] = Field(default_factory=list)
    canonical_question: str | None = None
    host_plan: PlanningHostOutput | None = None
    analyst_plan: PlanningAnalystOutput | None = None
    sub_results: list[SubResult] = Field(default_factory=list)
    primary_sql: str | None = None
    primary_exe: ExecuteSqlResponse | None = None
    completed_web_results: list[dict[str, Any]] = Field(default_factory=list)
    user_declined_web_search: bool = False
    completed_phase: PulsecastSessionPhase = "idle"
    expires_at_monotonic: float = 0.0

    def touch_expiry(self, ttl_seconds: float) -> None:
        self.expires_at_monotonic = time.monotonic() + ttl_seconds


def sub_question_prefix_reuse_count(
    old_results: list[SubResult], new_sub_questions: list[str]
) -> int:
    """Largest k such that old_results[i].sub_question matches new_sub_questions[i] for all i < k."""
    k = 0
    n = min(len(old_results), len(new_sub_questions))
    for i in range(n):
        if old_results[i].sub_question.strip() != new_sub_questions[i].strip():
            break
        k = i + 1
    return k


@runtime_checkable
class SessionStore(Protocol):
    async def get(self, session_id: str) -> PulsecastSession | None: ...

    async def save(self, session: PulsecastSession) -> None: ...

    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    """TTL map session_id -> PulsecastSession (one async worker / process)."""

    def __init__(self) -> None:
        self._entries: dict[str, PulsecastSession] = {}
        self._lock = asyncio.Lock()

    def _ttl(self) -> float:
        return float(get_settings().pulsecast_session_ttl_seconds)

    def _purge_expired_unlocked(self) -> None:
        now = time.monotonic()
        dead = [sid for sid, s in self._entries.items() if s.expires_at_monotonic <= now]
        for sid in dead:
            del self._entries[sid]

    async def get(self, session_id: str) -> PulsecastSession | None:
        async with self._lock:
            self._purge_expired_unlocked()
            s = self._entries.get(session_id)
            if s is None:
                return None
            if time.monotonic() > s.expires_at_monotonic:
                del self._entries[session_id]
                return None
            return s.model_copy(deep=True)

    async def save(self, session: PulsecastSession) -> None:
        async with self._lock:
            self._purge_expired_unlocked()
            session.touch_expiry(self._ttl())
            self._entries[session.session_id] = session.model_copy(deep=True)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._entries.pop(session_id, None)


# Module singleton; replace with Redis-backed store in multi-worker deployments.
pulsecast_session_store = InMemorySessionStore()
