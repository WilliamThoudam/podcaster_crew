"""Pulsecast session store — checkpointed pipeline state per session_id.

Supports two backends:
- ``InMemorySessionStore`` (default, single-process)
- ``PostgresSessionStore`` (durable, multi-worker)

The module-level ``pulsecast_session_store`` is a proxy that starts with the
in-memory backend and can be swapped to Postgres at startup via
``set_session_store_impl``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings
from app.models.schemas import (
    ExecuteSqlResponse,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)

logger = logging.getLogger(__name__)

PulsecastSessionPhase = Literal[
    "idle",
    "after_planning",
    "after_sub_questions",
    "after_web",
    "after_agents",
]


class PulsecastSession(BaseModel):
    """Server-side thread state for merge + refine; JSON-serializable."""

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

    last_discussion: dict[str, Any] | None = None
    last_pipeline: list[dict[str, Any]] | None = None

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


# ---------------------------------------------------------------------------
# In-memory backend (original, single-process)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# PostgreSQL backend (durable, multi-worker)
# ---------------------------------------------------------------------------

class PostgresSessionStore:
    """Postgres-backed session store using the ``pulsecast_sessions`` table."""

    def __init__(self, pool: AsyncConnectionPool, ttl_seconds: float | None = None) -> None:
        self._pool = pool
        self._ttl = ttl_seconds or float(get_settings().pulsecast_session_ttl_seconds)

    async def get(self, session_id: str) -> PulsecastSession | None:
        async with self._pool.connection() as conn:
            cur = await conn.execute(
                "SELECT data FROM pulsecast_sessions "
                "WHERE session_id = %s AND expires_at > NOW()",
                (session_id,),
            )
            row = await cur.fetchone()
            if row is None:
                return None
            # Pool uses dict_row; column access is by name, not row[0].
            return PulsecastSession.model_validate(row["data"])

    async def save(self, session: PulsecastSession) -> None:
        data_json = json.dumps(session.model_dump(mode="json"))
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO pulsecast_sessions (session_id, data, updated_at, expires_at)
                VALUES (%s, %s::jsonb, NOW(), NOW() + (%s::double precision * interval '1 second'))
                ON CONFLICT (session_id) DO UPDATE SET
                    data = EXCLUDED.data,
                    updated_at = NOW(),
                    expires_at = EXCLUDED.expires_at
                """,
                (session.session_id, data_json, self._ttl),
            )
            await conn.commit()

    async def delete(self, session_id: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "DELETE FROM pulsecast_sessions WHERE session_id = %s",
                (session_id,),
            )
            await conn.commit()


# ---------------------------------------------------------------------------
# Proxy: swappable singleton that all other modules import
# ---------------------------------------------------------------------------

class _SessionStoreProxy:
    """Thin proxy so the module-level ``pulsecast_session_store`` reference stays
    valid even after the backend is swapped from in-memory to Postgres at startup."""

    def __init__(self) -> None:
        self._impl: SessionStore = InMemorySessionStore()

    def set_impl(self, impl: SessionStore) -> None:
        self._impl = impl
        logger.info("Session store backend switched to %s", type(impl).__name__)

    async def get(self, session_id: str) -> PulsecastSession | None:
        return await self._impl.get(session_id)

    async def save(self, session: PulsecastSession) -> None:
        return await self._impl.save(session)

    async def delete(self, session_id: str) -> None:
        return await self._impl.delete(session_id)


pulsecast_session_store = _SessionStoreProxy()


def set_session_store_impl(impl: SessionStore) -> None:
    """Called at app startup to swap the backend (e.g. to PostgresSessionStore)."""
    pulsecast_session_store.set_impl(impl)
