"""Shared PostgreSQL connection pool and LangGraph checkpointer lifecycle.

Call ``init_postgres`` during app startup and ``close_postgres`` during shutdown.
All other modules obtain the pool / checkpointer via the module-level getters.
When DATABASE_URL is not configured the getters return ``None`` and callers
fall back to in-memory stores.
"""

from __future__ import annotations

import logging
from typing import Any

import psycopg
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)


async def _pool_check_connection(conn: AsyncConnection) -> None:
    """Reject dead/stale connections before LangGraph or app code uses them.

    Without this, the pool can hand out TCP connections the server already closed
    (common with managed Postgres + idle timeouts), which surfaces as
    ``OperationalError: server closed the connection unexpectedly`` inside
    ``AsyncPostgresSaver.aget_tuple``.
    """
    await conn.execute("SELECT 1")


_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None


def _normalize_dsn(url: str) -> str:
    """psycopg v3 requires the ``postgresql://`` scheme."""
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://"):]
    return url


async def init_postgres(database_url: str) -> None:
    """Open shared connection pool, create LangGraph checkpoint tables, and
    set up the application-level ``pulsecast_sessions`` / ``pulsecast_resume_tokens``
    tables if they don't already exist."""
    global _pool, _checkpointer

    dsn = _normalize_dsn(database_url)
    _pool = AsyncConnectionPool(
        conninfo=dsn,
        open=False,
        min_size=2,
        max_size=16,
        kwargs={
            # Match LangGraph's ``from_conn_string`` defaults where it matters.
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "connect_timeout": 60,
            # Keep TCP sessions alive through NAT / cloud load balancers.
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        },
        check=_pool_check_connection,
        # Recycle before typical managed-Postgres idle cuts (e.g. Neon).
        max_idle=120.0,
        max_lifetime=900.0,
        timeout=60.0,
        reconnect_timeout=120.0,
    )
    await _pool.open()
    logger.info("Postgres connection pool opened (%s)", dsn.split("@")[-1])

    # AsyncPostgresSaver.setup() runs CREATE INDEX CONCURRENTLY which
    # cannot execute inside a transaction.  Use a dedicated autocommit
    # connection for the migration DDL.
    async with await psycopg.AsyncConnection.connect(
        dsn,
        autocommit=True,
        prepare_threshold=0,
        row_factory=dict_row,
        connect_timeout=60,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    ) as setup_conn:
        _checkpointer = AsyncPostgresSaver(setup_conn)
        await _checkpointer.setup()

    # Now swap to the pool for all runtime operations.
    _checkpointer = AsyncPostgresSaver(_pool)
    logger.info("LangGraph checkpoint tables ready")

    async with _pool.connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pulsecast_sessions (
                session_id  TEXT PRIMARY KEY,
                data        JSONB NOT NULL DEFAULT '{}',
                updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at  TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 minutes')
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pulsecast_sessions_expires
            ON pulsecast_sessions (expires_at)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS pulsecast_resume_tokens (
                token       TEXT PRIMARY KEY,
                pause_kind  TEXT NOT NULL DEFAULT '',
                snapshot    JSONB NOT NULL DEFAULT '{}',
                created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                expires_at  TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '1 hour')
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_pulsecast_resume_tokens_expires
            ON pulsecast_resume_tokens (expires_at)
        """)
        await conn.commit()
    logger.info("Application tables ready (pulsecast_sessions, pulsecast_resume_tokens)")


async def close_postgres() -> None:
    global _pool, _checkpointer
    if _pool is not None:
        await _pool.close()
        logger.info("Postgres connection pool closed")
    _pool = None
    _checkpointer = None


def get_pool() -> AsyncConnectionPool | None:
    return _pool


def get_checkpointer() -> Any | None:
    """Return the ``AsyncPostgresSaver`` (or ``None`` when Postgres is not configured)."""
    return _checkpointer
