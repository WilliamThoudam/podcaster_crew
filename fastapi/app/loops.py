"""Custom asyncio loop factory for Uvicorn on Windows.

Uvicorn's built-in ``asyncio`` loop uses ``ProactorEventLoop`` on Windows
(see ``uvicorn.loops.asyncio``). Psycopg async requires ``SelectorEventLoop``.

Usage::

    uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop app.loops:selector_loop_factory

Or from the ``fastapi`` directory::

    python -m app
"""

from __future__ import annotations

import asyncio
import selectors


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """Return a new selector event loop (compatible with psycopg async on Windows)."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())
