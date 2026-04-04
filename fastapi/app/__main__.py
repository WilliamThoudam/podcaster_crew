"""Run the API with a Windows-safe event loop when using async Postgres (psycopg)."""

from __future__ import annotations

import sys

import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            loop="app.loops:selector_loop_factory",
        )
    else:
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
