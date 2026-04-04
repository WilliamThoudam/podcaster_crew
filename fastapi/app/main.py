from __future__ import annotations

import asyncio
import logging
import sys

# Psycopg async needs SelectorEventLoop on Windows. Uvicorn still forces
# ProactorEventLoop via its own loop factory, so run with:
#   --loop app.loops:selector_loop_factory
# or: python -m app   (from the fastapi directory)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import chat_completions, graph

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    db_url = (settings.database_url or "").strip()

    if db_url:
        from app.services.postgres import get_pool, init_postgres, close_postgres
        from app.services.pulsecast_session_store import (
            PostgresSessionStore,
            set_session_store_impl,
        )
        from app.services.pulsecast_resume_store import (
            PostgresResumeStore,
            set_resume_store_impl,
        )
        from app.graph.pulsecast_graph import reset_compiled_graph

        await init_postgres(db_url)
        pool = get_pool()

        set_session_store_impl(
            PostgresSessionStore(pool, ttl_seconds=settings.pulsecast_session_ttl_seconds)
        )
        set_resume_store_impl(PostgresResumeStore(pool))

        reset_compiled_graph()
        logger.info("Postgres persistence enabled (sessions, resume tokens, graph checkpoints)")
    else:
        logger.info("DATABASE_URL not set — using in-memory stores (single-process only)")

    yield

    if db_url:
        from app.services.postgres import close_postgres
        await close_postgres()


app = FastAPI(title="Pulsecast API", version="0.1.0", lifespan=lifespan)

_settings = get_settings()
_origins = [o.strip() for o in _settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Pulsecast-Stream-Job-Id"],
)

app.include_router(chat_completions.router)
app.include_router(graph.router)


def _openai_error(message: str, *, err_type: str, code: str | None = None, status_code: int = 400):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": err_type,
                "param": None,
                "code": code,
            }
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    err_type = "invalid_request_error" if exc.status_code < 500 else "server_error"
    return _openai_error(msg, err_type=err_type, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    return _openai_error(str(exc), err_type="invalid_request_error", code="validation_error", status_code=422)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
