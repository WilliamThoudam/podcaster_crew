from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routers import chat_completions

app = FastAPI(title="Pulsecast API", version="0.1.0")

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
