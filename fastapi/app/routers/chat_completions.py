from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.models.schemas import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIModelsListResponse,
    OpenAIModelCard,
    PulsecastChatResumeRequest,
)
from app.services.chat_completions import stream_completion_sse, stream_resume_sse
from app.services.pulsecast_resume_store import resume_store

router = APIRouter(tags=["openai-compatible"])


@router.get("/v1/models", response_model=OpenAIModelsListResponse)
async def list_models() -> OpenAIModelsListResponse:
    return OpenAIModelsListResponse(
        data=[
            OpenAIModelCard(id="pulsecast-qa"),
        ]
    )


@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def chat_completions(
    body: OpenAIChatCompletionRequest,
    settings: Settings = Depends(get_settings),
):
    if not body.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream must be true for this service",
        )
    return StreamingResponse(
        stream_completion_sse(settings=settings, req=body),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/v1/chat/completions/resume", response_model=OpenAIChatCompletionResponse)
async def chat_completions_resume(
    body: PulsecastChatResumeRequest,
    settings: Settings = Depends(get_settings),
):
    if not body.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream must be true for this service",
        )
    snapshot = resume_store.pop(body.resume_token)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired resume_token",
        )
    return StreamingResponse(
        stream_resume_sse(settings=settings, req=body, snapshot=snapshot),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
