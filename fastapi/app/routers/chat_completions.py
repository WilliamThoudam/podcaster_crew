from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.models.schemas import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIModelsListResponse,
    OpenAIModelCard,
)
from app.services.chat_completions import create_non_stream_response, stream_completion_sse

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
    if body.stream:
        return StreamingResponse(
            stream_completion_sse(settings=settings, req=body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    return await create_non_stream_response(settings=settings, req=body)
