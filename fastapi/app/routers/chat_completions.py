from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.models.schemas import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIModelsListResponse,
    OpenAIModelCard,
    PulsecastChatRefineRequest,
    PulsecastChatResumeRequest,
    PulsecastStreamControlRequest,
)
from app.services.chat_completions import stream_completion_sse, stream_refine_sse, stream_resume_sse
from app.services.stream_pause_store import stream_pause_store
from app.services.pulsecast_resume_store import (
    DuplicateSubQuestionPausedSnapshot,
    PulsecastPausedSnapshot,
    resume_store,
)
from app.services.sub_question_tts_guard import is_valid_tts_sub_question

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
    job_id = uuid.uuid4().hex
    return StreamingResponse(
        stream_completion_sse(settings=settings, req=body, job_id=job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Pulsecast-Stream-Job-Id": job_id,
        },
    )


@router.post("/v1/chat/completions/refine", response_model=OpenAIChatCompletionResponse)
async def chat_completions_refine(
    body: PulsecastChatRefineRequest,
    settings: Settings = Depends(get_settings),
):
    if not body.stream:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stream must be true for this service",
        )
    job_id = uuid.uuid4().hex
    return StreamingResponse(
        stream_refine_sse(settings=settings, req=body, job_id=job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Pulsecast-Stream-Job-Id": job_id,
        },
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
    snapshot = await resume_store.pop(body.resume_token)
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid or expired resume_token",
        )
    # Reject invalid text-to-SQL sub-questions before StreamingResponse: once the stream is opened the
    # client may already have received 200, so HTTPException from inside the generator cannot become 422.
    if body.approved and isinstance(
        snapshot, (DuplicateSubQuestionPausedSnapshot, PulsecastPausedSnapshot)
    ):
        sub_q = (body.edited_question or "").strip() or snapshot.proposed_sub_question
        if not is_valid_tts_sub_question(sub_q):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "invalid_sub_question_for_text_to_sql": True,
                    "sub_question": sub_q,
                    "message": "Question must be a single declarative analytics ask; edit and try again.",
                },
            )
    job_id = uuid.uuid4().hex
    return StreamingResponse(
        stream_resume_sse(settings=settings, req=body, snapshot=snapshot, job_id=job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Pulsecast-Stream-Job-Id": job_id,
        },
    )


@router.post("/v1/chat/completions/stream-control")
async def pulsecast_stream_control(
    body: PulsecastStreamControlRequest,
) -> Response:
    try:
        await stream_pause_store.set_paused(body.job_id, body.paused, body.user)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown or expired stream job_id",
        )
    except PermissionError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user does not match this stream job",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
