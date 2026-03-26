from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator

import httpx
from fastapi import HTTPException, status

from app.clients.execute_sql import execute_sql as execute_sql_client
from app.clients.text_to_sql import generate_sql
from app.config import Settings
from app.models.schemas import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionDelta,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChoice,
    OpenAIChatMessage,
    OpenAIUsage,
)
from app.models.schemas import TextToSqlResponse
from app.services.pulsecast_llm_agents import run_llm_agents
from app.services.qa_pipeline import build_answer_summary, validate_and_normalize_sql


def _extract_last_user_question(req: OpenAIChatCompletionRequest) -> str:
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="messages must include at least one non-empty user message",
    )


def _as_tts_messages(req: OpenAIChatCompletionRequest) -> list[dict[str, str]]:
    # Keep upstream simple and safe; pass only role/content.
    return [{"role": m.role, "content": m.content} for m in req.messages if m.content.strip()]


def _payload_json_string(*, answer: str, generated_sql: str, execute_payload: dict, pipeline: list, agent_messages: list) -> str:
    return json.dumps(
        {
            "answer": answer,
            "generated_sql": generated_sql,
            "execute": execute_payload,
            "pipeline": pipeline,
            "agent_messages": agent_messages,
        },
        ensure_ascii=False,
    )


def _chunk_json(
    *,
    completion_id: str,
    model: str,
    now: int,
    content: str | None = None,
    role: str | None = None,
    finish_reason: str | None = None,
) -> str:
    chunk = OpenAIChatCompletionChunk(
        id=completion_id,
        created=now,
        model=model,
        choices=[
            OpenAIChatCompletionChunkChoice(
                index=0,
                delta=OpenAIChatCompletionDelta(
                    role="assistant" if role == "assistant" else None,
                    content=content,
                ),
                finish_reason=finish_reason,
            )
        ],
    )
    return chunk.model_dump_json()


async def build_completion_payload(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
) -> tuple[str, str]:
    question = _extract_last_user_question(req)
    user_id = settings.default_user_id
    user_db_id = settings.default_user_db_id
    db_type = settings.default_db_type
    schema_name = settings.default_schema_name
    model = settings.default_model
    max_nodes = settings.default_max_nodes

    tts: TextToSqlResponse
    try:
        tts = await generate_sql(
            settings=settings,
            question=question,
            messages=_as_tts_messages(req),
            user_id=user_id,
            user_db_id=user_db_id,
            db_type=db_type,
            schema_name=schema_name,
            session_id=req.user,
            model=model,
            max_nodes=max_nodes,
            is_retry=False,
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Text-to-SQL service error: {e}",
        ) from e

    if tts.error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"text_to_sql_error": tts.error, "generated_sql": tts.generated_sql},
        )

    sql = validate_and_normalize_sql((tts.generated_sql or "").strip())
    try:
        exe = await execute_sql_client(
            settings=settings,
            sql=sql,
            user_id=user_id,
            user_db_id=user_db_id,
            db_type=db_type,
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Execute SQL service error: {e}",
        ) from e

    if not exe.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"generated_sql": sql, "execute": exe.model_dump()},
        )

    deterministic = build_answer_summary(exe)
    answer, pipeline, agent_messages = await run_llm_agents(
        settings=settings,
        question=question,
        generated_sql=sql,
        exe=exe,
        deterministic_summary=deterministic,
    )

    content = _payload_json_string(
        answer=answer,
        generated_sql=sql,
        execute_payload=exe.model_dump(),
        pipeline=[p.model_dump() for p in pipeline],
        agent_messages=[m.model_dump() for m in agent_messages],
    )
    return answer, content


async def create_non_stream_response(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
) -> OpenAIChatCompletionResponse:
    _answer, content = await build_completion_payload(settings=settings, req=req)
    now = int(time.time())
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    return OpenAIChatCompletionResponse(
        id=completion_id,
        created=now,
        model=req.model,
        choices=[
            OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIChatMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
        usage=OpenAIUsage(),
    )


async def stream_completion_sse(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
) -> AsyncIterator[str]:
    _answer, content = await build_completion_payload(settings=settings, req=req)
    now = int(time.time())
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=now, role='assistant')}\n\n"
    # Use tiny chunks + tiny pacing so clients render gradual typing
    # instead of receiving one large buffered payload.
    step = 12
    for i in range(0, len(content), step):
        part = content[i : i + step]
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=part)}\n\n"
        await asyncio.sleep(0.015)

    yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
    yield "data: [DONE]\n\n"
