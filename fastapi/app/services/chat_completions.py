from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import HTTPException, status

from app.clients.execute_sql import execute_sql as execute_sql_client
from app.clients.text_to_sql import generate_sql
from app.config import Settings
from app.graph.pulsecast_graph import run_pulsecast_completion_graph
from app.models.schemas import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionDelta,
    OpenAIChatCompletionRequest,
    PulsecastChatResumeRequest,
    SubResult,
    TextToSqlResponse,
)
from app.services.pulsecast_completion_steps import sub_question_tts_messages, to_markdown_table
from app.services.pulsecast_completion_types import (
    CompletionStreamComplete,
    CompletionStreamOutcome,
    CompletionStreamPaused,
    ProgressCallback,
)
from app.services.pulsecast_llm_agents import run_llm_agents_host_only
from app.services.pulsecast_resume_store import PulsecastPausedSnapshot
from app.services.pulsecast_sse_emit import (
    STREAM_CHUNK_SIZE,
    STREAM_DELAY_S,
    emit_progress,
    emit_text_chunks,
)
from app.services.qa_pipeline import build_answer_summary, validate_and_normalize_sql

# Re-export for routers / tests that imported from chat_completions
__all__ = [
    "CompletionStreamComplete",
    "CompletionStreamOutcome",
    "CompletionStreamPaused",
    "build_completion_payload",
    "build_resume_payload",
    "stream_completion_sse",
    "stream_resume_sse",
]


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
    on_progress: ProgressCallback | None = None,
) -> CompletionStreamOutcome:
    return await run_pulsecast_completion_graph(
        settings=settings,
        req=req,
        on_progress=on_progress,
    )


async def build_resume_payload(
    *,
    settings: Settings,
    snapshot: PulsecastPausedSnapshot,
    approved: bool,
    edited_question: str | None,
    on_progress: ProgressCallback | None = None,
) -> CompletionStreamComplete:
    """After HITL: optional follow-up SQL + HOST composer; always returns final markdown."""
    user_id = settings.default_user_id
    user_db_id = settings.default_user_db_id
    db_type = settings.default_db_type
    schema_name = settings.default_schema_name
    model = settings.default_model
    max_nodes = settings.default_max_nodes
    sub_results = list(snapshot.sub_results)

    if not approved:
        await emit_progress(on_progress, {"type": "sql_followup_declined"})
        agents_done = await run_llm_agents_host_only(
            settings=settings,
            question=snapshot.question,
            generated_sql=snapshot.generated_sql,
            exe=snapshot.primary_exe,
            deterministic_summary=snapshot.deterministic_summary,
            sub_results=sub_results,
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            discussion=snapshot.discussion,
            pipeline=snapshot.pipeline,
            user_declined_extra_sql=True,
        )
        return CompletionStreamComplete(content=agents_done.answer)

    sub_q = (edited_question or "").strip() or snapshot.proposed_sub_question
    idx = 0
    total = 1

    await emit_progress(
        on_progress,
        {
            "type": "sub_question_start",
            "index": idx + 1,
            "total": total,
            "sub_question": sub_q,
        },
    )
    try:
        await emit_progress(
            on_progress,
            {
                "type": "tts_started",
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
            },
        )
        await emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
            },
            text=f"Text-to-SQL {idx + 1}/{total}: {sub_q}",
            event_type="tts_label_chunk",
        )
        await emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
            },
            text="Generating…",
            event_type="tts_generating_chunk",
        )
        tts = await generate_sql(
            settings=settings,
            question=sub_q,
            messages=sub_question_tts_messages(sub_question=sub_q),
            user_id=user_id,
            user_db_id=user_db_id,
            db_type=db_type,
            schema_name=schema_name,
            session_id=snapshot.openai_user,
            model=model,
            max_nodes=max_nodes,
            is_retry=False,
            trace_context={
                "sub_question_index": idx + 1,
                "sub_question_total": total,
                "sub_question_text": sub_q,
                "planner_mode": "hitl_followup",
            },
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Text-to-SQL service error for follow-up SQL: {e}",
        ) from e

    if tts.error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "text_to_sql_error": tts.error,
                "generated_sql": tts.generated_sql,
                "sub_question": sub_q,
            },
        )

    sql = validate_and_normalize_sql((tts.generated_sql or "").strip())
    await emit_text_chunks(
        on_progress=on_progress,
        base_event={
            "index": idx + 1,
            "total": total,
            "sub_question": sub_q,
        },
        text=sql,
        event_type="tts_sql_chunk",
    )
    await emit_progress(
        on_progress,
        {
            "type": "tts_done",
            "index": idx + 1,
            "total": total,
            "sub_question": sub_q,
        },
    )
    duplicate_sql = any(sr.generated_sql == sql for sr in sub_results)
    if duplicate_sql:
        await emit_progress(
            on_progress,
            {
                "type": "sub_question_retry",
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
                "reason": "duplicate_sql_detected",
            },
        )
        try:
            retry_tts = await generate_sql(
                settings=settings,
                question=sub_q,
                messages=sub_question_tts_messages(sub_question=sub_q, retry_for_duplicate=True),
                user_id=user_id,
                user_db_id=user_db_id,
                db_type=db_type,
                schema_name=schema_name,
                session_id=snapshot.openai_user,
                model=model,
                max_nodes=max_nodes,
                is_retry=True,
                trace_context={
                    "sub_question_index": idx + 1,
                    "sub_question_total": total,
                    "sub_question_text": sub_q,
                    "planner_mode": "hitl_followup",
                    "retry_reason": "duplicate_sql",
                },
            )
            if not retry_tts.error:
                retry_sql = validate_and_normalize_sql((retry_tts.generated_sql or "").strip())
                if retry_sql and not any(sr.generated_sql == retry_sql for sr in sub_results):
                    sql = retry_sql
        except httpx.HTTPError:
            pass

    try:
        await emit_progress(
            on_progress,
            {
                "type": "execute_started",
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
            },
        )
        await emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
            },
            text=f"Executing {idx + 1}/{total}: {sub_q}",
            event_type="execute_label_chunk",
        )
        await emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
            },
            text="Executing…",
            event_type="execute_generating_chunk",
        )
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
            detail=f"Execute SQL service error for follow-up SQL: {e}",
        ) from e

    if not exe.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "generated_sql": sql,
                "execute": exe.model_dump(),
                "sub_question": sub_q,
            },
        )

    sub_results.append(
        SubResult(
            sub_question=sub_q,
            generated_sql=sql,
            execute=exe,
        )
    )
    table_md = to_markdown_table((exe.data or [])[:20])
    await emit_text_chunks(
        on_progress=on_progress,
        base_event={
            "index": idx + 1,
            "total": total,
            "sub_question": sub_q,
        },
        text=table_md,
        event_type="execute_table_chunk",
    )
    await emit_progress(
        on_progress,
        {
            "type": "execute_done",
            "index": idx + 1,
            "total": total,
            "sub_question": sub_q,
        },
    )
    await emit_progress(
        on_progress,
        {
            "type": "sub_question_done",
            "index": idx + 1,
            "total": total,
            "sub_question": sub_q,
        },
    )

    await emit_progress(on_progress, {"type": "summarizing_started"})
    await emit_text_chunks(
        on_progress=on_progress,
        base_event={},
        text="Summarizing…",
        event_type="summarizing_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await emit_progress(on_progress, {"type": "summarizing_done"})

    agents_done = await run_llm_agents_host_only(
        settings=settings,
        question=snapshot.question,
        generated_sql=snapshot.generated_sql,
        exe=snapshot.primary_exe,
        deterministic_summary=snapshot.deterministic_summary,
        sub_results=sub_results,
        host_plan=snapshot.host_plan,
        analyst_plan=snapshot.analyst_plan,
        discussion=snapshot.discussion,
        pipeline=snapshot.pipeline,
        user_declined_extra_sql=False,
    )
    return CompletionStreamComplete(content=agents_done.answer)


async def stream_resume_sse(
    *,
    settings: Settings,
    req: PulsecastChatResumeRequest,
    snapshot: PulsecastPausedSnapshot,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    progress_q: asyncio.Queue[str] = asyncio.Queue()
    started = False

    async def on_progress(event: dict[str, Any]) -> None:
        marker = f"<<PULSECAST_PROGRESS:{json.dumps(event, ensure_ascii=False)}>>"
        await progress_q.put(marker)

    task = asyncio.create_task(
        build_resume_payload(
            settings=settings,
            snapshot=snapshot,
            approved=req.approved,
            edited_question=req.edited_question,
            on_progress=on_progress,
        )
    )

    while True:
        if task.done() and progress_q.empty():
            break
        try:
            marker = await asyncio.wait_for(progress_q.get(), timeout=0.1)
        except TimeoutError:
            continue
        if not started:
            started = True
            yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=marker)}\n\n"

    try:
        outcome = await task
    except Exception as e:
        if not started:
            raise
        error_text = f"Failed to generate response: {e}"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=error_text)}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"
        return

    if not started:
        started = True
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n"

    content = outcome.content
    for i in range(0, len(content), STREAM_CHUNK_SIZE):
        part = content[i : i + STREAM_CHUNK_SIZE]
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=part)}\n\n"
        await asyncio.sleep(STREAM_DELAY_S)

    yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
    yield "data: [DONE]\n\n"


async def stream_completion_sse(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    progress_q: asyncio.Queue[str] = asyncio.Queue()
    started = False

    async def on_progress(event: dict[str, Any]) -> None:
        marker = f"<<PULSECAST_PROGRESS:{json.dumps(event, ensure_ascii=False)}>>"
        await progress_q.put(marker)

    task = asyncio.create_task(build_completion_payload(settings=settings, req=req, on_progress=on_progress))

    while True:
        if task.done() and progress_q.empty():
            break
        try:
            marker = await asyncio.wait_for(progress_q.get(), timeout=0.1)
        except TimeoutError:
            continue
        if not started:
            started = True
            yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=marker)}\n\n"

    try:
        outcome = await task
    except Exception as e:
        if not started:
            raise
        error_text = f"Failed to generate response: {e}"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=error_text)}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"
        return

    if isinstance(outcome, CompletionStreamPaused):
        if not started:
            started = True
            yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"
        return

    if not started:
        started = True
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n"

    content = outcome.content
    # Same chunk size + delay as PULSECAST_PROGRESS streams (see STREAM_* at top of module).
    for i in range(0, len(content), STREAM_CHUNK_SIZE):
        part = content[i : i + STREAM_CHUNK_SIZE]
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=part)}\n\n"
        await asyncio.sleep(STREAM_DELAY_S)

    yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
    yield "data: [DONE]\n\n"
