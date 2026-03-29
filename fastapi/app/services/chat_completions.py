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
    OpenAIChatMessage,
    PulsecastChatResumeRequest,
    SubResult,
    TextToSqlResponse,
)
from app.services.pulsecast_completion_steps import (
    phase_agents_finalize,
    run_sub_questions_slice,
    sub_question_tts_messages,
    to_markdown_table,
)
from app.services.pulsecast_completion_types import (
    CompletionStreamComplete,
    CompletionStreamOutcome,
    CompletionStreamPaused,
    ProgressCallback,
)
from app.services.pulsecast_llm_agents import (
    LlmAgentsPaused,
    LlmAgentsPausedWebSearch,
    run_llm_agents_after_web_hitl,
    run_llm_agents_host_only,
)
from app.services.pulsecast_resume_store import (
    DuplicateSubQuestionPausedSnapshot,
    PulsecastPausedSnapshot,
    PulsecastPausedSnapshotUnion,
    WebSearchPausedSnapshot,
    resume_store,
)
from app.services.sub_question_tts_guard import is_valid_tts_sub_question
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
    "build_resume_dispatcher",
    "build_resume_duplicate_sub_question_payload",
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
    prior_same_sql = next((sr for sr in sub_results if sr.generated_sql == sql), None)

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
    if prior_same_sql is not None:
        await emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": total,
                "sub_question": sub_q,
            },
            text="Reusing result from an earlier sub-question (identical SQL).",
            event_type="execute_generating_chunk",
        )
        exe = prior_same_sql.execute
    else:
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


def _openai_request_for_duplicate_snapshot(snap: DuplicateSubQuestionPausedSnapshot) -> OpenAIChatCompletionRequest:
    return OpenAIChatCompletionRequest(
        model="pulsecast-qa",
        stream=True,
        messages=[OpenAIChatMessage(role="user", content=snap.question)],
        user=snap.openai_user,
    )


async def build_resume_duplicate_sub_question_payload(
    *,
    settings: Settings,
    snapshot: DuplicateSubQuestionPausedSnapshot,
    approved: bool,
    edited_question: str | None,
    on_progress: ProgressCallback | None = None,
) -> CompletionStreamOutcome:
    """Resume after duplicate-SQL HITL: finish sub-questions, then agent panel."""
    openai_req = _openai_request_for_duplicate_snapshot(snapshot)

    if not approved:
        await emit_progress(on_progress, {"type": "duplicate_sub_question_skipped"})
        sub_out = await run_sub_questions_slice(
            settings=settings,
            req=openai_req,
            question=snapshot.question,
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            on_progress=on_progress,
            start_index=snapshot.pending_index + 1,
            sub_results=list(snapshot.sub_results),
            primary_sql=snapshot.primary_sql,
            primary_exe=snapshot.primary_exe,
            first_sub_question_override=None,
            duplicate_fail_index=None,
        )
        if isinstance(sub_out, CompletionStreamPaused):
            return sub_out
        sub_results, primary_sql, primary_exe = sub_out
        return await phase_agents_finalize(
            settings=settings,
            req=openai_req,
            question=snapshot.question,
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            primary_sql=primary_sql,
            primary_exe=primary_exe,
            sub_results=sub_results,
            on_progress=on_progress,
        )

    sub_q = (edited_question or "").strip() or snapshot.proposed_sub_question
    if not is_valid_tts_sub_question(sub_q):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "invalid_sub_question_for_text_to_sql": True,
                "sub_question": sub_q,
                "message": "Question must be a single declarative analytics ask; edit and try again.",
            },
        )

    sub_out = await run_sub_questions_slice(
        settings=settings,
        req=openai_req,
        question=snapshot.question,
        host_plan=snapshot.host_plan,
        analyst_plan=snapshot.analyst_plan,
        on_progress=on_progress,
        start_index=snapshot.pending_index,
        sub_results=list(snapshot.sub_results),
        primary_sql=snapshot.primary_sql,
        primary_exe=snapshot.primary_exe,
        first_sub_question_override=sub_q,
        duplicate_fail_index=snapshot.pending_index,
    )
    if isinstance(sub_out, CompletionStreamPaused):
        return sub_out
    sub_results, primary_sql, primary_exe = sub_out
    return await phase_agents_finalize(
        settings=settings,
        req=openai_req,
        question=snapshot.question,
        host_plan=snapshot.host_plan,
        analyst_plan=snapshot.analyst_plan,
        primary_sql=primary_sql,
        primary_exe=primary_exe,
        sub_results=sub_results,
        on_progress=on_progress,
    )


async def build_resume_web_search_payload(
    *,
    settings: Settings,
    snapshot: WebSearchPausedSnapshot,
    approved: bool,
    edited_question: str | None,
    on_progress: ProgressCallback | None = None,
) -> CompletionStreamOutcome:
    """After web-search HITL: Serper (if approved), Challenger + Host; may pause again on SQL follow-up."""
    if not approved:
        await emit_progress(on_progress, {"type": "web_search_declined"})

    agents_out = await run_llm_agents_after_web_hitl(
        settings=settings,
        question=snapshot.question,
        generated_sql=snapshot.generated_sql,
        exe=snapshot.primary_exe,
        deterministic_summary=snapshot.deterministic_summary,
        sub_results=list(snapshot.sub_results),
        host_plan=snapshot.host_plan,
        analyst_plan=snapshot.analyst_plan,
        discussion=snapshot.discussion,
        pipeline=snapshot.pipeline,
        approved=approved,
        edited_search_query=edited_question,
        proposed_search_query=snapshot.proposed_search_query,
        search_queries=list(snapshot.search_queries),
        pending_search_index=snapshot.pending_search_index,
        completed_web_results=list(snapshot.completed_web_results),
        hitl_rationale=snapshot.rationale,
        allow_sql_approval_pause=True,
        on_progress=on_progress,
    )
    if isinstance(agents_out, LlmAgentsPausedWebSearch):
        snap_ws = WebSearchPausedSnapshot(
            pipeline=agents_out.pipeline,
            discussion=agents_out.discussion,
            question=agents_out.question,
            generated_sql=agents_out.generated_sql,
            primary_exe=agents_out.primary_exe,
            deterministic_summary=agents_out.deterministic_summary,
            sub_results=agents_out.sub_results,
            host_plan=agents_out.host_plan,
            analyst_plan=agents_out.analyst_plan,
            proposed_search_query=agents_out.proposed_search_query,
            search_queries=agents_out.search_queries,
            pending_search_index=agents_out.pending_search_index,
            completed_web_results=agents_out.completed_web_results,
            rationale=agents_out.rationale,
            openai_user=snapshot.openai_user,
        )
        token_ws = resume_store.issue_token(snap_ws)
        n = len(agents_out.search_queries)
        step = agents_out.pending_search_index + 1
        await emit_progress(
            on_progress,
            {
                "type": "web_search_approval_required",
                "resume_token": token_ws,
                "proposed_search_query": agents_out.proposed_search_query,
                "rationale": agents_out.rationale,
                "pause_kind": "web_search",
                "web_search_step_index": step,
                "web_search_total_steps": max(1, n),
            },
        )
        return CompletionStreamPaused(
            resume_token=token_ws,
            proposed_sub_question=agents_out.proposed_search_query,
            rationale=agents_out.rationale,
        )
    if isinstance(agents_out, LlmAgentsPaused):
        snap2 = PulsecastPausedSnapshot(
            pipeline=agents_out.pipeline,
            discussion=agents_out.discussion,
            question=agents_out.question,
            generated_sql=agents_out.generated_sql,
            primary_exe=agents_out.primary_exe,
            deterministic_summary=agents_out.deterministic_summary,
            sub_results=agents_out.sub_results,
            host_plan=agents_out.host_plan,
            analyst_plan=agents_out.analyst_plan,
            proposed_sub_question=agents_out.proposed_sub_question,
            rationale=agents_out.rationale,
            openai_user=snapshot.openai_user,
        )
        token = resume_store.issue_token(snap2)
        await emit_progress(
            on_progress,
            {
                "type": "sql_approval_required",
                "resume_token": token,
                "proposed_sub_question": agents_out.proposed_sub_question,
                "rationale": agents_out.rationale,
                "pause_kind": "challenger_followup",
            },
        )
        return CompletionStreamPaused(
            resume_token=token,
            proposed_sub_question=agents_out.proposed_sub_question,
            rationale=agents_out.rationale,
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
    return CompletionStreamComplete(content=agents_out.answer)


async def build_resume_dispatcher(
    *,
    settings: Settings,
    snapshot: PulsecastPausedSnapshotUnion,
    req: PulsecastChatResumeRequest,
    on_progress: ProgressCallback | None = None,
) -> CompletionStreamOutcome:
    if isinstance(snapshot, WebSearchPausedSnapshot):
        return await build_resume_web_search_payload(
            settings=settings,
            snapshot=snapshot,
            approved=req.approved,
            edited_question=req.edited_question,
            on_progress=on_progress,
        )
    if isinstance(snapshot, DuplicateSubQuestionPausedSnapshot):
        return await build_resume_duplicate_sub_question_payload(
            settings=settings,
            snapshot=snapshot,
            approved=req.approved,
            edited_question=req.edited_question,
            on_progress=on_progress,
        )
    return await build_resume_payload(
        settings=settings,
        snapshot=snapshot,
        approved=req.approved,
        edited_question=req.edited_question,
        on_progress=on_progress,
    )


async def stream_resume_sse(
    *,
    settings: Settings,
    req: PulsecastChatResumeRequest,
    snapshot: PulsecastPausedSnapshotUnion,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    progress_q: asyncio.Queue[str] = asyncio.Queue()
    started = False

    async def on_progress(event: dict[str, Any]) -> None:
        marker = f"<<PULSECAST_PROGRESS:{json.dumps(event, ensure_ascii=False)}>>"
        await progress_q.put(marker)

    task = asyncio.create_task(
        build_resume_dispatcher(
            settings=settings,
            snapshot=snapshot,
            req=req,
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
    except HTTPException as e:
        # StreamingResponse may have already sent 200 before the first chunk; never re-raise or Starlette
        # raises RuntimeError("Caught handled exception, but response already started.").
        err_payload = json.dumps(
            {
                "pulsecast_http_error": True,
                "status_code": e.status_code,
                "detail": e.detail,
            },
            ensure_ascii=False,
        )
        if not started:
            started = True
            yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=f'<<PULSECAST_HTTP_ERROR:{err_payload}>>')}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"
        return
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
    except HTTPException as e:
        err_payload = json.dumps(
            {
                "pulsecast_http_error": True,
                "status_code": e.status_code,
                "detail": e.detail,
            },
            ensure_ascii=False,
        )
        if not started:
            started = True
            yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=f'<<PULSECAST_HTTP_ERROR:{err_payload}>>')}\n\n"
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"
        return
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
