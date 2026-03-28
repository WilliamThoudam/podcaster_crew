from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncIterator, Awaitable, Callable

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
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
    TextToSqlResponse,
)
from app.services.pulsecast_llm_agents import run_llm_agents
from app.services.pulsecast_planning import run_analyst_planner, run_host_planner
from app.services.qa_pipeline import build_answer_summary, validate_and_normalize_sql

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

# -----------------------------------------------------------------------------
# SSE progress stream pacing — used for every _emit_text_chunks step.
# -----------------------------------------------------------------------------
STREAM_CHUNK_SIZE = 2
STREAM_DELAY_S = 0.001


def _extract_last_user_question(req: OpenAIChatCompletionRequest) -> str:
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="messages must include at least one non-empty user message",
    )


def _sub_question_tts_messages(
    *,
    sub_question: str,
    retry_for_duplicate: bool = False,
) -> list[dict[str, str]]:
    system = (
        "You convert one analytics question into SQL for the configured warehouse.\n"
        "Use only the provided sub-question as the target intent.\n"
        "Return SQL for that intent only."
    )
    user = sub_question.strip()
    if retry_for_duplicate:
        user = (
            f"{user}\n\n"
            "Retry instruction: previous SQL looked duplicated from another sub-question. "
            "Generate SQL that is specific to this question intent."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _escape_md_cell(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.replace("|", "\\|")


def _to_markdown_table(rows: list[dict[str, Any]], max_rows: int = 20) -> str:
    if not rows:
        return "_No rows returned._"
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(_escape_md_cell(h) for h in headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = []
    for row in rows[:max_rows]:
        body_rows.append(
            "| " + " | ".join(_escape_md_cell(row.get(h)) for h in headers) + " |"
        )
    return "\n".join([header_row, sep_row, *body_rows])


def _md_table_header(headers: list[str]) -> str:
    header_row = "| " + " | ".join(_escape_md_cell(h) for h in headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([header_row, sep_row])


def _md_table_row(headers: list[str], row: dict[str, Any]) -> str:
    return "| " + " | ".join(_escape_md_cell(row.get(h)) for h in headers) + " |"


async def _emit_text_chunks(
    *,
    on_progress: ProgressCallback | None,
    base_event: dict[str, Any],
    text: str,
    chunk_size: int = STREAM_CHUNK_SIZE,
    event_type: str = "tts_sql_chunk",
    delay_s: float = STREAM_DELAY_S,
) -> None:
    s = text or ""
    for i in range(0, len(s), chunk_size):
        await _emit_progress(
            on_progress,
            {**base_event, "type": event_type, "chunk": s[i : i + chunk_size]},
        )
        await asyncio.sleep(delay_s)


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


async def _emit_progress(
    cb: ProgressCallback | None,
    event: dict[str, Any],
) -> None:
    if cb is None:
        return
    maybe = cb(event)
    if asyncio.iscoroutine(maybe):
        await maybe


async def build_completion_payload(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    on_progress: ProgressCallback | None = None,
) -> tuple[str, str]:
    question = _extract_last_user_question(req)
    user_id = settings.default_user_id
    user_db_id = settings.default_user_db_id
    db_type = settings.default_db_type
    schema_name = settings.default_schema_name
    model = settings.default_model
    max_nodes = settings.default_max_nodes

    # 1) Planning stage: Host (streamed to client) then Analyst sub_questions.
    host_plan: PlanningHostOutput = await run_host_planner(settings=settings, question=question)
    await _emit_progress(on_progress, {"type": "host_plan_started"})
    host_line = (host_plan.primary_focus or "").strip() or question
    await _emit_text_chunks(
        on_progress=on_progress,
        base_event={},
        text=host_line,
        event_type="host_plan_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await _emit_progress(on_progress, {"type": "host_plan_done"})

    analyst_plan: PlanningAnalystOutput = await run_analyst_planner(
        settings=settings,
        question=question,
        host=host_plan,
    )
    total_sub = len(analyst_plan.sub_questions)
    await _emit_progress(on_progress, {"type": "planned_sub_questions_started", "total": total_sub})
    plan_md = "To answer this, I will break it down into steps:\n\n" + "\n".join(
        f"{i + 1}. {sq}" for i, sq in enumerate(analyst_plan.sub_questions)
    )
    await _emit_text_chunks(
        on_progress=on_progress,
        base_event={"total": total_sub},
        text=plan_md,
        event_type="planned_sub_questions_chunk",
    )
    await _emit_progress(on_progress, {"type": "planned_sub_questions_done", "total": total_sub})

    # 2) Loop over sub_questions, run text-to-SQL + execute for each.
    sub_results: list[SubResult] = []
    primary_sql: str | None = None
    primary_exe = None

    for idx, sub_q in enumerate(analyst_plan.sub_questions):
        await _emit_progress(
            on_progress,
            {
                "type": "sub_question_start",
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
            },
        )
        tts: TextToSqlResponse
        try:
            await _emit_progress(
                on_progress,
                {
                    "type": "tts_started",
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
            )
            # Stream the "Text-to-SQL N/total: <sub_question>" label chunk by chunk
            await _emit_text_chunks(
                on_progress=on_progress,
                base_event={
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
                text=f"Text-to-SQL {idx + 1}/{len(analyst_plan.sub_questions)}: {sub_q}",
                event_type="tts_label_chunk",
            )
            # After the full label, stream "Generating…" on the next line (separate phase)
            await _emit_text_chunks(
                on_progress=on_progress,
                base_event={
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
                text="Generating…",
                event_type="tts_generating_chunk",
            )
            tts = await generate_sql(
                settings=settings,
                question=sub_q,
                messages=_sub_question_tts_messages(
                    sub_question=sub_q,
                ),
                user_id=user_id,
                user_db_id=user_db_id,
                db_type=db_type,
                schema_name=schema_name,
                session_id=req.user,
                model=model,
                max_nodes=max_nodes,
                is_retry=False,
                trace_context={
                    "sub_question_index": idx + 1,
                    "sub_question_total": len(analyst_plan.sub_questions),
                    "sub_question_text": sub_q,
                    "planner_mode": "analyst_decomposition",
                },
            )
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Text-to-SQL service error for sub_question {idx + 1}: {e}",
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
        await _emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
            },
            text=sql,
            event_type="tts_sql_chunk",
        )
        await _emit_progress(
            on_progress,
            {
                "type": "tts_done",
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
            },
        )
        duplicate_sql = any(sr.generated_sql == sql for sr in sub_results)
        if duplicate_sql:
            await _emit_progress(
                on_progress,
                {
                    "type": "sub_question_retry",
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                    "reason": "duplicate_sql_detected",
                },
            )
            try:
                retry_tts = await generate_sql(
                    settings=settings,
                    question=sub_q,
                    messages=_sub_question_tts_messages(
                        sub_question=sub_q,
                        retry_for_duplicate=True,
                    ),
                    user_id=user_id,
                    user_db_id=user_db_id,
                    db_type=db_type,
                    schema_name=schema_name,
                    session_id=req.user,
                    model=model,
                    max_nodes=max_nodes,
                    is_retry=True,
                    trace_context={
                        "sub_question_index": idx + 1,
                        "sub_question_total": len(analyst_plan.sub_questions),
                        "sub_question_text": sub_q,
                        "planner_mode": "analyst_decomposition",
                        "retry_reason": "duplicate_sql",
                    },
                )
                if not retry_tts.error:
                    retry_sql = validate_and_normalize_sql((retry_tts.generated_sql or "").strip())
                    if retry_sql and not any(sr.generated_sql == retry_sql for sr in sub_results):
                        sql = retry_sql
            except httpx.HTTPError:
                # Keep the first successful SQL if retry transport fails.
                pass
        try:
            await _emit_progress(
                on_progress,
                {
                    "type": "execute_started",
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
            )
            # Stream "Executing N/total: <sub_question>" first, then "Executing…" on the next line
            await _emit_text_chunks(
                on_progress=on_progress,
                base_event={
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
                text=f"Executing {idx + 1}/{len(analyst_plan.sub_questions)}: {sub_q}",
                event_type="execute_label_chunk",
            )
            await _emit_text_chunks(
                on_progress=on_progress,
                base_event={
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
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
                detail=f"Execute SQL service error for sub_question {idx + 1}: {e}",
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

        if primary_sql is None:
            primary_sql = sql
            primary_exe = exe

        sub_results.append(
            SubResult(
                sub_question=sub_q,
                generated_sql=sql,
                execute=exe,
            )
        )
        table_md = _to_markdown_table((exe.data or [])[:20])
        await _emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
            },
            text=table_md,
            event_type="execute_table_chunk",
        )
        await _emit_progress(
            on_progress,
            {
                "type": "execute_done",
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
                "row_count": exe.rowCount,
            },
        )
        await _emit_progress(
            on_progress,
            {
                "type": "sub_question_done",
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
                "row_count": exe.rowCount,
            },
        )

    # Safety: ensure we have at least one successful result.
    if primary_sql is None or primary_exe is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No successful sub-question results were produced",
        )

    # 3) Deterministic summary, then multi-agent reasoning (stream "Summarizing…" first).
    deterministic = build_answer_summary(primary_exe)
    await _emit_progress(on_progress, {"type": "summarizing_started"})
    await _emit_text_chunks(
        on_progress=on_progress,
        base_event={},
        text="Summarizing…",
        event_type="summarizing_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await _emit_progress(on_progress, {"type": "summarizing_done"})
    answer, pipeline, agent_messages = await run_llm_agents(
        settings=settings,
        question=question,
        generated_sql=primary_sql,
        exe=primary_exe,
        deterministic_summary=deterministic,
        sub_results=sub_results,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
    )
    # Stream contract is markdown-first: assistant content should be plain markdown text.
    return answer, answer


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
        _answer, content = await task
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

    # Same chunk size + delay as PULSECAST_PROGRESS streams (see STREAM_* at top of module).
    for i in range(0, len(content), STREAM_CHUNK_SIZE):
        part = content[i : i + STREAM_CHUNK_SIZE]
        yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=part)}\n\n"
        await asyncio.sleep(STREAM_DELAY_S)

    yield f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n"
    yield "data: [DONE]\n\n"
