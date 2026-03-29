from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, status

from app.clients.execute_sql import execute_sql as execute_sql_client
from app.clients.text_to_sql import generate_sql
from app.config import Settings
from app.models.schemas import (
    ExecuteSqlResponse,
    OpenAIChatCompletionRequest,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
    TextToSqlResponse,
)
from app.services.pulsecast_completion_types import (
    CompletionStreamComplete,
    CompletionStreamOutcome,
    CompletionStreamPaused,
    ProgressCallback,
)
from app.services.pulsecast_llm_agents import LlmAgentsPaused, run_llm_agents
from app.services.pulsecast_planning import run_analyst_planner, run_host_planner
from app.services.pulsecast_resume_store import PulsecastPausedSnapshot, resume_store
from app.services.pulsecast_sse_emit import (
    STREAM_CHUNK_SIZE,
    STREAM_DELAY_S,
    emit_progress,
    emit_text_chunks,
)
from app.prompts.text_to_sql import (
    TEXT_TO_SQL_SUB_QUESTION_RETRY_SUFFIX,
    text_to_sql_sub_question_system_prompt,
)
from app.services.qa_pipeline import build_answer_summary, validate_and_normalize_sql


def extract_last_user_question(req: OpenAIChatCompletionRequest) -> str:
    for msg in reversed(req.messages):
        if msg.role == "user" and msg.content.strip():
            return msg.content.strip()
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="messages must include at least one non-empty user message",
    )


def sub_question_tts_messages(
    *,
    sub_question: str,
    retry_for_duplicate: bool = False,
) -> list[dict[str, str]]:
    system = text_to_sql_sub_question_system_prompt()
    user = sub_question.strip()
    if retry_for_duplicate:
        user = f"{user}\n\n{TEXT_TO_SQL_SUB_QUESTION_RETRY_SUFFIX}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def escape_md_cell(value: Any) -> str:
    s = "" if value is None else str(value)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.replace("|", "\\|")


def to_markdown_table(rows: list[dict[str, Any]], max_rows: int = 20) -> str:
    if not rows:
        return "_No rows returned._"
    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(escape_md_cell(h) for h in headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = []
    for row in rows[:max_rows]:
        body_rows.append(
            "| " + " | ".join(escape_md_cell(row.get(h)) for h in headers) + " |"
        )
    return "\n".join([header_row, sep_row, *body_rows])


async def phase_planning(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    on_progress: ProgressCallback | None,
) -> tuple[str, PlanningHostOutput, PlanningAnalystOutput]:
    question = extract_last_user_question(req)
    host_plan = await run_host_planner(settings=settings, question=question)
    await emit_progress(on_progress, {"type": "host_plan_started"})
    host_line = (host_plan.primary_focus or "").strip() or question
    await emit_text_chunks(
        on_progress=on_progress,
        base_event={},
        text=host_line,
        event_type="host_plan_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await emit_progress(on_progress, {"type": "host_plan_done"})

    analyst_plan = await run_analyst_planner(
        settings=settings,
        question=question,
        host=host_plan,
    )
    total_sub = len(analyst_plan.sub_questions)
    await emit_progress(on_progress, {"type": "planned_sub_questions_started", "total": total_sub})
    plan_md = "To answer this, I will break it down into steps:\n\n" + "\n".join(
        f"{i + 1}. {sq}" for i, sq in enumerate(analyst_plan.sub_questions)
    )
    await emit_text_chunks(
        on_progress=on_progress,
        base_event={"total": total_sub},
        text=plan_md,
        event_type="planned_sub_questions_chunk",
    )
    await emit_progress(on_progress, {"type": "planned_sub_questions_done", "total": total_sub})
    return question, host_plan, analyst_plan


async def phase_sub_questions(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    analyst_plan: PlanningAnalystOutput,
    on_progress: ProgressCallback | None,
) -> tuple[list[SubResult], str, ExecuteSqlResponse]:
    user_id = settings.default_user_id
    user_db_id = settings.default_user_db_id
    db_type = settings.default_db_type
    schema_name = settings.default_schema_name
    model = settings.default_model
    max_nodes = settings.default_max_nodes

    sub_results: list[SubResult] = []
    primary_sql: str | None = None
    primary_exe = None

    for idx, sub_q in enumerate(analyst_plan.sub_questions):
        await emit_progress(
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
            await emit_progress(
                on_progress,
                {
                    "type": "tts_started",
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
            )
            await emit_text_chunks(
                on_progress=on_progress,
                base_event={
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
                text=f"Text-to-SQL {idx + 1}/{len(analyst_plan.sub_questions)}: {sub_q}",
                event_type="tts_label_chunk",
            )
            await emit_text_chunks(
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
                messages=sub_question_tts_messages(sub_question=sub_q),
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
        await emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
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
                "total": len(analyst_plan.sub_questions),
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
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                    "reason": "duplicate_sql_detected",
                },
            )
            try:
                retry_tts = await generate_sql(
                    settings=settings,
                    question=sub_q,
                    messages=sub_question_tts_messages(
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
                pass
        try:
            await emit_progress(
                on_progress,
                {
                    "type": "execute_started",
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
            )
            await emit_text_chunks(
                on_progress=on_progress,
                base_event={
                    "index": idx + 1,
                    "total": len(analyst_plan.sub_questions),
                    "sub_question": sub_q,
                },
                text=f"Executing {idx + 1}/{len(analyst_plan.sub_questions)}: {sub_q}",
                event_type="execute_label_chunk",
            )
            await emit_text_chunks(
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
        table_md = to_markdown_table((exe.data or [])[:20])
        await emit_text_chunks(
            on_progress=on_progress,
            base_event={
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
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
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
            },
        )
        await emit_progress(
            on_progress,
            {
                "type": "sub_question_done",
                "index": idx + 1,
                "total": len(analyst_plan.sub_questions),
                "sub_question": sub_q,
            },
        )

    if primary_sql is None or primary_exe is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No successful sub-question results were produced",
        )
    return sub_results, primary_sql, primary_exe


async def phase_agents_finalize(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    question: str,
    host_plan: PlanningHostOutput,
    analyst_plan: PlanningAnalystOutput,
    primary_sql: str,
    primary_exe: ExecuteSqlResponse,
    sub_results: list[SubResult],
    on_progress: ProgressCallback | None,
) -> CompletionStreamOutcome:
    deterministic = build_answer_summary(primary_exe)
    agents_out = await run_llm_agents(
        settings=settings,
        question=question,
        generated_sql=primary_sql,
        exe=primary_exe,
        deterministic_summary=deterministic,
        sub_results=sub_results,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
        on_progress=on_progress,
    )
    if isinstance(agents_out, LlmAgentsPaused):
        snap = PulsecastPausedSnapshot(
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
            openai_user=req.user,
        )
        token = resume_store.issue_token(snap)
        await emit_progress(
            on_progress,
            {
                "type": "sql_approval_required",
                "resume_token": token,
                "proposed_sub_question": agents_out.proposed_sub_question,
                "rationale": agents_out.rationale,
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
