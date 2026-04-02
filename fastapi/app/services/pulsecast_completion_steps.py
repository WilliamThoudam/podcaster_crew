from __future__ import annotations

from typing import Any

import httpx
from fastapi import HTTPException, status

from app.clients.execute_sql import execute_sql as execute_sql_client
from app.clients.merge_bi_query import merge_conversational_bi_query
from app.clients.text_to_sql import generate_sql
from app.config import Settings
from app.errors import UpstreamServiceError
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
from app.services.pulsecast_llm_agents import (
    LlmAgentsPaused,
    LlmAgentsPausedWebSearch,
    LlmAgentsPausedDiscussion,
    run_llm_agents,
)
from app.services.pulsecast_planning import (
    run_analyst_planner,
    run_duplicate_sub_question_rephrase,
    run_host_planner,
)
from app.services.pulsecast_resume_store import (
    DuplicateSubQuestionPausedSnapshot,
    PulsecastPausedSnapshot,
    WebSearchPausedSnapshot,
    DiscussionPausedSnapshot,
    resume_store,
)
from app.services.pulsecast_sse_emit import (
    STREAM_CHUNK_SIZE,
    STREAM_DELAY_S,
    emit_progress,
    emit_text_chunks,
)
from app.services.qa_pipeline import build_answer_summary, validate_and_normalize_sql


def collect_user_queries(req: OpenAIChatCompletionRequest) -> list[str]:
    """Ordered non-empty user turns (chronological), for follow-up merge before planning."""
    out: list[str] = []
    for msg in req.messages:
        if msg.role == "user":
            text = msg.content.strip()
            if text:
                out.append(text)
    if not out:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="messages must include at least one non-empty user message",
        )
    return out


def sub_question_tts_messages(*, sub_question: str) -> list[dict[str, str]]:
    """Upstream text-to-SQL accepts user messages only (no system role)."""
    return [{"role": "user", "content": sub_question.strip()}]


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
    queries = collect_user_queries(req)
    try:
        question = await merge_conversational_bi_query(settings=settings, queries=queries)
    except UpstreamServiceError as e:
        is_4xx = e.upstream_status_code is not None and 400 <= e.upstream_status_code < 500
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if is_4xx else status.HTTP_502_BAD_GATEWAY,
            detail={
                "service": e.service,
                "message": e.message,
                "upstream_status_code": e.upstream_status_code,
                "upstream_body": e.upstream_body,
            },
        ) from e
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
    sql_n = len(analyst_plan.sub_questions)
    web_n = len(analyst_plan.web_sub_questions)
    total_sub = sql_n + web_n
    await emit_progress(on_progress, {"type": "planned_sub_questions_started", "total": total_sub})
    plan_lines = [f"{i + 1}. {sq}" for i, sq in enumerate(analyst_plan.sub_questions)]
    base_idx = len(plan_lines)
    for j, wq in enumerate(analyst_plan.web_sub_questions):
        plan_lines.append(f"{base_idx + j + 1}. [Web] {wq}")
    plan_md = "To answer this, I will break it down into steps:\n\n" + "\n".join(plan_lines)
    await emit_text_chunks(
        on_progress=on_progress,
        base_event={"total": total_sub},
        text=plan_md,
        event_type="planned_sub_questions_chunk",
    )
    await emit_progress(on_progress, {"type": "planned_sub_questions_done", "total": total_sub})
    return question, host_plan, analyst_plan


async def run_sub_questions_slice(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    question: str,
    host_plan: PlanningHostOutput,
    analyst_plan: PlanningAnalystOutput,
    on_progress: ProgressCallback | None,
    start_index: int,
    sub_results: list[SubResult],
    primary_sql: str | None,
    primary_exe: ExecuteSqlResponse | None,
    first_sub_question_override: str | None,
    duplicate_fail_index: int | None,
) -> tuple[list[SubResult], str, ExecuteSqlResponse] | CompletionStreamPaused:
    """Run sub-questions from start_index. On duplicate SQL: HITL pause or 422 if idx == duplicate_fail_index."""
    user_id = settings.default_user_id
    user_db_id = settings.default_user_db_id
    db_type = settings.default_db_type
    schema_name = settings.default_schema_name
    model = settings.default_model
    max_nodes = settings.default_max_nodes
    total = len(analyst_plan.sub_questions)
    pending_override = first_sub_question_override

    for idx in range(start_index, total):
        sub_q = pending_override if pending_override is not None else analyst_plan.sub_questions[idx]
        if pending_override is not None:
            pending_override = None

        await emit_progress(
            on_progress,
            {
                "type": "sub_question_start",
                "index": idx + 1,
                "total": total,
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
                session_id=req.user,
                model=model,
                max_nodes=max_nodes,
                is_retry=False,
                trace_context={
                    "sub_question_index": idx + 1,
                    "sub_question_total": total,
                    "sub_question_text": sub_q,
                    "planner_mode": "analyst_decomposition",
                },
            )
        except UpstreamServiceError as e:
            is_4xx = e.upstream_status_code is not None and 400 <= e.upstream_status_code < 500
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if is_4xx else status.HTTP_502_BAD_GATEWAY,
                detail={
                    "service": e.service,
                    "message": e.message,
                    "sub_question_index": idx + 1,
                    "sub_question": sub_q,
                    "upstream_status_code": e.upstream_status_code,
                    "upstream_body": e.upstream_body,
                },
            ) from e
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
                    "message": (
                        f"Step {idx + 1} of {total} would repeat an earlier query; "
                        "refining the wording."
                    ),
                },
            )
            if duplicate_fail_index is not None and idx == duplicate_fail_index:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "duplicate_sql_after_hitl": True,
                        "sub_question": sub_q,
                        "generated_sql": sql,
                        "message": "Text-to-SQL still produced SQL identical to a prior sub-question; edit the question and try again.",
                    },
                )
            orig = analyst_plan.sub_questions[idx]
            proposed, rationale = await run_duplicate_sub_question_rephrase(
                settings=settings,
                question=question,
                host_plan=host_plan,
                analyst_plan=analyst_plan,
                sub_results=sub_results,
                pending_index=idx,
                original_sub_question=orig,
                duplicate_sql=sql,
            )
            if primary_sql is None or primary_exe is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="duplicate_sql_detected but no primary_sql (unexpected)",
                )
            snap = DuplicateSubQuestionPausedSnapshot(
                question=question,
                host_plan=host_plan,
                analyst_plan=analyst_plan,
                sub_results=list(sub_results),
                pending_index=idx,
                original_sub_question=orig,
                duplicate_sql=sql,
                proposed_sub_question=proposed,
                rationale=rationale,
                primary_sql=primary_sql,
                primary_exe=primary_exe,
                openai_user=req.user,
            )
            token = resume_store.issue_token(snap)
            await emit_progress(
                on_progress,
                {
                    "type": "sql_approval_required",
                    "resume_token": token,
                    "proposed_sub_question": proposed,
                    "rationale": rationale,
                    "pause_kind": "duplicate_sub_question",
                },
            )
            return CompletionStreamPaused(
                resume_token=token,
                proposed_sub_question=proposed,
                rationale=rationale,
            )

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
        except UpstreamServiceError as e:
            is_4xx = e.upstream_status_code is not None and 400 <= e.upstream_status_code < 500
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY if is_4xx else status.HTTP_502_BAD_GATEWAY,
                detail={
                    "service": e.service,
                    "message": e.message,
                    "sub_question_index": idx + 1,
                    "sub_question": sub_q,
                    "generated_sql": sql,
                    "upstream_status_code": e.upstream_status_code,
                    "upstream_body": e.upstream_body,
                },
            ) from e
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

    if primary_sql is None or primary_exe is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No successful sub-question results were produced",
        )
    return sub_results, primary_sql, primary_exe


async def phase_sub_questions(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    question: str,
    host_plan: PlanningHostOutput,
    analyst_plan: PlanningAnalystOutput,
    on_progress: ProgressCallback | None,
) -> tuple[list[SubResult], str, ExecuteSqlResponse] | CompletionStreamPaused:
    return await run_sub_questions_slice(
        settings=settings,
        req=req,
        question=question,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
        on_progress=on_progress,
        start_index=0,
        sub_results=[],
        primary_sql=None,
        primary_exe=None,
        first_sub_question_override=None,
        duplicate_fail_index=None,
    )


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
    completed_web_results: list[dict[str, Any]] | None = None,
    user_declined_web_search: bool = False,
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
        completed_web_results=completed_web_results,
        user_declined_web_search=user_declined_web_search,
        on_progress=on_progress,
    )
    if isinstance(agents_out, LlmAgentsPausedWebSearch):
        snap = WebSearchPausedSnapshot(
            stage="mid",
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
            openai_user=req.user,
        )
        token = resume_store.issue_token(snap)
        n = len(agents_out.search_queries)
        step = agents_out.pending_search_index + 1
        await emit_progress(
            on_progress,
            {
                "type": "web_search_approval_required",
                "resume_token": token,
                "proposed_search_query": agents_out.proposed_search_query,
                "rationale": agents_out.rationale,
                "pause_kind": "web_search",
                "web_search_step_index": step,
                "web_search_total_steps": max(1, n),
            },
        )
        return CompletionStreamPaused(
            resume_token=token,
            proposed_sub_question=agents_out.proposed_search_query,
            rationale=agents_out.rationale,
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
                "pause_kind": "challenger_followup",
            },
        )
        return CompletionStreamPaused(
            resume_token=token,
            proposed_sub_question=agents_out.proposed_sub_question,
            rationale=agents_out.rationale,
        )
    if isinstance(agents_out, LlmAgentsPausedDiscussion):
        snap = DiscussionPausedSnapshot(
            stage=agents_out.stage,
            requested_depth=agents_out.requested_depth,
            pipeline=agents_out.pipeline,
            discussion=agents_out.discussion,
            question=agents_out.question,
            generated_sql=agents_out.generated_sql,
            primary_exe=agents_out.primary_exe,
            deterministic_summary=agents_out.deterministic_summary,
            sub_results=agents_out.sub_results,
            host_plan=agents_out.host_plan,
            analyst_plan=agents_out.analyst_plan,
            rationale=agents_out.rationale,
            max_rounds=int(agents_out.max_rounds),
            next_round_index=int(agents_out.next_round_index),
            focus_for_next_round=agents_out.focus_for_next_round,
            openai_user=req.user,
        )
        token = resume_store.issue_token(snap)
        if agents_out.stage == "pre":
            prompt = "Start 1-round panel" if agents_out.requested_depth == "linear" else "Start multi-round panel"
        else:
            prompt = agents_out.focus_for_next_round or f"Continue to round {agents_out.next_round_index}"
        await emit_progress(
            on_progress,
            {
                "type": "discussion_approval_required",
                "resume_token": token,
                "pause_kind": "discussion",
                "stage": agents_out.stage,
                "requested_depth": agents_out.requested_depth,
                "round_index": agents_out.next_round_index,
                "max_rounds": agents_out.max_rounds,
                "focus_for_next_round": agents_out.focus_for_next_round,
                "rationale": agents_out.rationale,
            },
        )
        return CompletionStreamPaused(
            resume_token=token,
            proposed_sub_question=prompt,
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


async def phase_web_search_hitl(
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
) -> CompletionStreamOutcome | tuple[list[dict[str, Any]], bool]:
    """
    Option A: pre-discussion web search HITL.

    Returns:
    - CompletionStreamPaused when user approval is required (resume_token emitted)
    - (completed_web_results, user_declined_web_search) when no pause is needed
      (either no web_sub_questions or Serper not configured).
    """
    web_qs = [str(x).strip() for x in (analyst_plan.web_sub_questions or []) if str(x).strip()]
    if not web_qs:
        return ([], False)
    # If Serper isn't configured, skip pre-search; WEB_CRAWLER may still decide later.
    if not (settings.serper_api_key or "").strip():
        return ([], False)

    deterministic = build_answer_summary(primary_exe)
    snap = WebSearchPausedSnapshot(
        stage="pre",
        pipeline=None,
        discussion=None,
        question=question,
        generated_sql=primary_sql,
        primary_exe=primary_exe,
        deterministic_summary=deterministic,
        sub_results=list(sub_results),
        host_plan=host_plan,
        analyst_plan=analyst_plan,
        proposed_search_query=web_qs[0],
        search_queries=web_qs,
        pending_search_index=0,
        completed_web_results=[],
        rationale="Planned web evidence steps from analyst_plan.web_sub_questions.",
        openai_user=req.user,
    )
    token = resume_store.issue_token(snap)
    await emit_progress(
        on_progress,
        {
            "type": "web_search_approval_required",
            "resume_token": token,
            "proposed_search_query": web_qs[0],
            "rationale": snap.rationale,
            "pause_kind": "web_search",
            "web_search_step_index": 1,
            "web_search_total_steps": max(1, len(web_qs)),
        },
    )
    return CompletionStreamPaused(
        resume_token=token,
        proposed_sub_question=web_qs[0],
        rationale=snap.rationale,
    )
