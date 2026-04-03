from __future__ import annotations

import asyncio
import contextlib
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
    run_llm_agents_minimal_only,
    _emit_web_search_results_sse,
    _serper_results_markdown,
    _run_llm_agents_moderated_discussion,
)
from app.services.pulsecast_resume_store import (
    DuplicateSubQuestionPausedSnapshot,
    DiscussionPausedSnapshot,
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
from app.services.stream_pause_store import stream_pause_store
from app.services.qa_pipeline import build_answer_summary, validate_and_normalize_sql
from app.clients.serper_search import serper_google_search

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

def _extract_upstream_error_text(detail: Any) -> str | None:
    """
    Convert structured HTTPException.detail (dict) into a plain, user-facing error line.
    Used to avoid streaming the internal <<PULSECAST_HTTP_ERROR:{...}>> marker to the UI.
    """
    if not isinstance(detail, dict):
        return None

    # Our UpstreamServiceError mapping emits this shape from pulsecast_completion_steps.py
    upstream_body = detail.get("upstream_body")
    if isinstance(upstream_body, str) and upstream_body.strip():
        try:
            body_obj = json.loads(upstream_body)
            if isinstance(body_obj, dict):
                err = body_obj.get("error")
                if isinstance(err, str) and err.strip():
                    return err.strip()
        except Exception:
            # Not JSON or not parseable; fall through to return raw text.
            return upstream_body.strip()

    # Other known shapes
    tts_err = detail.get("text_to_sql_error")
    if isinstance(tts_err, str) and tts_err.strip():
        return tts_err.strip()

    msg = detail.get("message")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()

    return None


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


async def _gated_yields(job_id: str, line: str) -> AsyncIterator[str]:
    for part in await stream_pause_store.emit_line(job_id, line):
        yield part


async def _drain_gated_yields(job_id: str) -> AsyncIterator[str]:
    for part in await stream_pause_store.drain_outbound(job_id):
        yield part


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


async def _pause_from_discussion_agents_out(
    *,
    settings: Settings,
    agents_out: Any,
    openai_user: str | None,
    on_progress: ProgressCallback | None,
) -> CompletionStreamOutcome:
    from app.services.pulsecast_llm_agents import LlmAgentsComplete, LlmAgentsPausedDiscussion

    if isinstance(agents_out, LlmAgentsComplete):
        return CompletionStreamComplete(content=agents_out.answer)

    if isinstance(agents_out, LlmAgentsPausedWebSearch):
        snap_ws = WebSearchPausedSnapshot(
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
            openai_user=openai_user,
        )
        token_ws = resume_store.issue_token(snap_ws)
        await emit_progress(
            on_progress,
            {
                "type": "web_search_approval_required",
                "resume_token": token_ws,
                "proposed_search_query": agents_out.proposed_search_query,
                "rationale": agents_out.rationale,
                "pause_kind": "web_search",
                "web_search_step_index": 1,
                "web_search_total_steps": 1,
            },
        )
        return CompletionStreamPaused(
            resume_token=token_ws,
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
            openai_user=openai_user,
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
        snap_d = DiscussionPausedSnapshot(
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
            openai_user=openai_user,
        )
        token_d = resume_store.issue_token(snap_d)
        await emit_progress(
            on_progress,
            {
                "type": "discussion_approval_required",
                "resume_token": token_d,
                "pause_kind": "discussion",
                "stage": agents_out.stage,
                "requested_depth": agents_out.requested_depth,
                "round_index": agents_out.next_round_index,
                "max_rounds": agents_out.max_rounds,
                "focus_for_next_round": agents_out.focus_for_next_round,
                "rationale": agents_out.rationale,
            },
        )
        prompt = (
            ("Start 1-round panel" if agents_out.requested_depth == "linear" else "Start multi-round panel")
            if agents_out.stage == "pre"
            else (agents_out.focus_for_next_round or f"Continue to round {agents_out.next_round_index}")
        )
        return CompletionStreamPaused(resume_token=token_d, proposed_sub_question=prompt, rationale=agents_out.rationale)

    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unknown discussion outcome")


async def build_resume_discussion_payload(
    *,
    settings: Settings,
    snapshot: DiscussionPausedSnapshot,
    approved: bool,
    on_progress: ProgressCallback | None = None,
) -> CompletionStreamComplete | CompletionStreamPaused:
    # Decline starting moderated discussion -> minimal (ANALYST→HOST)
    if snapshot.stage == "pre" and not approved:
        agents_done = await run_llm_agents_minimal_only(
            settings=settings,
            question=snapshot.question,
            generated_sql=snapshot.generated_sql,
            exe=snapshot.primary_exe,
            deterministic_summary=snapshot.deterministic_summary,
            sub_results=list(snapshot.sub_results),
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            user_declined_web_search=False,
            on_progress=on_progress,
        )
        return CompletionStreamComplete(content=agents_done.answer)

    # Approve starting discussion (linear or moderated)
    if snapshot.stage == "pre" and approved:
        from app.services.pulsecast_llm_agents import _context_blob_compact

        ctx = _context_blob_compact(
            question=snapshot.question,
            generated_sql=snapshot.generated_sql,
            exe=snapshot.primary_exe,
            deterministic_summary=snapshot.deterministic_summary,
            sub_results=list(snapshot.sub_results),
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            settings=settings,
        )
        if snapshot.requested_depth == "linear":
            from app.services.pulsecast_llm_agents import _run_llm_agents_linear

            agents_out = await _run_llm_agents_linear(
                settings=settings,
                question=snapshot.question,
                generated_sql=snapshot.generated_sql,
                exe=snapshot.primary_exe,
                deterministic_summary=snapshot.deterministic_summary,
                sr_list=list(snapshot.sub_results),
                host_plan=snapshot.host_plan,
                analyst_plan=snapshot.analyst_plan,
                ctx=ctx,
                allow_sql_approval_pause=True,
                on_progress=on_progress,
            )
        else:
            max_rounds = max(1, settings.pulsecast_discussion_max_rounds)
            agents_out = await _run_llm_agents_moderated_discussion(
                settings=settings,
                question=snapshot.question,
                generated_sql=snapshot.generated_sql,
                exe=snapshot.primary_exe,
                deterministic_summary=snapshot.deterministic_summary,
                sr_list=list(snapshot.sub_results),
                host_plan=snapshot.host_plan,
                analyst_plan=snapshot.analyst_plan,
                ctx=ctx,
                allow_sql_approval_pause=True,
                max_rounds=max_rounds,
                on_progress=on_progress,
            )
        return await _pause_from_discussion_agents_out(
            settings=settings,
            agents_out=agents_out,
            openai_user=snapshot.openai_user,
            on_progress=on_progress,
        )

    # Between rounds: approved -> continue moderated; declined -> HOST finalize from transcript so far
    if snapshot.stage == "mid" and not approved:
        agents_done = await run_llm_agents_host_only(
            settings=settings,
            question=snapshot.question,
            generated_sql=snapshot.generated_sql,
            exe=snapshot.primary_exe,
            deterministic_summary=snapshot.deterministic_summary,
            sub_results=list(snapshot.sub_results),
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            discussion=snapshot.discussion,  # type: ignore[arg-type]
            pipeline=snapshot.pipeline or [],
            user_declined_extra_sql=False,
        )
        return CompletionStreamComplete(content=agents_done.answer)

    if snapshot.stage == "mid" and approved:
        from app.services.pulsecast_llm_agents import _context_blob_compact

        ctx = _context_blob_compact(
            question=snapshot.question,
            generated_sql=snapshot.generated_sql,
            exe=snapshot.primary_exe,
            deterministic_summary=snapshot.deterministic_summary,
            sub_results=list(snapshot.sub_results),
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            settings=settings,
        )
        agents_out = await _run_llm_agents_moderated_discussion(
            settings=settings,
            question=snapshot.question,
            generated_sql=snapshot.generated_sql,
            exe=snapshot.primary_exe,
            deterministic_summary=snapshot.deterministic_summary,
            sr_list=list(snapshot.sub_results),
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            ctx=ctx,
            allow_sql_approval_pause=True,
            max_rounds=int(snapshot.max_rounds or settings.pulsecast_discussion_max_rounds),
            on_progress=on_progress,
            initial_discussion=snapshot.discussion,  # type: ignore[arg-type]
            initial_pipeline=snapshot.pipeline,
            start_round=int(snapshot.next_round_index or 1),
            focus_for_next_round=(snapshot.focus_for_next_round or "").strip() or None,
        )
        return await _pause_from_discussion_agents_out(
            settings=settings,
            agents_out=agents_out,
            openai_user=snapshot.openai_user,
            on_progress=on_progress,
        )

    # Should not happen
    agents_done = await run_llm_agents_host_only(
        settings=settings,
        question=snapshot.question,
        generated_sql=snapshot.generated_sql,
        exe=snapshot.primary_exe,
        deterministic_summary=snapshot.deterministic_summary,
        sub_results=list(snapshot.sub_results),
        host_plan=snapshot.host_plan,
        analyst_plan=snapshot.analyst_plan,
        discussion=snapshot.discussion,  # type: ignore[arg-type]
        pipeline=snapshot.pipeline or [],
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

    # Option A: pre-discussion web-search phase (no agent transcript exists yet).
    if getattr(snapshot, "stage", "mid") == "pre":
        openai_req = OpenAIChatCompletionRequest(
            model="pulsecast-qa",
            stream=True,
            messages=[OpenAIChatMessage(role="user", content=snapshot.question)],
            user=snapshot.openai_user,
        )

        queries = [str(x).strip() for x in (snapshot.search_queries or []) if str(x).strip()]
        if not queries and (snapshot.proposed_search_query or "").strip():
            queries = [(snapshot.proposed_search_query or "").strip()]
        idx = int(snapshot.pending_search_index or 0)
        if idx < 0:
            idx = 0
        if queries and idx >= len(queries):
            idx = len(queries) - 1

        completed = [dict(x) for x in (snapshot.completed_web_results or [])]

        if not approved:
            md = _serper_results_markdown(query="", results=None, error=None, declined=True)
            await _emit_web_search_results_sse(
                on_progress,
                settings=settings,
                query=None,
                markdown_body=md,
                context_question=snapshot.question,
                organic_for_takeaways=None,
            )
            return await phase_agents_finalize(
                settings=settings,
                req=openai_req,
                question=snapshot.question,
                host_plan=snapshot.host_plan,
                analyst_plan=snapshot.analyst_plan,
                primary_sql=snapshot.generated_sql,
                primary_exe=snapshot.primary_exe,
                sub_results=list(snapshot.sub_results),
                completed_web_results=completed,
                user_declined_web_search=True,
                on_progress=on_progress,
            )

        q = (edited_question or "").strip() or (queries[idx] if queries else "").strip()
        step_results: list[dict[str, Any]] | None = None
        step_err: str | None = None
        if q:
            try:
                step_results = await serper_google_search(
                    settings=settings,
                    query=q,
                    num=settings.serper_num_results,
                )
                completed.append({"query": q, "results": step_results, "error": None})
            except Exception as e:
                step_err = str(e)
                completed.append({"query": q, "results": None, "error": step_err})
        else:
            completed.append({"query": "", "results": None, "error": "empty query after approval"})

        if step_err:
            md = _serper_results_markdown(query=q, results=None, error=str(step_err), declined=False)
        elif not q:
            md = "_No search query was available after approval._"
        else:
            md = _serper_results_markdown(query=q, results=step_results, error=None, declined=False)
        takeaway_rows: list[dict[str, Any]] | None = None
        if approved and not step_err and q and step_results:
            takeaway_rows = list(step_results)
        await _emit_web_search_results_sse(
            on_progress,
            settings=settings,
            query=q or None,
            markdown_body=md,
            context_question=snapshot.question,
            organic_for_takeaways=takeaway_rows,
        )

        more = bool(queries) and (idx + 1) < len(queries)
        if more:
            snap_ws = WebSearchPausedSnapshot(
                stage="pre",
                pipeline=None,
                discussion=None,
                question=snapshot.question,
                generated_sql=snapshot.generated_sql,
                primary_exe=snapshot.primary_exe,
                deterministic_summary=snapshot.deterministic_summary,
                sub_results=list(snapshot.sub_results),
                host_plan=snapshot.host_plan,
                analyst_plan=snapshot.analyst_plan,
                proposed_search_query=queries[idx + 1],
                search_queries=queries,
                pending_search_index=idx + 1,
                completed_web_results=completed,
                rationale=snapshot.rationale,
                openai_user=snapshot.openai_user,
            )
            token_ws = resume_store.issue_token(snap_ws)
            await emit_progress(
                on_progress,
                {
                    "type": "web_search_approval_required",
                    "resume_token": token_ws,
                    "proposed_search_query": snap_ws.proposed_search_query,
                    "rationale": snap_ws.rationale,
                    "pause_kind": "web_search",
                    "web_search_step_index": 1,
                    "web_search_total_steps": 1,
                },
            )
            return CompletionStreamPaused(
                resume_token=token_ws,
                proposed_sub_question=snap_ws.proposed_search_query,
                rationale=snap_ws.rationale,
            )

        return await phase_agents_finalize(
            settings=settings,
            req=openai_req,
            question=snapshot.question,
            host_plan=snapshot.host_plan,
            analyst_plan=snapshot.analyst_plan,
            primary_sql=snapshot.generated_sql,
            primary_exe=snapshot.primary_exe,
            sub_results=list(snapshot.sub_results),
            completed_web_results=completed,
            user_declined_web_search=False,
            on_progress=on_progress,
        )

    # Existing path: mid-discussion web-search (WEB_CRAWLER requested it).
    agents_out = await run_llm_agents_after_web_hitl(
        settings=settings,
        question=snapshot.question,
        generated_sql=snapshot.generated_sql,
        exe=snapshot.primary_exe,
        deterministic_summary=snapshot.deterministic_summary,
        sub_results=list(snapshot.sub_results),
        host_plan=snapshot.host_plan,
        analyst_plan=snapshot.analyst_plan,
        discussion=snapshot.discussion,  # type: ignore[arg-type]
        pipeline=snapshot.pipeline or [],
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
            openai_user=snapshot.openai_user,
        )
        token_ws = resume_store.issue_token(snap_ws)
        await emit_progress(
            on_progress,
            {
                "type": "web_search_approval_required",
                "resume_token": token_ws,
                "proposed_search_query": agents_out.proposed_search_query,
                "rationale": agents_out.rationale,
                "pause_kind": "web_search",
                "web_search_step_index": 1,
                "web_search_total_steps": 1,
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
    if isinstance(snapshot, DiscussionPausedSnapshot):
        return await build_resume_discussion_payload(
            settings=settings,
            snapshot=snapshot,
            approved=req.approved,
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
    job_id: str,
) -> AsyncIterator[str]:
    await stream_pause_store.register(job_id, req.user)
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    progress_q: asyncio.Queue[str] = asyncio.Queue()
    started = False
    task: asyncio.Task[CompletionStreamOutcome] | None = None

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

    try:
        while True:
            if task.done() and progress_q.empty():
                break
            try:
                marker = await asyncio.wait_for(progress_q.get(), timeout=0.1)
            except TimeoutError:
                continue
            if not started:
                started = True
                async for line in _gated_yields(
                    job_id,
                    f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
                ):
                    yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=marker)}\n\n",
            ):
                yield line

        try:
            outcome = await task
        except HTTPException as e:
            if not started:
                started = True
                async for line in _gated_yields(
                    job_id,
                    f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
                ):
                    yield line

            plain = _extract_upstream_error_text(e.detail) or (
                str(e.detail) if e.detail is not None else str(e)
            )
            async for line in _drain_gated_yields(job_id):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=plain)}\n\n",
            ):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
            ):
                yield line
            async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
                yield line
            return
        except Exception as e:
            if not started:
                raise
            error_text = f"Failed to generate response: {e}"
            async for line in _drain_gated_yields(job_id):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=error_text)}\n\n",
            ):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
            ):
                yield line
            async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
                yield line
            return

        if isinstance(outcome, CompletionStreamPaused):
            if not started:
                started = True
                async for line in _gated_yields(
                    job_id,
                    f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
                ):
                    yield line
            async for line in _drain_gated_yields(job_id):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
            ):
                yield line
            async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
                yield line
            return

        if not started:
            started = True
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
            ):
                yield line

        content = outcome.content
        for i in range(0, len(content), STREAM_CHUNK_SIZE):
            part = content[i : i + STREAM_CHUNK_SIZE]
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=part)}\n\n",
            ):
                yield line
            await asyncio.sleep(STREAM_DELAY_S)

        async for line in _drain_gated_yields(job_id):
            yield line
        async for line in _gated_yields(
            job_id,
            f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
        ):
            yield line
        async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
            yield line
    except asyncio.CancelledError:
        raise
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await stream_pause_store.unregister(job_id)


async def stream_completion_sse(
    *,
    settings: Settings,
    req: OpenAIChatCompletionRequest,
    job_id: str,
) -> AsyncIterator[str]:
    await stream_pause_store.register(job_id, req.user)
    completion_id = f"chatcmpl_{uuid.uuid4().hex}"
    progress_q: asyncio.Queue[str] = asyncio.Queue()
    started = False
    task: asyncio.Task[CompletionStreamOutcome] | None = None

    async def on_progress(event: dict[str, Any]) -> None:
        marker = f"<<PULSECAST_PROGRESS:{json.dumps(event, ensure_ascii=False)}>>"
        await progress_q.put(marker)

    task = asyncio.create_task(
        build_completion_payload(settings=settings, req=req, on_progress=on_progress)
    )

    try:
        while True:
            if task.done() and progress_q.empty():
                break
            try:
                marker = await asyncio.wait_for(progress_q.get(), timeout=0.1)
            except TimeoutError:
                continue
            if not started:
                started = True
                async for line in _gated_yields(
                    job_id,
                    f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
                ):
                    yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=marker)}\n\n",
            ):
                yield line

        try:
            outcome = await task
        except HTTPException as e:
            if not started:
                started = True
                async for line in _gated_yields(
                    job_id,
                    f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
                ):
                    yield line

            plain = _extract_upstream_error_text(e.detail) or (
                str(e.detail) if e.detail is not None else str(e)
            )
            async for line in _drain_gated_yields(job_id):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=plain)}\n\n",
            ):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
            ):
                yield line
            async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
                yield line
            return
        except Exception as e:
            if not started:
                raise
            error_text = f"Failed to generate response: {e}"
            async for line in _drain_gated_yields(job_id):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=error_text)}\n\n",
            ):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
            ):
                yield line
            async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
                yield line
            return

        if isinstance(outcome, CompletionStreamPaused):
            if not started:
                started = True
                async for line in _gated_yields(
                    job_id,
                    f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
                ):
                    yield line
            async for line in _drain_gated_yields(job_id):
                yield line
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
            ):
                yield line
            async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
                yield line
            return

        if not started:
            started = True
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), role='assistant')}\n\n",
            ):
                yield line

        content = outcome.content
        for i in range(0, len(content), STREAM_CHUNK_SIZE):
            part = content[i : i + STREAM_CHUNK_SIZE]
            async for line in _gated_yields(
                job_id,
                f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), content=part)}\n\n",
            ):
                yield line
            await asyncio.sleep(STREAM_DELAY_S)

        async for line in _drain_gated_yields(job_id):
            yield line
        async for line in _gated_yields(
            job_id,
            f"data: {_chunk_json(completion_id=completion_id, model=req.model, now=int(time.time()), finish_reason='stop')}\n\n",
        ):
            yield line
        async for line in _gated_yields(job_id, "data: [DONE]\n\n"):
            yield line
    except asyncio.CancelledError:
        raise
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await stream_pause_store.unregister(job_id)
