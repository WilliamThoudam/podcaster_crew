from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Union

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm.chat_model import build_chat_model
from app.prompts.llm_agents import (
    system_prompt_host_composer,
    system_prompt_internal,
    system_prompt_moderator,
)
from app.models.schemas import (
    AgentInsight,
    AgentPipelineStep,
    ExecuteSqlResponse,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)
from app.services.pulsecast_sse_emit import (
    STREAM_CHUNK_SIZE,
    STREAM_DELAY_S,
    emit_progress,
    emit_text_chunks,
)
from app.services.sub_question_tts_guard import is_valid_tts_sub_question

PulsecastRole = Literal["HOST", "ANALYST", "MARKETING", "FINANCE", "CHALLENGER"]
PulsecastAgentId = Literal["host", "analyst", "marketing", "finance", "challenger"]
DiscussantRole = Literal["MARKETING", "FINANCE", "CHALLENGER"]

AgentProgressCallback = Callable[[dict[str, Any]], Union[Awaitable[None], None]] | None


class _AgentOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1)
    phase: str = Field(..., min_length=1)
    detail: str | None = None
    needs_more_data: bool = False
    proposed_sub_question: str | None = None
    why: str | None = None


class DiscussionTurn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: DiscussantRole
    round_index: int = Field(..., ge=1)
    output: _AgentOut


@dataclass
class DiscussionState:
    analyst: _AgentOut
    turns: list[DiscussionTurn]


class _ModeratorOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    continue_discussion: bool
    reason: str = ""
    focus_for_next_round: str | None = None


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _coerce_optional_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _coerce_required_str(v: Any, *, field: str) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    if v is None:
        raise ValueError(f"Agent JSON field {field!r} is missing or null")
    return str(v)


def _normalize_agent_out_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """LLMs sometimes put structured JSON in `detail`; Pydantic expects str | null."""
    out = dict(obj)
    if "text" in out:
        out["text"] = _coerce_required_str(out["text"], field="text")
    if "phase" in out:
        out["phase"] = _coerce_required_str(out["phase"], field="phase")
    for key in ("detail", "proposed_sub_question", "why"):
        if key in out:
            out[key] = _coerce_optional_str(out[key])
    return out


def _parse_agent_json(raw: str) -> _AgentOut:
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if not m:
        raise ValueError("Agent output did not contain a JSON object")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Agent output JSON must be an object")
    obj = _normalize_agent_out_dict(obj)
    return _AgentOut.model_validate(obj)


def _parse_moderator_json(raw: str) -> _ModeratorOut:
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if not m:
        raise ValueError("Moderator output did not contain a JSON object")
    obj = json.loads(m.group(0))
    return _ModeratorOut.model_validate(obj)


def _discussion_markdown(out: _AgentOut) -> str:
    parts = [f"**{out.phase}**\n\n{out.text}"]
    if out.detail:
        parts.append(f"\n\n_{out.detail}_")
    return "".join(parts)


async def _sse_discussion_analyst(
    on_progress: AgentProgressCallback,
    out: _AgentOut,
) -> None:
    if not on_progress:
        return
    await emit_progress(on_progress, {"type": "discussion_analyst_started"})
    await emit_text_chunks(
        on_progress=on_progress,
        base_event={},
        text=_discussion_markdown(out),
        event_type="discussion_analyst_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await emit_progress(on_progress, {"type": "discussion_analyst_done"})


async def _sse_discussion_turn(
    on_progress: AgentProgressCallback,
    role: str,
    round_index: int,
    out: _AgentOut,
) -> None:
    if not on_progress:
        return
    base: dict[str, Any] = {"role": role, "round": round_index}
    await emit_progress(on_progress, {**base, "type": "discussion_turn_started"})
    await emit_text_chunks(
        on_progress=on_progress,
        base_event=dict(base),
        text=_discussion_markdown(out),
        event_type="discussion_turn_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await emit_progress(on_progress, {**base, "type": "discussion_turn_done"})


async def _sse_discussion_moderator(
    on_progress: AgentProgressCallback,
    mod: _ModeratorOut,
) -> None:
    if not on_progress:
        return
    await emit_progress(
        on_progress,
        {
            "type": "discussion_moderator",
            "continue_discussion": mod.continue_discussion,
            "reason": mod.reason,
        },
    )


def _safe_sample(exe: ExecuteSqlResponse, max_rows: int = 10, max_cell_len: int = 120) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in (exe.data or [])[:max_rows]:
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, str) and len(v) > max_cell_len:
                clean[k] = v[: max_cell_len - 1] + "…"
            else:
                clean[k] = v
        out.append(clean)
    return out


def _compact_sub_results(
    sub_results: list[SubResult],
    max_rows_per_result: int = 3,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for sr in sub_results:
        rows = sr.execute.data or []
        compact.append(
            {
                "sub_question": sr.sub_question,
                "generated_sql": sr.generated_sql,
                "rows_returned": len(rows),
                "columns_in_data": list(rows[0].keys()) if rows else [],
                "data_sample": _safe_sample(sr.execute, max_rows=max_rows_per_result),
            }
        )
    return compact


def _context_blob_compact(
    *,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
    sub_results: list[SubResult] | None = None,
    host_plan: PlanningHostOutput | None = None,
    analyst_plan: PlanningAnalystOutput | None = None,
) -> dict[str, Any]:
    primary = exe.data or []
    base: dict[str, Any] = {
        "question": question,
        "generated_sql": generated_sql,
        "rows_returned": len(primary),
        "columns_in_data": list(primary[0].keys()) if primary else [],
        "data_sample": _safe_sample(exe),
        "deterministic_summary": deterministic_summary,
    }
    if sub_results:
        base["sub_results"] = _compact_sub_results(sub_results)
    if host_plan:
        base["host_plan"] = {
            "primary_focus": host_plan.primary_focus,
            "time_window": host_plan.time_window,
            "region_focus": host_plan.region_focus,
            "metrics": host_plan.metrics,
            "notes": host_plan.notes,
        }
    if analyst_plan:
        base["analyst_plan"] = {
            "sub_questions": analyst_plan.sub_questions,
            "rationale": analyst_plan.rationale,
        }
    return base


def _human_prompt(*, ctx: dict[str, Any], prior: dict[str, _AgentOut]) -> str:
    prior_obj = {k: v.model_dump() for k, v in prior.items()}
    return (
        "Context JSON:\n"
        f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
        "Prior agent outputs JSON:\n"
        f"{json.dumps(prior_obj, ensure_ascii=False)}\n"
    )


def _analyst_opening_prompt(*, ctx: dict[str, Any]) -> str:
    return (
        "Context JSON:\n"
        f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
        "You are the first internal agent (ANALYST). No prior agent outputs exist yet.\n"
    )


def _discussion_user_prompt(
    *,
    ctx: dict[str, Any],
    analyst: _AgentOut,
    turns: list[DiscussionTurn],
    role: DiscussantRole,
    round_index: int,
    focus_for_next_round: str | None,
) -> str:
    lines: list[str] = [
        "Context JSON:\n",
        json.dumps(ctx, ensure_ascii=False),
        "\n\nANALYST opening (JSON):\n",
        json.dumps(analyst.model_dump(), ensure_ascii=False),
        "\n\nDiscussion transcript (chronological, internal JSON per turn):\n",
    ]
    for t in turns:
        lines.append(
            f"- Round {t.round_index} {t.role}: {json.dumps(t.output.model_dump(), ensure_ascii=False)}\n"
        )
    if not turns:
        lines.append("(none yet — you speak first after the Analyst.)\n")
    lines.append(f"\nYour turn: {role} in round {round_index}.\n")
    if focus_for_next_round and focus_for_next_round.strip():
        lines.append(f"Moderator focus for this round: {focus_for_next_round.strip()}\n")
    return "".join(lines)


def _moderator_user_prompt(*, ctx: dict[str, Any], analyst: _AgentOut, turns: list[DiscussionTurn]) -> str:
    compact_round: list[dict[str, Any]] = [
        {"round": t.round_index, "role": t.role, "text": t.output.text, "detail": t.output.detail}
        for t in turns
    ]
    return (
        "User question (from context.question):\n"
        f"{json.dumps(ctx.get('question', ''), ensure_ascii=False)}\n\n"
        "ANALYST opening summary:\n"
        f"{json.dumps(analyst.model_dump(), ensure_ascii=False)}\n\n"
        "Latest full round transcript (all turns this round, in order):\n"
        f"{json.dumps(compact_round, ensure_ascii=False)}\n"
    )


def _host_user_prompt(*, ctx: dict[str, Any], discussion: DiscussionState) -> str:
    turn_objs = [
        {"round": t.round_index, "role": t.role, "output": t.output.model_dump()}
        for t in discussion.turns
    ]
    return (
        "Context JSON:\n"
        f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
        "ANALYST opening (JSON):\n"
        f"{json.dumps(discussion.analyst.model_dump(), ensure_ascii=False)}\n\n"
        "Full internal discussion transcript (JSON):\n"
        f"{json.dumps(turn_objs, ensure_ascii=False)}\n"
    )


def _agent_id(role: PulsecastRole) -> PulsecastAgentId:
    return {
        "HOST": "host",
        "ANALYST": "analyst",
        "MARKETING": "marketing",
        "FINANCE": "finance",
        "CHALLENGER": "challenger",
    }[role]


def _chunk_text(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


async def _stream_llm_text(*, settings: Settings, messages: list[dict[str, str]]) -> str:
    model = build_chat_model(settings)
    lc_messages: list[SystemMessage | HumanMessage] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    text_parts: list[str] = []
    async for chunk in model.astream(lc_messages):
        piece = _chunk_text(chunk.content)
        if piece:
            text_parts.append(piece)
    return "".join(text_parts).strip()


async def _run_role_call(
    *,
    settings: Settings,
    role: PulsecastRole,
    system_prompt: str,
    ctx: dict[str, Any],
    prior: dict[str, _AgentOut],
) -> _AgentOut:
    user_content = _human_prompt(ctx=ctx, prior=prior)
    return await _run_agent_json(settings, role, system_prompt, user_content)


async def _run_agent_json(
    settings: Settings,
    role_label: str,
    system_prompt: str,
    user_content: str,
) -> _AgentOut:
    try:
        text = await _stream_llm_text(
            settings=settings,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        if not text:
            raise ValueError(f"Empty streamed content from agent {role_label}")
        return _parse_agent_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM agent {role_label} failed: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM agent {role_label} failed: {e}",
        ) from e


async def _run_moderator_call(*, settings: Settings, user_content: str) -> _ModeratorOut:
    try:
        text = await _stream_llm_text(
            settings=settings,
            messages=[
                {"role": "system", "content": system_prompt_moderator()},
                {"role": "user", "content": user_content},
            ],
        )
        if not text:
            raise ValueError("Empty streamed content from moderator")
        return _parse_moderator_json(text)
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM moderator failed: {e}",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM moderator failed: {e}",
        ) from e


@dataclass
class LlmAgentsComplete:
    answer: str
    pipeline: list[AgentPipelineStep]
    messages: list[AgentInsight]


@dataclass
class LlmAgentsPaused:
    pipeline: list[AgentPipelineStep]
    discussion: DiscussionState
    question: str
    generated_sql: str
    primary_exe: ExecuteSqlResponse
    deterministic_summary: str
    sub_results: list[SubResult]
    host_plan: PlanningHostOutput
    analyst_plan: PlanningAnalystOutput
    proposed_sub_question: str
    rationale: str | None


def discussion_state_from_legacy_prior(prior: dict[str, _AgentOut]) -> DiscussionState | None:
    """Best-effort conversion for snapshot v1 (single pass)."""
    if "ANALYST" not in prior:
        return None
    analyst = prior["ANALYST"]
    turns: list[DiscussionTurn] = []
    for role in ("MARKETING", "FINANCE", "CHALLENGER"):
        if role in prior:
            turns.append(DiscussionTurn(role=role, round_index=1, output=prior[role]))
    return DiscussionState(analyst=analyst, turns=turns)


async def _run_llm_agents_linear(
    *,
    settings: Settings,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
    sr_list: list[SubResult],
    host_plan: PlanningHostOutput | None,
    analyst_plan: PlanningAnalystOutput | None,
    ctx: dict[str, Any],
    allow_sql_approval_pause: bool,
    on_progress: AgentProgressCallback,
) -> Union[LlmAgentsComplete, LlmAgentsPaused]:
    internal_roles: list[PulsecastRole] = ["ANALYST", "MARKETING", "FINANCE", "CHALLENGER"]
    prior: dict[str, _AgentOut] = {}
    pipeline: list[AgentPipelineStep] = []

    for role in internal_roles:
        out = await _run_role_call(
            settings=settings,
            role=role,
            system_prompt=system_prompt_internal(role, discussion_aware=False),
            ctx=ctx,
            prior=prior,
        )
        prior[role] = out
        pipeline.append(
            AgentPipelineStep(
                id=_agent_id(role),
                status="completed",
                phase=out.phase,
                detail=out.detail,
            )
        )
        if role == "ANALYST":
            await _sse_discussion_analyst(on_progress, out)
        elif role in ("MARKETING", "FINANCE", "CHALLENGER"):
            await _sse_discussion_turn(on_progress, role, 1, out)
        if role == "CHALLENGER" and allow_sql_approval_pause:
            pq = (out.proposed_sub_question or "").strip()
            if (
                out.needs_more_data
                and pq
                and is_valid_tts_sub_question(pq)
                and host_plan is not None
                and analyst_plan is not None
            ):
                ds = discussion_state_from_legacy_prior(prior)
                if ds is None:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to build discussion state for pause",
                    )
                return LlmAgentsPaused(
                    pipeline=pipeline,
                    discussion=ds,
                    question=question,
                    generated_sql=generated_sql,
                    primary_exe=exe,
                    deterministic_summary=deterministic_summary,
                    sub_results=sr_list,
                    host_plan=host_plan,
                    analyst_plan=analyst_plan,
                    proposed_sub_question=pq,
                    rationale=out.why,
                )

    discussion = discussion_state_from_legacy_prior(prior)
    if discussion is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build discussion state after linear agent run",
        )
    host_out = await _run_agent_json(
        settings,
        "HOST",
        system_prompt_host_composer(),
        _host_user_prompt(ctx=ctx, discussion=discussion),
    )
    pipeline.append(
        AgentPipelineStep(
            id=_agent_id("HOST"),
            status="completed",
            phase=host_out.phase,
            detail=host_out.detail,
        )
    )
    return LlmAgentsComplete(answer=host_out.text, pipeline=pipeline, messages=[AgentInsight(role="HOST", text=host_out.text)])


async def _run_llm_agents_moderated_discussion(
    *,
    settings: Settings,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
    sr_list: list[SubResult],
    host_plan: PlanningHostOutput | None,
    analyst_plan: PlanningAnalystOutput | None,
    ctx: dict[str, Any],
    allow_sql_approval_pause: bool,
    max_rounds: int,
    on_progress: AgentProgressCallback,
) -> Union[LlmAgentsComplete, LlmAgentsPaused]:
    pipeline: list[AgentPipelineStep] = []
    analyst_out = await _run_agent_json(
        settings,
        "ANALYST",
        system_prompt_internal("ANALYST", discussion_aware=False),
        _analyst_opening_prompt(ctx=ctx),
    )
    pipeline.append(
        AgentPipelineStep(
            id=_agent_id("ANALYST"),
            status="completed",
            phase=analyst_out.phase,
            detail=analyst_out.detail,
        )
    )
    await _sse_discussion_analyst(on_progress, analyst_out)
    turns: list[DiscussionTurn] = []
    focus: str | None = None

    for r in range(1, max_rounds + 1):
        if on_progress:
            await emit_progress(on_progress, {"type": "discussion_round_started", "round": r})
        round_slice_start = len(turns)
        for role in ("MARKETING", "FINANCE", "CHALLENGER"):
            discussant: DiscussantRole = role
            user_content = _discussion_user_prompt(
                ctx=ctx,
                analyst=analyst_out,
                turns=turns,
                role=discussant,
                round_index=r,
                focus_for_next_round=focus if r > 1 else None,
            )
            pr: PulsecastRole = discussant  # MARKETING|FINANCE|CHALLENGER ⊆ PulsecastRole
            out = await _run_agent_json(
                settings,
                role,
                system_prompt_internal(pr, discussion_aware=True),
                user_content,
            )
            turns.append(DiscussionTurn(role=discussant, round_index=r, output=out))
            snippet = (out.detail or out.text or out.phase or "")[:500]
            pipeline.append(
                AgentPipelineStep(
                    id=_agent_id(pr),
                    status="completed",
                    phase=f"round_{r}",
                    detail=snippet or None,
                )
            )
            await _sse_discussion_turn(on_progress, role, r, out)
            if role == "CHALLENGER" and allow_sql_approval_pause:
                pq = (out.proposed_sub_question or "").strip()
                if (
                    out.needs_more_data
                    and pq
                    and is_valid_tts_sub_question(pq)
                    and host_plan is not None
                    and analyst_plan is not None
                ):
                    return LlmAgentsPaused(
                        pipeline=pipeline,
                        discussion=DiscussionState(analyst=analyst_out, turns=list(turns)),
                        question=question,
                        generated_sql=generated_sql,
                        primary_exe=exe,
                        deterministic_summary=deterministic_summary,
                        sub_results=sr_list,
                        host_plan=host_plan,
                        analyst_plan=analyst_plan,
                        proposed_sub_question=pq,
                        rationale=out.why,
                    )

        if r >= max_rounds:
            break

        round_turns = turns[round_slice_start:]
        mod_user = _moderator_user_prompt(ctx=ctx, analyst=analyst_out, turns=round_turns)
        mod = await _run_moderator_call(settings=settings, user_content=mod_user)
        await _sse_discussion_moderator(on_progress, mod)
        if not mod.continue_discussion:
            break
        focus = (mod.focus_for_next_round or "").strip() or None

    discussion = DiscussionState(analyst=analyst_out, turns=turns)
    host_out = await _run_agent_json(
        settings,
        "HOST",
        system_prompt_host_composer(),
        _host_user_prompt(ctx=ctx, discussion=discussion),
    )
    pipeline.append(
        AgentPipelineStep(
            id=_agent_id("HOST"),
            status="completed",
            phase=host_out.phase,
            detail=host_out.detail,
        )
    )
    return LlmAgentsComplete(answer=host_out.text, pipeline=pipeline, messages=[AgentInsight(role="HOST", text=host_out.text)])


async def run_llm_agents(
    *,
    settings: Settings,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
    sub_results: list[SubResult] | None = None,
    host_plan: PlanningHostOutput | None = None,
    analyst_plan: PlanningAnalystOutput | None = None,
    allow_sql_approval_pause: bool = True,
    on_progress: AgentProgressCallback = None,
) -> Union[LlmAgentsComplete, LlmAgentsPaused]:
    """
    Run internal enrichment, then HOST.
    When pulsecast_discussion_enabled: ANALYST once, then moderated M→F→C rounds (cap), else linear chain.
    If CHALLENGER requests more data and allow_sql_approval_pause, return LlmAgentsPaused (no HOST yet).
    """
    sr_list = list(sub_results) if sub_results is not None else []

    ctx = _context_blob_compact(
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
        sub_results=sr_list,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
    )

    max_rounds = max(1, settings.pulsecast_discussion_max_rounds)
    if settings.pulsecast_discussion_enabled:
        return await _run_llm_agents_moderated_discussion(
            settings=settings,
            question=question,
            generated_sql=generated_sql,
            exe=exe,
            deterministic_summary=deterministic_summary,
            sr_list=sr_list,
            host_plan=host_plan,
            analyst_plan=analyst_plan,
            ctx=ctx,
            allow_sql_approval_pause=allow_sql_approval_pause,
            max_rounds=max_rounds,
            on_progress=on_progress,
        )
    return await _run_llm_agents_linear(
        settings=settings,
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
        sr_list=sr_list,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
        ctx=ctx,
        allow_sql_approval_pause=allow_sql_approval_pause,
        on_progress=on_progress,
    )


async def run_llm_agents_host_only(
    *,
    settings: Settings,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
    sub_results: list[SubResult],
    host_plan: PlanningHostOutput,
    analyst_plan: PlanningAnalystOutput,
    discussion: DiscussionState,
    pipeline: list[AgentPipelineStep],
    user_declined_extra_sql: bool = False,
) -> LlmAgentsComplete:
    """Run only the HOST composer after HITL resume."""
    ctx = _context_blob_compact(
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
        sub_results=sub_results,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
    )
    if user_declined_extra_sql:
        ctx["user_declined_extra_sql"] = True
        ctx["hitl_note"] = (
            "The user declined to run an additional SQL query. "
            "Answer using only existing data samples in this context."
        )

    host_out = await _run_agent_json(
        settings,
        "HOST",
        system_prompt_host_composer(),
        _host_user_prompt(ctx=ctx, discussion=discussion),
    )
    pipeline_out = [
        *pipeline,
        AgentPipelineStep(
            id=_agent_id("HOST"),
            status="completed",
            phase=host_out.phase,
            detail=host_out.detail,
        ),
    ]
    return LlmAgentsComplete(
        answer=host_out.text,
        pipeline=pipeline_out,
        messages=[AgentInsight(role="HOST", text=host_out.text)],
    )
