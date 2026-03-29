from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Union

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import Settings
from app.llm.chat_model import build_chat_model
from app.prompts.llm_agents import (
    system_prompt_host_composer,
    system_prompt_host_composer_minimal,
    system_prompt_internal,
    system_prompt_moderator,
    system_prompt_web_crawler,
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
from app.clients.serper_search import serper_google_search
from app.services.sub_question_tts_guard import is_valid_tts_sub_question


def _markdown_escape_cell(s: str) -> str:
    t = (s or "").replace("\n", " ").replace("\r", " ")
    return t.replace("|", "\\|")


def _serper_results_markdown(
    *,
    query: str,
    results: list[dict[str, Any]] | None,
    error: str | None,
    declined: bool,
) -> str:
    if declined:
        return "_User declined web search. Continuing without Serper results._"
    if error:
        return f"**Serper error:** {_markdown_escape_cell(error)}"
    rows = results or []
    if not rows:
        return "_No organic search results returned._"
    lines = [
        "| # | Title | Snippet | Source |",
        "| --- | --- | --- | --- |",
    ]
    for i, r in enumerate(rows, start=1):
        title = _markdown_escape_cell(str(r.get("title") or ""))
        snip = _markdown_escape_cell(str(r.get("snippet") or ""))[:400]
        link = _markdown_escape_cell(str(r.get("link") or ""))
        lines.append(f"| {i} | {title} | {snip} | {link} |")
    return "\n".join(lines)


async def _emit_web_search_results_sse(
    on_progress: AgentProgressCallback,
    *,
    query: str | None,
    markdown_body: str,
) -> None:
    """Stream Serper snapshot to UI (Web Crawler lane), same pattern as execute_table_chunk."""
    if not on_progress:
        return
    q = (query or "").strip()
    base: dict[str, Any] = {"query": q}
    await emit_progress(on_progress, {**base, "type": "web_search_results_started"})
    header = f"**Web search**\n\n**Query:** {q}\n\n" if q else "**Web search**\n\n"
    await emit_text_chunks(
        on_progress=on_progress,
        base_event=base,
        text=header,
        event_type="web_search_results_label_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await emit_text_chunks(
        on_progress=on_progress,
        base_event=base,
        text=markdown_body,
        event_type="web_search_results_table_chunk",
        chunk_size=STREAM_CHUNK_SIZE,
        delay_s=STREAM_DELAY_S,
    )
    await emit_progress(on_progress, {**base, "type": "web_search_results_done"})

PulsecastRole = Literal["HOST", "ANALYST", "MARKETING", "FINANCE", "WEB_CRAWLER", "CHALLENGER"]
PulsecastAgentId = Literal["host", "analyst", "marketing", "finance", "web_crawler", "challenger"]
DiscussantRole = Literal["MARKETING", "FINANCE", "WEB_CRAWLER", "CHALLENGER"]

AgentProgressCallback = Callable[[dict[str, Any]], Union[Awaitable[None], None]] | None


class _HostOut(BaseModel):
    """HOST composer only; unchanged JSON shape (text / phase / detail)."""

    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1)
    phase: str = Field(..., min_length=1)
    detail: str | None = None


class _AgentOut(BaseModel):
    """Unified internal panel output (ANALYST, MARKETING, FINANCE, CHALLENGER)."""

    model_config = ConfigDict(extra="ignore")

    insight: str = Field(..., min_length=1)
    reasoning: str = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    phase: str = Field(..., min_length=1)
    detail: str | None = None
    headline: str | None = None
    needs_more_data: bool = False
    new_question: str | None = None
    new_question_rationale: str | None = None
    needs_web_search: bool = False
    search_query: str | None = None
    web_search_rationale: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_panel_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return _normalize_panel_agent_out_dict(data)
        return data


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


_DEFAULT_REASONING = "Derived from the samples and role mandate as described in insight."


def _normalize_panel_agent_out_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Map legacy keys (text, proposed_sub_question, why) and defaults before validation."""
    out = dict(obj)

    ins = out.get("insight")
    tx = out.get("text")
    if ins is not None and str(ins).strip():
        out["insight"] = _coerce_required_str(ins, field="insight")
    elif tx is not None and str(tx).strip():
        out["insight"] = _coerce_required_str(tx, field="text")
    else:
        # Models sometimes omit insight (e.g. WEB_CRAWLER with only search_query). Synthesize from other fields.
        hl = _coerce_optional_str(out.get("headline"))
        ph0 = str(out.get("phase") or "").strip()
        det0 = _coerce_optional_str(out.get("detail"))
        wsr0 = _coerce_optional_str(out.get("web_search_rationale"))
        sq0 = _coerce_optional_str(out.get("search_query"))
        parts: list[str] = []
        if hl and str(hl).strip():
            parts.append(str(hl).strip())
        if ph0:
            parts.append(ph0)
        if det0 and str(det0).strip():
            parts.append(str(det0).strip())
        if wsr0 and str(wsr0).strip():
            parts.append(str(wsr0).strip())
        elif sq0 and str(sq0).strip():
            parts.append(f"Proposed web search: {sq0.strip()}")
        if parts:
            out["insight"] = " ".join(parts)[:8000]
        else:
            out["insight"] = (
                "The model returned structured fields without narrative text; "
                "downstream steps will use search flags and rationale if present."
            )

    rs = out.get("reasoning")
    if rs is None or not str(rs).strip():
        out["reasoning"] = _DEFAULT_REASONING
    else:
        out["reasoning"] = _coerce_required_str(rs, field="reasoning")

    c = out.get("confidence")
    if c is None:
        out["confidence"] = 0.7
    else:
        try:
            cf = float(c)
            out["confidence"] = max(0.0, min(1.0, cf))
        except (TypeError, ValueError):
            out["confidence"] = 0.7

    ph_raw = out.get("phase")
    if ph_raw is not None and str(ph_raw).strip():
        out["phase"] = _coerce_required_str(ph_raw, field="phase")
    else:
        out["phase"] = "Panel review"

    nq = _coerce_optional_str(out.get("new_question"))
    psq = _coerce_optional_str(out.get("proposed_sub_question"))
    if nq and str(nq).strip():
        out["new_question"] = nq
    elif psq and str(psq).strip():
        out["new_question"] = psq
    else:
        out["new_question"] = None

    nr = _coerce_optional_str(out.get("new_question_rationale"))
    why = _coerce_optional_str(out.get("why"))
    if nr and str(nr).strip():
        out["new_question_rationale"] = nr
    elif why and str(why).strip():
        out["new_question_rationale"] = why
    else:
        out["new_question_rationale"] = None

    for key in ("detail", "headline"):
        if key in out:
            out[key] = _coerce_optional_str(out[key])

    if "needs_more_data" in out:
        out["needs_more_data"] = bool(out["needs_more_data"])

    if "needs_web_search" in out:
        out["needs_web_search"] = bool(out["needs_web_search"])
    sq = _coerce_optional_str(out.get("search_query"))
    if sq and str(sq).strip():
        out["search_query"] = sq
    else:
        out["search_query"] = None
    wsr = _coerce_optional_str(out.get("web_search_rationale"))
    if wsr and str(wsr).strip():
        out["web_search_rationale"] = wsr
    else:
        out["web_search_rationale"] = None

    return out


def _normalize_host_out_dict(obj: dict[str, Any]) -> dict[str, Any]:
    out = dict(obj)
    if "text" in out:
        out["text"] = _coerce_required_str(out["text"], field="text")
    if "phase" in out:
        out["phase"] = _coerce_required_str(out["phase"], field="phase")
    if "detail" in out:
        out["detail"] = _coerce_optional_str(out["detail"])
    return out


def _parse_panel_agent_json(raw: str) -> _AgentOut:
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if not m:
        raise ValueError("Agent output did not contain a JSON object")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Agent output JSON must be an object")
    return _AgentOut.model_validate(obj)


def _parse_host_json(raw: str) -> _HostOut:
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if not m:
        raise ValueError("Host output did not contain a JSON object")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Host output JSON must be an object")
    obj = _normalize_host_out_dict(obj)
    return _HostOut.model_validate(obj)


def _parse_moderator_json(raw: str) -> _ModeratorOut:
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if not m:
        raise ValueError("Moderator output did not contain a JSON object")
    obj = json.loads(m.group(0))
    return _ModeratorOut.model_validate(obj)


def _discussion_markdown(out: _AgentOut) -> str:
    lead = f"**{out.headline}**\n\n" if (out.headline and out.headline.strip()) else ""
    parts = [f"{lead}**{out.phase}**\n\n{out.insight}"]
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
            "discussion_depth": host_plan.discussion_depth,
        }
    if analyst_plan:
        base["analyst_plan"] = {
            "sub_questions": analyst_plan.sub_questions,
            "web_sub_questions": analyst_plan.web_sub_questions,
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
        {
            "round": t.round_index,
            "role": t.role,
            "insight": t.output.insight,
            "confidence": t.output.confidence,
            "detail": t.output.detail,
        }
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
        "WEB_CRAWLER": "web_crawler",
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
    return await _run_panel_agent_json(settings, role, system_prompt, user_content)


async def _run_web_crawler_call(
    *,
    settings: Settings,
    ctx: dict[str, Any],
    prior: dict[str, _AgentOut],
) -> _AgentOut:
    return await _run_role_call(
        settings=settings,
        role="WEB_CRAWLER",
        system_prompt=system_prompt_web_crawler(),
        ctx=ctx,
        prior=prior,
    )


async def _run_panel_agent_json(
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
        return _parse_panel_agent_json(text)
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


async def _run_host_json(
    settings: Settings,
    role_label: str,
    system_prompt: str,
    user_content: str,
) -> _HostOut:
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
        return _parse_host_json(text)
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


@dataclass
class LlmAgentsPausedWebSearch:
    pipeline: list[AgentPipelineStep]
    discussion: DiscussionState
    question: str
    generated_sql: str
    primary_exe: ExecuteSqlResponse
    deterministic_summary: str
    sub_results: list[SubResult]
    host_plan: PlanningHostOutput
    analyst_plan: PlanningAnalystOutput
    proposed_search_query: str
    rationale: str | None


def discussion_state_from_legacy_prior(prior: dict[str, _AgentOut]) -> DiscussionState | None:
    """Best-effort conversion for snapshot v1 (single pass)."""
    if "ANALYST" not in prior:
        return None
    analyst = prior["ANALYST"]
    turns: list[DiscussionTurn] = []
    for role in ("MARKETING", "FINANCE", "WEB_CRAWLER", "CHALLENGER"):
        if role in prior:
            turns.append(DiscussionTurn(role=role, round_index=1, output=prior[role]))
    return DiscussionState(analyst=analyst, turns=turns)


def _serper_configured(settings: Settings) -> bool:
    """Web search HITL is only offered when Serper can run after approval."""
    return bool((settings.serper_api_key or "").strip())


async def _run_llm_agents_minimal(
    *,
    settings: Settings,
    ctx: dict[str, Any],
    on_progress: AgentProgressCallback,
) -> LlmAgentsComplete:
    """ANALYST then HOST only. No Marketing/Finance/Challenger — no CHALLENGER sql_approval_pause."""
    pipeline: list[AgentPipelineStep] = []
    analyst_out = await _run_panel_agent_json(
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
    for role in ("MARKETING", "FINANCE", "WEB_CRAWLER", "CHALLENGER"):
        pr: PulsecastRole = role  # MARKETING|FINANCE|WEB_CRAWLER|CHALLENGER
        pipeline.append(
            AgentPipelineStep(
                id=_agent_id(pr),
                status="skipped",
                phase="Skipped (minimal)",
                detail=None,
            )
        )
    discussion = DiscussionState(analyst=analyst_out, turns=[])
    host_out = await _run_host_json(
        settings,
        "HOST",
        system_prompt_host_composer_minimal(),
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
) -> Union[LlmAgentsComplete, LlmAgentsPaused, LlmAgentsPausedWebSearch]:
    internal_roles: list[PulsecastRole] = [
        "ANALYST",
        "MARKETING",
        "FINANCE",
        "WEB_CRAWLER",
        "CHALLENGER",
    ]
    prior: dict[str, _AgentOut] = {}
    pipeline: list[AgentPipelineStep] = []

    for role in internal_roles:
        if role == "WEB_CRAWLER":
            out = await _run_web_crawler_call(settings=settings, ctx=ctx, prior=prior)
        else:
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
        elif role in ("MARKETING", "FINANCE", "WEB_CRAWLER", "CHALLENGER"):
            await _sse_discussion_turn(on_progress, role, 1, out)
        if role == "WEB_CRAWLER" and _serper_configured(settings):
            sq = (out.search_query or "").strip()
            if out.needs_web_search and sq and host_plan is not None and analyst_plan is not None:
                ds = discussion_state_from_legacy_prior(prior)
                if ds is None:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to build discussion state for web search pause",
                    )
                return LlmAgentsPausedWebSearch(
                    pipeline=pipeline,
                    discussion=ds,
                    question=question,
                    generated_sql=generated_sql,
                    primary_exe=exe,
                    deterministic_summary=deterministic_summary,
                    sub_results=sr_list,
                    host_plan=host_plan,
                    analyst_plan=analyst_plan,
                    proposed_search_query=sq,
                    rationale=out.web_search_rationale,
                )
        if role == "CHALLENGER" and allow_sql_approval_pause:
            pq = (out.new_question or "").strip()
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
                    rationale=out.new_question_rationale,
                )

    discussion = discussion_state_from_legacy_prior(prior)
    if discussion is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build discussion state after linear agent run",
        )
    host_out = await _run_host_json(
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
) -> Union[LlmAgentsComplete, LlmAgentsPaused, LlmAgentsPausedWebSearch]:
    pipeline: list[AgentPipelineStep] = []
    analyst_out = await _run_panel_agent_json(
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

    async def _run_one_discussant(
        *,
        discussant: DiscussantRole,
        round_index: int,
    ) -> LlmAgentsPaused | None:
        nonlocal turns
        user_content = _discussion_user_prompt(
            ctx=ctx,
            analyst=analyst_out,
            turns=turns,
            role=discussant,
            round_index=round_index,
            focus_for_next_round=focus if round_index > 1 else None,
        )
        pr: PulsecastRole = discussant
        out = await _run_panel_agent_json(
            settings,
            pr,
            system_prompt_internal(pr, discussion_aware=True),
            user_content,
        )
        turns.append(DiscussionTurn(role=discussant, round_index=round_index, output=out))
        snippet = (out.detail or out.insight or out.phase or "")[:500]
        pipeline.append(
            AgentPipelineStep(
                id=_agent_id(pr),
                status="completed",
                phase=f"round_{round_index}",
                detail=snippet or None,
            )
        )
        await _sse_discussion_turn(on_progress, pr, round_index, out)
        if discussant == "CHALLENGER" and allow_sql_approval_pause:
            pq = (out.new_question or "").strip()
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
                    rationale=out.new_question_rationale,
                )
        return None

    for r in range(1, max_rounds + 1):
        if on_progress:
            await emit_progress(on_progress, {"type": "discussion_round_started", "round": r})
        round_slice_start = len(turns)
        if r == 1:
            hitl = await _run_one_discussant(discussant="MARKETING", round_index=r)
            if hitl is not None:
                return hitl
            hitl = await _run_one_discussant(discussant="FINANCE", round_index=r)
            if hitl is not None:
                return hitl
            prior_wc: dict[str, _AgentOut] = {"ANALYST": analyst_out}
            for t in turns:
                prior_wc[t.role] = t.output
            wc_out = await _run_web_crawler_call(settings=settings, ctx=ctx, prior=prior_wc)
            turns.append(DiscussionTurn(role="WEB_CRAWLER", round_index=r, output=wc_out))
            wc_snippet = (wc_out.detail or wc_out.insight or wc_out.phase or "")[:500]
            pipeline.append(
                AgentPipelineStep(
                    id=_agent_id("WEB_CRAWLER"),
                    status="completed",
                    phase=f"round_{r}",
                    detail=wc_snippet or None,
                )
            )
            await _sse_discussion_turn(on_progress, "WEB_CRAWLER", r, wc_out)
            if _serper_configured(settings):
                sq = (wc_out.search_query or "").strip()
                if (
                    wc_out.needs_web_search
                    and sq
                    and host_plan is not None
                    and analyst_plan is not None
                ):
                    return LlmAgentsPausedWebSearch(
                        pipeline=pipeline,
                        discussion=DiscussionState(analyst=analyst_out, turns=list(turns)),
                        question=question,
                        generated_sql=generated_sql,
                        primary_exe=exe,
                        deterministic_summary=deterministic_summary,
                        sub_results=sr_list,
                        host_plan=host_plan,
                        analyst_plan=analyst_plan,
                        proposed_search_query=sq,
                        rationale=wc_out.web_search_rationale,
                    )
            hitl = await _run_one_discussant(discussant="CHALLENGER", round_index=r)
            if hitl is not None:
                return hitl
        else:
            for role in ("MARKETING", "FINANCE", "CHALLENGER"):
                hitl = await _run_one_discussant(discussant=role, round_index=r)
                if hitl is not None:
                    return hitl

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
    host_out = await _run_host_json(
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
) -> Union[LlmAgentsComplete, LlmAgentsPaused, LlmAgentsPausedWebSearch]:
    """
    Run internal enrichment, then HOST.
    Routing uses host_plan.discussion_depth: minimal (ANALYST→HOST), linear (single M→F→WC→C pass), or moderated
    (multi-round when pulsecast_discussion_enabled). If moderated but discussion is globally disabled, uses linear.
    If depth is minimal but analyst_plan.web_sub_questions is non-empty, depth is treated as linear so WEB_CRAWLER runs.
    If CHALLENGER requests more data and allow_sql_approval_pause, return LlmAgentsPaused (no HOST yet) — not on minimal path.
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

    depth = host_plan.discussion_depth if host_plan else "moderated"
    # Minimal path skips Marketing/Finance/WEB_CRAWLER/Challenger. Planning may still put public-web intents
    # in web_sub_questions — those require WEB_CRAWLER (and optional Serper HITL), so use at least linear.
    if (
        depth == "minimal"
        and analyst_plan
        and analyst_plan.web_sub_questions
    ):
        depth = "linear"

    if depth == "minimal":
        return await _run_llm_agents_minimal(
            settings=settings,
            ctx=ctx,
            on_progress=on_progress,
        )

    max_rounds = max(1, settings.pulsecast_discussion_max_rounds)
    use_moderated = depth == "moderated" and settings.pulsecast_discussion_enabled
    if use_moderated:
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

    host_out = await _run_host_json(
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


async def run_llm_agents_after_web_hitl(
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
    approved: bool,
    edited_search_query: str | None,
    proposed_search_query: str,
    allow_sql_approval_pause: bool,
    on_progress: AgentProgressCallback = None,
) -> Union[LlmAgentsComplete, LlmAgentsPaused]:
    """After web-search HITL: Serper (if approved), then CHALLENGER + HOST."""
    ctx = _context_blob_compact(
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
        sub_results=sub_results,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
    )
    if approved:
        q = (edited_search_query or "").strip() or (proposed_search_query or "").strip()
        if q:
            try:
                results = await serper_google_search(settings=settings, query=q, num=8)
                ctx["web_search_query_used"] = q
                ctx["web_search_results"] = results
            except Exception as e:
                ctx["web_search_error"] = str(e)
                ctx["web_search_query_attempted"] = q
    else:
        ctx["user_declined_web_search"] = True
        ctx["hitl_note"] = (
            "The user declined to run a public web search. Answer using warehouse samples and "
            "prior panel notes only; do not imply external web results were retrieved."
        )

    if not approved:
        _md = _serper_results_markdown(query="", results=None, error=None, declined=True)
        await _emit_web_search_results_sse(on_progress, query=None, markdown_body=_md)
    else:
        _q = (edited_search_query or "").strip() or (proposed_search_query or "").strip()
        _err = ctx.get("web_search_error")
        if _err:
            _md = _serper_results_markdown(query=_q, results=None, error=str(_err), declined=False)
        elif not _q:
            _md = "_No search query was available after approval._"
        else:
            _md = _serper_results_markdown(
                query=_q,
                results=ctx.get("web_search_results"),
                error=None,
                declined=False,
            )
        await _emit_web_search_results_sse(on_progress, query=_q or None, markdown_body=_md)

    prior: dict[str, _AgentOut] = {"ANALYST": discussion.analyst}
    for t in discussion.turns:
        prior[t.role] = t.output

    ch_out = await _run_role_call(
        settings=settings,
        role="CHALLENGER",
        system_prompt=system_prompt_internal("CHALLENGER", discussion_aware=False),
        ctx=ctx,
        prior=prior,
    )
    pipeline_out = [
        *pipeline,
        AgentPipelineStep(
            id=_agent_id("CHALLENGER"),
            status="completed",
            phase=ch_out.phase,
            detail=ch_out.detail,
        ),
    ]
    await _sse_discussion_turn(on_progress, "CHALLENGER", 1, ch_out)

    if allow_sql_approval_pause:
        pq = (ch_out.new_question or "").strip()
        if (
            ch_out.needs_more_data
            and pq
            and is_valid_tts_sub_question(pq)
        ):
            turns_final = [
                *discussion.turns,
                DiscussionTurn(role="CHALLENGER", round_index=1, output=ch_out),
            ]
            return LlmAgentsPaused(
                pipeline=pipeline_out,
                discussion=DiscussionState(analyst=discussion.analyst, turns=turns_final),
                question=question,
                generated_sql=generated_sql,
                primary_exe=exe,
                deterministic_summary=deterministic_summary,
                sub_results=sub_results,
                host_plan=host_plan,
                analyst_plan=analyst_plan,
                proposed_sub_question=pq,
                rationale=ch_out.new_question_rationale,
            )

    turns_final = [
        *discussion.turns,
        DiscussionTurn(role="CHALLENGER", round_index=1, output=ch_out),
    ]
    discussion_final = DiscussionState(analyst=discussion.analyst, turns=turns_final)

    host_out = await _run_host_json(
        settings,
        "HOST",
        system_prompt_host_composer(),
        _host_user_prompt(ctx=ctx, discussion=discussion_final),
    )
    pipeline_out.append(
        AgentPipelineStep(
            id=_agent_id("HOST"),
            status="completed",
            phase=host_out.phase,
            detail=host_out.detail,
        )
    )
    return LlmAgentsComplete(
        answer=host_out.text,
        pipeline=pipeline_out,
        messages=[AgentInsight(role="HOST", text=host_out.text)],
    )
