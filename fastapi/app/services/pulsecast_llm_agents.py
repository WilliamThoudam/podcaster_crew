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
from app.services.data_quality_hints import sub_result_quality_hints
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


PulsecastRole = Literal["HOST", "ANALYST", "MARKETING", "FINANCE", "FORECASTER", "WEB_CRAWLER", "CHALLENGER"]
PulsecastAgentId = Literal["host", "analyst", "marketing", "finance", "forecaster", "web_crawler", "challenger"]
DiscussantRole = Literal["MARKETING", "FINANCE", "FORECASTER", "WEB_CRAWLER", "CHALLENGER"]

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
    search_queries: list[str] | None = None
    web_search_rationale: str | None = None
    continue_discussion: bool = False
    stop_reason: str | None = None
    focus_for_next_round: str | None = None

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
    sqlist = out.get("search_queries")
    if isinstance(sqlist, list):
        cleaned_sq = [str(x).strip() for x in sqlist if str(x).strip()]
        out["search_queries"] = cleaned_sq if cleaned_sq else None
    else:
        out["search_queries"] = None
    wsr = _coerce_optional_str(out.get("web_search_rationale"))
    if wsr and str(wsr).strip():
        out["web_search_rationale"] = wsr
    else:
        out["web_search_rationale"] = None

    return out


def _web_search_queries_for_hitl(
    wc_out: _AgentOut,
    analyst_plan: PlanningAnalystOutput | None,
) -> list[str]:
    """One Serper call per entry; prefer model `search_queries`, else single `search_query`, else planner lines."""
    if wc_out.search_queries:
        return list(wc_out.search_queries)
    sq = (wc_out.search_query or "").strip()
    if sq:
        return [sq]
    if analyst_plan and analyst_plan.web_sub_questions:
        return [str(x).strip() for x in analyst_plan.web_sub_questions if str(x).strip()]
    return []


def _normalize_web_query_key(s: str) -> str:
    """Normalize for dedupe: lowercase, collapse whitespace, strip common punctuation edges."""
    t = (s or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _existing_web_queries_in_ctx(ctx: dict[str, Any]) -> set[str]:
    existing: set[str] = set()
    by_q = ctx.get("web_search_results_by_query")
    if isinstance(by_q, list):
        for item in by_q:
            if isinstance(item, dict):
                q = str(item.get("query") or "").strip()
                if q:
                    existing.add(_normalize_web_query_key(q))
    return existing


def _filter_new_web_queries(ctx: dict[str, Any], queries: list[str]) -> list[str]:
    existing = _existing_web_queries_in_ctx(ctx)
    out: list[str] = []
    for q in queries:
        t = str(q).strip()
        if not t:
            continue
        key = _normalize_web_query_key(t)
        if key in existing:
            continue
        existing.add(key)
        out.append(t)
    return out

def _merge_completed_web_into_ctx(ctx: dict[str, Any], completed: list[dict[str, Any]]) -> None:
    """Flatten organic rows with source_query + keep per-query grouping for prompts."""
    by_q: list[dict[str, Any]] = []
    flat: list[dict[str, Any]] = []
    for item in completed:
        q = str(item.get("query") or "").strip()
        err = item.get("error")
        res = item.get("results")
        rows: list[dict[str, Any]] = res if isinstance(res, list) else []
        by_q.append({"query": q, "error": err, "results": rows})
        if err:
            continue
        for row in rows:
            if isinstance(row, dict):
                d = dict(row)
                d["source_query"] = q
                flat.append(d)
    if by_q:
        ctx["web_search_results_by_query"] = by_q
    if flat:
        ctx["web_search_results"] = flat


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


async def _sse_challenger_control(
    on_progress: AgentProgressCallback,
    out: _AgentOut,
) -> None:
    if not on_progress:
        return
    await emit_progress(
        on_progress,
        {
            "type": "discussion_moderator",
            "continue_discussion": out.continue_discussion,
            "reason": out.stop_reason or "",
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
    *,
    max_rows_per_result: int = 3,
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for sr in sub_results:
        rows = sr.execute.data or []
        entry: dict[str, Any] = {
            "sub_question": sr.sub_question,
            "generated_sql": sr.generated_sql,
            "rows_returned": len(rows),
            "columns_in_data": list(rows[0].keys()) if rows else [],
            "data_sample": _safe_sample(sr.execute, max_rows=max_rows_per_result),
        }
        if sr.original_sql is not None:
            entry["original_sql"] = sr.original_sql
            entry["was_aggregated"] = True
        compact.append(entry)
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
    settings: Settings | None = None,
) -> dict[str, Any]:
    primary_max = (
        settings.pulsecast_context_sample_max_rows if settings is not None else 10
    )
    sub_max = (
        settings.pulsecast_sub_result_sample_max_rows if settings is not None else 3
    )
    primary = exe.data or []
    base: dict[str, Any] = {
        "question": question,
        "generated_sql": generated_sql,
        "rows_returned": len(primary),
        "columns_in_data": list(primary[0].keys()) if primary else [],
        "data_sample": _safe_sample(exe, max_rows=primary_max),
        "deterministic_summary": deterministic_summary,
        "context_sample_meta": {
            "primary_max_rows": primary_max,
            "sub_result_max_rows_per_step": sub_max,
            "note": (
                "data_sample is truncated to primary_max_rows rows; rows_returned is the full "
                "result set size. Use rows_returned and sub_results[].rows_returned when judging coverage."
            ),
        },
    }
    if sub_results:
        base["sub_results"] = _compact_sub_results(
            sub_results,
            max_rows_per_result=sub_max,
        )
        base["sub_result_quality_hints"] = sub_result_quality_hints(sub_results)
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
        lines.append(f"Challenger focus for this round: {focus_for_next_round.strip()}\n")
    return "".join(lines)


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


def _host_final_answer_with_canonical(*, canonical_question: str, host_body: str) -> str:
    """Prepend merged canonical question to the HOST answer shown in the UI."""
    q = (canonical_question or "").strip()
    body_stripped = (host_body or "").strip()
    if not q:
        return body_stripped if body_stripped else (host_body or "")
    header = f"**Question addressed:** {q}"
    if not body_stripped:
        return header
    return f"{header}\n\n{body_stripped}"


def _agent_id(role: PulsecastRole) -> PulsecastAgentId:
    return {
        "HOST": "host",
        "ANALYST": "analyst",
        "MARKETING": "marketing",
        "FINANCE": "finance",
        "FORECASTER": "forecaster",
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

async def _invoke_json_object(*, settings: Settings, messages: list[dict[str, str]]) -> str:
    """
    Force the model to return a single JSON object (no surrounding prose).
    This avoids JSONDecodeError from unescaped control characters in string fields.
    """
    llm = build_chat_model(settings).bind(response_format={"type": "json_object"})
    lc_messages: list[SystemMessage | HumanMessage] = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    resp = await llm.ainvoke(lc_messages)
    text = _chunk_text(resp.content).strip()
    if not text:
        raise ValueError("Empty model content")
    return text


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


_WEB_SEARCH_SUMMARIZE_SYSTEM = (
    "You format organic web search results for an analytics assistant UI.\n\n"
    "CORE RULE — extract and map, do not invent:\n"
    "- **Extract** claims only from the provided titles/snippets/links. **Map** them to `user_question` "
    "(how they relate to what the user is trying to understand).\n"
    "- Do **not** fabricate statistics, brands, or trends not present in the snippets. If sources are thin or "
    "off-topic, say that briefly under Impact on Analysis.\n\n"
    "Output markdown with EXACTLY two sections in this order:\n\n"
    "### Web Insights (Relevant to [TopicPhrase])\n"
    "- The heading MUST use this pattern: `### Web Insights (Relevant to X)` where X is a short topic phrase "
    "(2–8 words, Title Case) derived from `user_question` and/or `search_query` (e.g. Regional Differences, "
    "Seasonal Demand, New Product Launches). If unclear, use: `### Web Insights (Relevant to your question)`.\n"
    "- After a blank line, output **3–4** lines starting with `- ` (markdown bullets). Each line: one sentence "
    "that states what the sources suggest, **as it relates to the user’s analytical angle**, in plain language "
    "(no citation numbers; no table).\n\n"
    "### Impact on Analysis\n"
    "- After a blank line, output **exactly 3** lines starting with `- `. Each is one sentence: how this external "
    "context **helps interpret** internal sales/warehouse-style results for the user’s question.\n"
    "- Examples of intent: explains why patterns might appear; suggests what to compare or validate; "
    "helps distinguish demand-driven vs supply-driven readings — only when grounded in what the sources actually "
    "support.\n\n"
    "- Do not add `#` or `##` headings. Do not repeat the results table. Keep total under ~240 words.\n"
)


async def _summarize_serper_organic_results(
    *,
    settings: Settings,
    context_question: str | None,
    search_query: str,
    organic: list[dict[str, Any]],
) -> str:
    """LLM summary of Serper organic rows for display below the results table."""
    rows: list[dict[str, str]] = []
    for r in organic:
        if not isinstance(r, dict):
            continue
        rows.append(
            {
                "title": str(r.get("title") or ""),
                "snippet": str(r.get("snippet") or ""),
                "link": str(r.get("link") or ""),
            }
        )
    if not rows:
        return ""
    payload = {
        "user_question": (context_question or "").strip() or None,
        "search_query": (search_query or "").strip(),
        "organic_results": rows,
    }
    user = (
        "Produce the two-section markdown described in your instructions. "
        "Extract from organic_results only; map each point to user_question. "
        "Use `### Web Insights (Relevant to …)` and `### Impact on Analysis` with the bullet counts specified.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        text = await _stream_llm_text(
            settings=settings,
            messages=[
                {"role": "system", "content": _WEB_SEARCH_SUMMARIZE_SYSTEM},
                {"role": "user", "content": user},
            ],
        )
        return (text or "").strip()
    except Exception:
        return "_Web insights unavailable._"


async def _emit_web_search_results_sse(
    on_progress: AgentProgressCallback,
    *,
    settings: Settings | None,
    query: str | None,
    markdown_body: str,
    context_question: str | None = None,
    organic_for_takeaways: list[dict[str, Any]] | None = None,
) -> None:
    """Stream Serper snapshot to UI (Web Crawler lane); optional LLM takeaways below the table."""
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
    take_rows = organic_for_takeaways if isinstance(organic_for_takeaways, list) else None
    if (
        settings is not None
        and getattr(settings, "web_search_summarize_enabled", True)
        and take_rows
        and len(take_rows) > 0
    ):
        takeaway_md = await _summarize_serper_organic_results(
            settings=settings,
            context_question=context_question,
            search_query=q,
            organic=take_rows,
        )
        if takeaway_md:
            block = "\n\n---\n\n" + takeaway_md.strip() + "\n"
            await emit_text_chunks(
                on_progress=on_progress,
                base_event=base,
                text=block,
                event_type="web_search_results_table_chunk",
                chunk_size=STREAM_CHUNK_SIZE,
                delay_s=STREAM_DELAY_S,
            )
    await emit_progress(on_progress, {**base, "type": "web_search_results_done"})


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
        text = await _invoke_json_object(
            settings=settings,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        if not text:
            raise ValueError(f"Empty streamed content from agent {role_label}")
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError("Host output JSON must be an object")
        obj = _normalize_host_out_dict(obj)
        return _HostOut.model_validate(obj)
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


@dataclass
class LlmAgentsComplete:
    answer: str
    pipeline: list[AgentPipelineStep]
    messages: list[AgentInsight]
    discussion: DiscussionState | None = None


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
    search_queries: list[str]
    pending_search_index: int
    completed_web_results: list[dict[str, Any]]
    rationale: str | None


@dataclass
class LlmAgentsPausedDiscussion:
    stage: Literal["pre", "mid"]
    question: str
    generated_sql: str
    primary_exe: ExecuteSqlResponse
    deterministic_summary: str
    sub_results: list[SubResult]
    host_plan: PlanningHostOutput
    analyst_plan: PlanningAnalystOutput
    openai_user: str | None = None

    requested_depth: Literal["linear", "moderated"] = "moderated"
    max_rounds: int = 3
    next_round_index: int = 1
    focus_for_next_round: str | None = None
    rationale: str | None = None

    pipeline: list[AgentPipelineStep] | None = None
    discussion: DiscussionState | None = None


def discussion_state_from_legacy_prior(prior: dict[str, _AgentOut]) -> DiscussionState | None:
    """Best-effort conversion for snapshot v1 (single pass)."""
    if "ANALYST" not in prior:
        return None
    analyst = prior["ANALYST"]
    turns: list[DiscussionTurn] = []
    for role in ("MARKETING", "FINANCE", "FORECASTER", "WEB_CRAWLER", "CHALLENGER"):
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
    for role in ("MARKETING", "FINANCE", "FORECASTER", "WEB_CRAWLER", "CHALLENGER"):
        pr: PulsecastRole = role  # MARKETING|FINANCE|FORECASTER|WEB_CRAWLER|CHALLENGER
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
    final_text = _host_final_answer_with_canonical(
        canonical_question=str(ctx.get("question") or ""),
        host_body=host_out.text,
    )
    return LlmAgentsComplete(
        answer=final_text,
        pipeline=pipeline,
        messages=[AgentInsight(role="HOST", text=final_text)],
        discussion=discussion,
    )


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
        "FORECASTER",
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
        elif role in ("MARKETING", "FINANCE", "FORECASTER", "WEB_CRAWLER", "CHALLENGER"):
            await _sse_discussion_turn(on_progress, role, 1, out)
        if role == "WEB_CRAWLER" and _serper_configured(settings):
            queries = _web_search_queries_for_hitl(out, analyst_plan)
            queries = _filter_new_web_queries(ctx, queries)
            queries = queries[:1]
            if out.needs_web_search and queries and host_plan is not None and analyst_plan is not None:
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
                    proposed_search_query=queries[0],
                    search_queries=queries,
                    pending_search_index=0,
                    completed_web_results=[],
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
    final_text = _host_final_answer_with_canonical(
        canonical_question=question,
        host_body=host_out.text,
    )
    return LlmAgentsComplete(
        answer=final_text,
        pipeline=pipeline,
        messages=[AgentInsight(role="HOST", text=final_text)],
        discussion=discussion,
    )


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
    initial_discussion: DiscussionState | None = None,
    initial_pipeline: list[AgentPipelineStep] | None = None,
    start_round: int = 1,
    focus_for_next_round: str | None = None,
) -> Union[LlmAgentsComplete, LlmAgentsPaused, LlmAgentsPausedWebSearch]:
    pipeline: list[AgentPipelineStep] = list(initial_pipeline) if initial_pipeline is not None else []
    turns: list[DiscussionTurn] = list(initial_discussion.turns) if initial_discussion is not None else []
    focus: str | None = focus_for_next_round

    if initial_discussion is not None:
        analyst_out = initial_discussion.analyst
    else:
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

    if start_round < 1:
        start_round = 1
    for r in range(start_round, max_rounds + 1):
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
            hitl = await _run_one_discussant(discussant="FORECASTER", round_index=r)
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
                queries = _web_search_queries_for_hitl(wc_out, analyst_plan)
                queries = _filter_new_web_queries(ctx, queries)
                queries = queries[:1]
                if (
                    wc_out.needs_web_search
                    and queries
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
                        proposed_search_query=queries[0],
                        search_queries=queries,
                        pending_search_index=0,
                        completed_web_results=[],
                        rationale=wc_out.web_search_rationale,
                    )
            hitl = await _run_one_discussant(discussant="CHALLENGER", round_index=r)
            if hitl is not None:
                return hitl
        else:
            for role in ("MARKETING", "FINANCE", "FORECASTER", "CHALLENGER"):
                hitl = await _run_one_discussant(discussant=role, round_index=r)
                if hitl is not None:
                    return hitl

        if r >= max_rounds:
            break

        challenger_turn = turns[-1]
        await _sse_challenger_control(on_progress, challenger_turn.output)
        if not challenger_turn.output.continue_discussion:
            break
        focus = (challenger_turn.output.focus_for_next_round or "").strip() or None
        # HITL pause before continuing to next round.
        if host_plan is None or analyst_plan is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Discussion pause requires host_plan and analyst_plan",
            )
        return LlmAgentsPausedDiscussion(
            stage="mid",
            requested_depth="moderated",
            pipeline=list(pipeline),
            discussion=DiscussionState(analyst=analyst_out, turns=list(turns)),
            question=question,
            generated_sql=generated_sql,
            primary_exe=exe,
            deterministic_summary=deterministic_summary,
            sub_results=sr_list,
            host_plan=host_plan,
            analyst_plan=analyst_plan,
            max_rounds=max_rounds,
            next_round_index=r + 1,
            focus_for_next_round=focus,
            rationale=challenger_turn.output.stop_reason,
        )

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
    final_text = _host_final_answer_with_canonical(
        canonical_question=question,
        host_body=host_out.text,
    )
    return LlmAgentsComplete(
        answer=final_text,
        pipeline=pipeline,
        messages=[AgentInsight(role="HOST", text=final_text)],
        discussion=discussion,
    )


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
    completed_web_results: list[dict[str, Any]] | None = None,
    user_declined_web_search: bool = False,
    allow_sql_approval_pause: bool = True,
    on_progress: AgentProgressCallback = None,
) -> Union[LlmAgentsComplete, LlmAgentsPaused, LlmAgentsPausedWebSearch, LlmAgentsPausedDiscussion]:
    """
    Run internal enrichment, then HOST.
    Routing uses host_plan.discussion_depth: minimal (ANALYST→HOST), linear (single M→F→WC→C pass), or moderated
    (multi-round). If moderated is requested, run the multi-round discussion loop.
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
        settings=settings,
    )
    if completed_web_results:
        _merge_completed_web_into_ctx(ctx, [dict(x) for x in completed_web_results])
    if user_declined_web_search:
        ctx["user_declined_web_search"] = True
        ctx["hitl_note"] = (
            "The user declined to run a public web search (or this step). "
            "Answer using warehouse samples and prior panel notes only; "
            "do not imply external web results were retrieved."
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
    if depth in ("linear", "moderated"):
        # HITL pause before discussion begins (linear or moderated).
        if host_plan is not None and analyst_plan is not None:
            return LlmAgentsPausedDiscussion(
                stage="pre",
                requested_depth=depth,  # type: ignore[arg-type]
                question=question,
                generated_sql=generated_sql,
                primary_exe=exe,
                deterministic_summary=deterministic_summary,
                sub_results=sr_list,
                host_plan=host_plan,
                analyst_plan=analyst_plan,
                max_rounds=max_rounds,
                next_round_index=1,
                focus_for_next_round=None,
                rationale="Start 1-round panel." if depth == "linear" else "Start multi-round panel.",
            )
        if depth == "moderated":
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


async def run_llm_agents_minimal_only(
    *,
    settings: Settings,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
    sub_results: list[SubResult],
    host_plan: PlanningHostOutput,
    analyst_plan: PlanningAnalystOutput,
    completed_web_results: list[dict[str, Any]] | None = None,
    user_declined_web_search: bool = False,
    on_progress: AgentProgressCallback = None,
) -> LlmAgentsComplete:
    """Force minimal path (ANALYST→HOST), ignoring host_plan.discussion_depth."""
    ctx = _context_blob_compact(
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
        sub_results=list(sub_results),
        host_plan=host_plan,
        analyst_plan=analyst_plan,
        settings=settings,
    )
    if completed_web_results:
        _merge_completed_web_into_ctx(ctx, [dict(x) for x in completed_web_results])
    if user_declined_web_search:
        ctx["user_declined_web_search"] = True
        ctx["hitl_note"] = (
            "The user declined to run a public web search (or this step). "
            "Answer using warehouse samples and prior panel notes only; "
            "do not imply external web results were retrieved."
        )
    return await _run_llm_agents_minimal(settings=settings, ctx=ctx, on_progress=on_progress)


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
        settings=settings,
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
    final_text = _host_final_answer_with_canonical(
        canonical_question=question,
        host_body=host_out.text,
    )
    return LlmAgentsComplete(
        answer=final_text,
        pipeline=pipeline_out,
        messages=[AgentInsight(role="HOST", text=final_text)],
        discussion=discussion,
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
    search_queries: list[str],
    pending_search_index: int,
    completed_web_results: list[dict[str, Any]],
    hitl_rationale: str | None,
    allow_sql_approval_pause: bool,
    on_progress: AgentProgressCallback = None,
) -> Union[LlmAgentsComplete, LlmAgentsPaused, LlmAgentsPausedWebSearch]:
    """After web-search HITL: Serper for current step (if approved); may pause again for next query; then CHALLENGER + HOST."""
    queries = [x.strip() for x in search_queries if str(x).strip()]
    if not queries and (proposed_search_query or "").strip():
        queries = [(proposed_search_query or "").strip()]
    idx = pending_search_index
    if idx < 0:
        idx = 0
    if queries and idx >= len(queries):
        idx = len(queries) - 1

    ctx = _context_blob_compact(
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
        sub_results=sub_results,
        host_plan=host_plan,
        analyst_plan=analyst_plan,
        settings=settings,
    )
    completed = [dict(x) for x in completed_web_results]

    if not approved:
        _merge_completed_web_into_ctx(ctx, completed)
        ctx["user_declined_web_search"] = True
        ctx["hitl_note"] = (
            "The user declined to run a public web search (or this step). Answer using warehouse samples and "
            "prior panel notes only; do not imply external web results were retrieved."
        )
        _md = _serper_results_markdown(query="", results=None, error=None, declined=True)
        await _emit_web_search_results_sse(
            on_progress,
            settings=settings,
            query=None,
            markdown_body=_md,
            context_question=question,
            organic_for_takeaways=None,
        )
    else:
        q = (edited_search_query or "").strip() or (proposed_search_query or "").strip()
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
                ctx["web_search_error"] = step_err
                ctx["web_search_query_attempted"] = q
        else:
            completed.append({"query": "", "results": None, "error": "empty query after approval"})

        _q = q
        if step_err:
            _md = _serper_results_markdown(query=_q, results=None, error=str(step_err), declined=False)
        elif not _q:
            _md = "_No search query was available after approval._"
        else:
            _md = _serper_results_markdown(
                query=_q,
                results=step_results,
                error=None,
                declined=False,
            )
        takeaway_rows: list[dict[str, Any]] | None = None
        if approved and not step_err and _q and step_results:
            takeaway_rows = list(step_results)
        await _emit_web_search_results_sse(
            on_progress,
            settings=settings,
            query=_q or None,
            markdown_body=_md,
            context_question=question,
            organic_for_takeaways=takeaway_rows,
        )

        more = bool(queries) and (idx + 1) < len(queries)
        if more:
            return LlmAgentsPausedWebSearch(
                pipeline=pipeline,
                discussion=discussion,
                question=question,
                generated_sql=generated_sql,
                primary_exe=exe,
                deterministic_summary=deterministic_summary,
                sub_results=sub_results,
                host_plan=host_plan,
                analyst_plan=analyst_plan,
                proposed_search_query=queries[idx + 1],
                search_queries=queries,
                pending_search_index=idx + 1,
                completed_web_results=completed,
                rationale=hitl_rationale,
            )

        _merge_completed_web_into_ctx(ctx, completed)
        if queries:
            ctx["web_search_query_used"] = "; ".join(queries)

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
    final_text = _host_final_answer_with_canonical(
        canonical_question=question,
        host_body=host_out.text,
    )
    return LlmAgentsComplete(
        answer=final_text,
        pipeline=pipeline_out,
        messages=[AgentInsight(role="HOST", text=final_text)],
        discussion=discussion_final,
    )
