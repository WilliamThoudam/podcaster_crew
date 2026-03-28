from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Union

import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.clients.openai_chat import chat_complete_stream_text
from app.config import Settings
from app.models.schemas import (
    AgentInsight,
    AgentPipelineStep,
    ExecuteSqlResponse,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)

PulsecastRole = Literal["HOST", "ANALYST", "MARKETING", "FINANCE", "CHALLENGER"]
PulsecastAgentId = Literal["host", "analyst", "marketing", "finance", "challenger"]


class _AgentOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str = Field(..., min_length=1)
    phase: str = Field(..., min_length=1)
    detail: str | None = None
    needs_more_data: bool = False
    proposed_sub_question: str | None = None
    why: str | None = None


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


def _parse_agent_json(raw: str) -> _AgentOut:
    s = raw.strip()
    m = _JSON_BLOCK.search(s)
    if not m:
        raise ValueError("Agent output did not contain a JSON object")
    obj = json.loads(m.group(0))
    return _AgentOut.model_validate(obj)


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


def _system_prompt_internal(role: PulsecastRole) -> str:
    base = (
        "You are a Pulsecast internal enrichment agent.\n"
        f"Your role is: {role}\n\n"
        "Database execution context includes ONLY the `data_sample` row objects (and per sub_question "
        "`data_sample` entries under sub_results), plus `rows_returned` and `columns_in_data` derived "
        "from that same data array. There is no separate metadata from the execute tool (no server "
        "rowCount beyond len(data), no limit flags, no column typing from the tool).\n"
        "Treat quantitative claims as supported only by values visible in those samples; SQL states intent.\n\n"
    )
    if role == "CHALLENGER":
        return (
            base
            + "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
            "Schema:\n"
            '{\n'
            '  "text": string,\n'
            '  "phase": string,\n'
            '  "detail": string|null,\n'
            '  "needs_more_data": boolean,\n'
            '  "proposed_sub_question": string|null,\n'
            '  "why": string|null\n'
            "}\n\n"
            "Rules:\n"
            "- Set needs_more_data true ONLY if the current samples are clearly insufficient to answer "
            "the user question with confidence (e.g. empty, wrong grain, missing dimension).\n"
            "- If true, proposed_sub_question MUST be one plain-English analytics sub-question (no SQL).\n"
            "- If false, set proposed_sub_question and why to null.\n"
            "- Base your response on the context JSON; keep text short (2-5 sentences).\n"
        )
    return (
        base
        + "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
        "Schema:\n"
        '{ "text": string, "phase": string, "detail": string|null }\n\n'
        "Rules:\n"
        "- Base your response strictly on the context JSON: question, generated_sql, data samples, planning.\n"
        "- Your output is internal notes for the Host composer, not a user-facing response.\n"
        "- If samples are empty or too thin, say so and suggest the smallest next query refinement.\n"
        "- Keep it short and actionable (2-5 sentences).\n"
    )


def _system_prompt_host_composer() -> str:
    return (
        "You are the HOST composer in Pulsecast.\n"
        "You produce the single final answer shown to the user. Prior agents only saw `data_sample` rows "
        "(and sub_results[].data_sample) from execute_sql—no extra execution metadata.\n"
        "Synthesize their internal JSON notes with that evidence; do not invent totals, limits, or cell "
        "values not present in the samples.\n"
        "If context contains user_declined_extra_sql true, the user chose not to run a follow-up query—"
        "answer only from existing samples; do not imply new data was loaded.\n\n"
        "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
        "Schema:\n"
        '{ "text": string, "phase": string, "detail": string|null }\n\n'
        "Rules:\n"
        "- Write one clean, user-facing answer (plain language; markdown lists ok).\n"
        "- Do not output internal debate or role-play dialogue.\n"
        "- If evidence is thin, state what the samples support and what would need another query.\n"
        "- Keep the answer concise and decision-oriented.\n"
    )


def _human_prompt(*, ctx: dict[str, Any], prior: dict[str, _AgentOut]) -> str:
    prior_obj = {k: v.model_dump() for k, v in prior.items()}
    return (
        "Context JSON:\n"
        f"{json.dumps(ctx, ensure_ascii=False)}\n\n"
        "Prior agent outputs JSON:\n"
        f"{json.dumps(prior_obj, ensure_ascii=False)}\n"
    )


def _agent_id(role: PulsecastRole) -> PulsecastAgentId:
    return {
        "HOST": "host",
        "ANALYST": "analyst",
        "MARKETING": "marketing",
        "FINANCE": "finance",
        "CHALLENGER": "challenger",
    }[role]


@dataclass
class LlmAgentsComplete:
    answer: str
    pipeline: list[AgentPipelineStep]
    messages: list[AgentInsight]


@dataclass
class LlmAgentsPaused:
    pipeline: list[AgentPipelineStep]
    prior: dict[str, _AgentOut]
    question: str
    generated_sql: str
    primary_exe: ExecuteSqlResponse
    deterministic_summary: str
    sub_results: list[SubResult]
    host_plan: PlanningHostOutput
    analyst_plan: PlanningAnalystOutput
    proposed_sub_question: str
    rationale: str | None


async def _run_role_call(
    *,
    settings: Settings,
    role: PulsecastRole,
    system_prompt: str,
    ctx: dict[str, Any],
    prior: dict[str, _AgentOut],
) -> _AgentOut:
    try:
        text_parts: list[str] = []
        async for delta in chat_complete_stream_text(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=settings.openai_temperature,
            timeout_seconds=settings.openai_timeout_seconds,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": _human_prompt(ctx=ctx, prior=prior)},
            ],
        ):
            text_parts.append(delta)
        text = "".join(text_parts).strip()
        if not text:
            raise ValueError(f"Empty streamed content from agent {role}")
        return _parse_agent_json(text)
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM agent {role} failed: {e}",
        ) from e


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
) -> Union[LlmAgentsComplete, LlmAgentsPaused]:
    """
    Run internal enrichment calls (Analyst→Marketing→Finance→Challenger).
    If CHALLENGER requests more data and allow_sql_approval_pause, return LlmAgentsPaused (no HOST yet).
    Otherwise run HOST and return LlmAgentsComplete.
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

    internal_roles: list[PulsecastRole] = ["ANALYST", "MARKETING", "FINANCE", "CHALLENGER"]
    prior: dict[str, _AgentOut] = {}
    pipeline: list[AgentPipelineStep] = []

    for role in internal_roles:
        out = await _run_role_call(
            settings=settings,
            role=role,
            system_prompt=_system_prompt_internal(role),
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
        if role == "CHALLENGER" and allow_sql_approval_pause:
            pq = (out.proposed_sub_question or "").strip()
            if (
                out.needs_more_data
                and pq
                and host_plan is not None
                and analyst_plan is not None
            ):
                return LlmAgentsPaused(
                    pipeline=pipeline,
                    prior=prior,
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

    host_out = await _run_role_call(
        settings=settings,
        role="HOST",
        system_prompt=_system_prompt_host_composer(),
        ctx=ctx,
        prior=prior,
    )
    prior["HOST"] = host_out
    pipeline.append(
        AgentPipelineStep(
            id=_agent_id("HOST"),
            status="completed",
            phase=host_out.phase,
            detail=host_out.detail,
        )
    )

    answer = host_out.text
    messages = [AgentInsight(role="HOST", text=answer)]
    return LlmAgentsComplete(answer=answer, pipeline=pipeline, messages=messages)


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
    prior: dict[str, _AgentOut],
    pipeline: list[AgentPipelineStep],
    user_declined_extra_sql: bool = False,
) -> LlmAgentsComplete:
    """Run only the HOST composer after HITL resume (prior contains ANALYST..CHALLENGER)."""
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

    host_out = await _run_role_call(
        settings=settings,
        role="HOST",
        system_prompt=_system_prompt_host_composer(),
        ctx=ctx,
        prior=prior,
    )
    prior_with_host = {**prior, "HOST": host_out}
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

