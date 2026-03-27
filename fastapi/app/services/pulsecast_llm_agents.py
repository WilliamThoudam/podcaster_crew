from __future__ import annotations

import json
import re
from typing import Any, Literal

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
        compact.append(
            {
                "sub_question": sr.sub_question,
                "generated_sql": sr.generated_sql,
                "rowCount": sr.execute.rowCount,
                "limited": sr.execute.limited,
                "note": sr.execute.note,
                "sample_rows": _safe_sample(sr.execute, max_rows=max_rows_per_result),
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
    base: dict[str, Any] = {
        "question": question,
        "generated_sql": generated_sql,
        "executed_query": exe.query,
        "rowCount": exe.rowCount,
        "limited": exe.limited,
        "maxRecords": exe.maxRecords,
        "note": exe.note,
        "fields": [f.model_dump() for f in (exe.fields or [])],
        "sample_rows": _safe_sample(exe),
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
    return (
        "You are a Pulsecast internal enrichment agent.\n"
        f"Your role is: {role}\n\n"
        "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
        "Schema:\n"
        '{ "text": string, "phase": string, "detail": string|null }\n\n'
        "Rules:\n"
        "- Base your response strictly on the provided context (SQL + execution results).\n"
        "- Your output is internal notes for the Host composer, not a user-facing response.\n"
        "- If data is insufficient, say what is missing and suggest the smallest next query refinement.\n"
        "- Keep it short and actionable (2-5 sentences).\n"
    )


def _system_prompt_host_composer() -> str:
    return (
        "You are the HOST composer in Pulsecast.\n"
        "You synthesize one final user-facing answer from SQL evidence and internal agent notes.\n\n"
        "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
        "Schema:\n"
        '{ "text": string, "phase": string, "detail": string|null }\n\n'
        "Rules:\n"
        "- Write one clean final answer for the user.\n"
        "- Do not output internal debate or role-play dialogue.\n"
        "- If evidence is limited, clearly state the constraint and smallest next data step.\n"
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
) -> tuple[str, list[AgentPipelineStep], list[AgentInsight]]:
    """
    Run internal enrichment calls (Analyst→Marketing→Finance→Challenger)
    followed by a final Host composer call.
    """
    ctx = _context_blob_compact(
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
        sub_results=sub_results,
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
    return answer, pipeline, messages

