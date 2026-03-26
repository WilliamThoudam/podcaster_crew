from __future__ import annotations

import json
import re
from typing import Any, Literal

import httpx
from fastapi import HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.clients.openai_chat import chat_complete_json, extract_assistant_text
from app.config import Settings
from app.models.schemas import AgentInsight, AgentPipelineStep, ExecuteSqlResponse

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


def _context_blob(
    *,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
) -> dict[str, Any]:
    return {
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


def _system_prompt(role: PulsecastRole) -> str:
    return (
        "You are a Pulsecast agent in a multi-agent analytics panel.\n"
        f"Your role is: {role}\n\n"
        "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
        "Schema:\n"
        '{ "text": string, "phase": string, "detail": string|null }\n\n'
        "Rules:\n"
        "- Base your response strictly on the provided context (SQL + execution results).\n"
        "- If data is insufficient, say what is missing and suggest the smallest next query refinement.\n"
        "- Keep it short and actionable (2-5 sentences).\n"
    )


def _human_prompt(*, ctx: dict[str, Any], prior: dict[PulsecastRole, _AgentOut]) -> str:
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


async def run_llm_agents(
    *,
    settings: Settings,
    question: str,
    generated_sql: str,
    exe: ExecuteSqlResponse,
    deterministic_summary: str,
) -> tuple[str, list[AgentPipelineStep], list[AgentInsight]]:
    """
    Run 5 sequential OpenAI-compatible LLM calls (mandatory), returning:
    - answer: Analyst text
    - pipeline: 5 steps (host→challenger) with phase/detail
    - agent_messages: 5 chat messages (HOST…CHALLENGER)
    """
    ctx = _context_blob(
        question=question,
        generated_sql=generated_sql,
        exe=exe,
        deterministic_summary=deterministic_summary,
    )

    roles: list[PulsecastRole] = ["HOST", "ANALYST", "MARKETING", "FINANCE", "CHALLENGER"]
    prior: dict[PulsecastRole, _AgentOut] = {}
    pipeline: list[AgentPipelineStep] = []
    messages: list[AgentInsight] = []

    for role in roles:
        try:
            body = await chat_complete_json(
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=settings.openai_temperature,
                timeout_seconds=settings.openai_timeout_seconds,
                messages=[
                    {"role": "system", "content": _system_prompt(role)},
                    {"role": "user", "content": _human_prompt(ctx=ctx, prior=prior)},
                ],
            )
            text = extract_assistant_text(body)
            out = _parse_agent_json(text)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as e:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"LLM agent {role} failed: {e}",
            ) from e

        prior[role] = out
        pipeline.append(
            AgentPipelineStep(
                id=_agent_id(role),
                status="completed",
                phase=out.phase,
                detail=out.detail,
            )
        )
        messages.append(AgentInsight(role=role, text=out.text))

    analyst_answer = prior["ANALYST"].text if "ANALYST" in prior else messages[1].text
    return analyst_answer, pipeline, messages

