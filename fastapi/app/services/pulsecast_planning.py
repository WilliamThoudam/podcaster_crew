from __future__ import annotations

import json
import re
from typing import Any

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm.chat_model import build_chat_model
from app.models.schemas import PlanningAnalystOutput, PlanningHostOutput


class _HostOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_focus: str = Field(..., min_length=1)
    time_window: str | None = None
    region_focus: str | None = None
    metrics: list[str] = Field(default_factory=list)
    notes: str | None = None


class _AnalystOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub_questions: list[str] = Field(default_factory=list)
    rationale: str | None = None


_SQLISH_PATTERN = re.compile(
    r"\b(select|with|from|join|group\s+by|order\s+by|having|where|limit|row_number|rank)\b",
    re.IGNORECASE,
)


def _looks_like_sql(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return bool(_SQLISH_PATTERN.search(s)) or ";" in s


def _clean_sub_question(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^sub-?question\s*\d+\s*:\s*", "", s, flags=re.IGNORECASE)
    s = s.replace("```sql", "").replace("```", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _host_system_prompt() -> str:
    return (
        "You are the HOST agent in a Pulsecast analytics panel.\n"
        "Your job is to restate the user's question, clarify the business focus, and set constraints "
        "for downstream analysis.\n\n"
        "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
        'Schema:\n'
        '{\n'
        '  "primary_focus": string,\n'
        '  "time_window": string|null,\n'
        '  "region_focus": string|null,\n'
        '  "metrics": string[],\n'
        '  "notes": string|null\n'
        "}\n"
        "Rules:\n"
        "- primary_focus: 1-2 sentences summarising what decision or insight the user cares about.\n"
        "- time_window: if the question implies a period (e.g. last year, last 12 months), capture it; "
        "otherwise null.\n"
        "- region_focus: capture specific region/market mentions (e.g. Europe, North region); otherwise null.\n"
        "- metrics: list key business measures mentioned or obviously implied (e.g. sales value, volume, margin).\n"
        "- notes: optional guardrails or assumptions for the analyst.\n"
    )


def _host_user_prompt(question: str) -> str:
    return (
        "User question:\n"
        f"{json.dumps(question, ensure_ascii=False)}\n\n"
        "Analyse this question and fill the JSON schema."
    )


def _analyst_system_prompt() -> str:
    return (
        "You are the ANALYST agent in a Pulsecast analytics panel.\n"
        "Your job is to decompose the framed business question into 2-6 concrete, independently SQL-answerable "
        "sub-questions.\n\n"
        "You MUST return ONLY valid JSON (no markdown, no backticks, no extra text).\n"
        'Schema:\n'
        '{\n'
        '  "sub_questions": string[],\n'
        '  "rationale": string|null\n'
        "}\n"
        "Rules for each sub_question:\n"
        "- It MUST be answerable with a single SELECT/WITH query against a sales data warehouse.\n"
        "- Write each sub_question in plain English only.\n"
        "- DO NOT output SQL keywords, SQL snippets, CTEs, or code blocks.\n"
        "- Be explicit about the metric(s), time window, region/product filters, and whether you need a TOP N.\n"
        "- Prefer 2-6 sub_questions. Fewer is better if they fully answer the intent.\n"
        "- Avoid referencing previous answers; each sub_question stands alone.\n"
    )


def _analyst_system_prompt_strict() -> str:
    return (
        _analyst_system_prompt()
        + "\nSTRICT FAILURE CONDITION:\n"
        + "- If any sub_question contains SQL syntax, your response is invalid.\n"
        + "- Every sub_question must read like a business question a non-technical user can understand.\n"
    )


def _analyst_user_prompt(question: str, host: PlanningHostOutput) -> str:
    payload: dict[str, Any] = {
        "question": question,
        "host_planning": {
            "primary_focus": host.primary_focus,
            "time_window": host.time_window,
            "region_focus": host.region_focus,
            "metrics": host.metrics,
            "notes": host.notes,
        },
    }
    return (
        "Framed planning input JSON:\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Produce the JSON planning output according to the schema."
    )


def _message_content_str(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


async def _invoke_json_object(
    *,
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
) -> str:
    llm = build_chat_model(settings).bind(response_format={"type": "json_object"})
    resp = await llm.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
    )
    text = _message_content_str(resp.content).strip()
    if not text:
        raise ValueError("Empty model content")
    return text


async def run_host_planner(*, settings: Settings, question: str) -> PlanningHostOutput:
    try:
        content = await _invoke_json_object(
            settings=settings,
            system_prompt=_host_system_prompt(),
            user_prompt=_host_user_prompt(question),
        )
        data = json.loads(content)
        out = _HostOut.model_validate(data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Host planning failed: {e}",
        ) from e
    return PlanningHostOutput(**out.model_dump())


async def run_analyst_planner(
    *, settings: Settings, question: str, host: PlanningHostOutput
) -> PlanningAnalystOutput:
    async def _call_planner(system_prompt: str) -> _AnalystOut:
        content = await _invoke_json_object(
            settings=settings,
            system_prompt=system_prompt,
            user_prompt=_analyst_user_prompt(question, host),
        )
        data = json.loads(content)
        return _AnalystOut.model_validate(data)

    try:
        out = await _call_planner(_analyst_system_prompt())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Analyst planning failed: {e}",
        ) from e

    # Enforce plain-English sub-questions; retry once with stricter instructions if needed.
    cleaned = [_clean_sub_question(q) for q in out.sub_questions if q and _clean_sub_question(q)]
    has_sqlish = any(_looks_like_sql(q) for q in cleaned)
    if has_sqlish:
        try:
            out = await _call_planner(_analyst_system_prompt_strict())
            cleaned = [_clean_sub_question(q) for q in out.sub_questions if q and _clean_sub_question(q)]
        except Exception:
            pass

    sub_questions = [q for q in cleaned if not _looks_like_sql(q)]
    if not sub_questions:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Analyst planning produced no valid plain-English sub_questions",
        )
    if len(sub_questions) > 6:
        sub_questions = sub_questions[:6]

    return PlanningAnalystOutput(sub_questions=sub_questions, rationale=out.rationale)

