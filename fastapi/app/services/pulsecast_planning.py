from __future__ import annotations

import json
import re
from typing import Any, Literal

from fastapi import HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm.chat_model import build_chat_model
from app.models.schemas import PlanningAnalystOutput, PlanningHostOutput, SubResult
from app.prompts.duplicate_sub_question import (
    duplicate_sub_question_strict_suffix,
    duplicate_sub_question_system_prompt,
)
from app.prompts.planning import (
    planning_analyst_system_prompt,
    planning_analyst_system_prompt_strict,
    planning_host_system_prompt,
)
from app.services.sub_question_tts_guard import (
    is_valid_tts_sub_question,
    looks_like_non_warehouse_sub_question,
)


class _HostOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    primary_focus: str = Field(..., min_length=1)
    time_window: str | None = None
    region_focus: str | None = None
    metrics: list[str] = Field(default_factory=list)
    notes: str | None = None
    discussion_depth: Literal["minimal", "linear", "moderated"] = "moderated"


class _AnalystOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub_questions: list[str] = Field(default_factory=list)
    web_sub_questions: list[str] = Field(default_factory=list)
    rationale: str | None = None


class _DuplicateRephraseOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    replacement_sub_question: str = Field(..., min_length=1)
    rationale: str = Field(..., min_length=1)


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


def _clamp_int(v: Any, *, default: int, min_value: int, max_value: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        n = default
    if n < min_value:
        return min_value
    if n > max_value:
        return max_value
    return n


def _dedupe_preserve_order(strings: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in strings:
        t = (x or "").strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def _host_user_prompt(question: str) -> str:
    return (
        "User question:\n"
        f"{json.dumps(question, ensure_ascii=False)}\n\n"
        "Analyse this question and fill the JSON schema."
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
            "discussion_depth": host.discussion_depth,
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
            system_prompt=planning_host_system_prompt(),
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
        out = await _call_planner(planning_analyst_system_prompt())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Analyst planning failed: {e}",
        ) from e

    # Enforce plain-English; retry once with stricter instructions if needed.
    cleaned_sql = [_clean_sub_question(q) for q in out.sub_questions if q and _clean_sub_question(q)]
    cleaned_web = [_clean_sub_question(q) for q in (out.web_sub_questions or []) if q and _clean_sub_question(q)]
    has_sqlish = any(_looks_like_sql(q) for q in cleaned_sql + cleaned_web)
    if has_sqlish:
        try:
            out = await _call_planner(planning_analyst_system_prompt_strict())
            cleaned_sql = [_clean_sub_question(q) for q in out.sub_questions if q and _clean_sub_question(q)]
            cleaned_web = [_clean_sub_question(q) for q in (out.web_sub_questions or []) if q and _clean_sub_question(q)]
        except Exception:
            pass

    sql_pass: list[str] = []
    moved_to_web: list[str] = []
    for q in cleaned_sql:
        if _looks_like_sql(q):
            continue
        if looks_like_non_warehouse_sub_question(q):
            moved_to_web.append(q)
        else:
            sql_pass.append(q)

    web_all = _dedupe_preserve_order(moved_to_web + cleaned_web)
    sql_pass = _dedupe_preserve_order([q for q in sql_pass if not _looks_like_sql(q)])
    sql_keys = {s.lower() for s in sql_pass}
    web_filtered = [w for w in web_all if w.lower() not in sql_keys]

    if not sql_pass:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Analyst planning produced no warehouse sub_questions; at least one SQL-answerable step is required."
            ),
        )

    max_sql = _clamp_int(settings.pulsecast_max_plan_sql, default=4, min_value=1, max_value=12)
    max_web = _clamp_int(settings.pulsecast_max_plan_web, default=2, min_value=0, max_value=6)

    return PlanningAnalystOutput(
        sub_questions=sql_pass[:max_sql],
        web_sub_questions=web_filtered[:max_web],
        rationale=out.rationale,
    )


def _duplicate_rephrase_user_payload(
    *,
    question: str,
    host: PlanningHostOutput,
    analyst_plan: PlanningAnalystOutput,
    sub_results: list[SubResult],
    pending_index: int,
    original_sub_question: str,
    duplicate_sql: str,
) -> str:
    prior = [
        {
            "index": i + 1,
            "sub_question": sr.sub_question,
            "generated_sql": sr.generated_sql,
        }
        for i, sr in enumerate(sub_results)
    ]
    payload: dict[str, Any] = {
        "user_question": question,
        "host_planning": {
            "primary_focus": host.primary_focus,
            "time_window": host.time_window,
            "region_focus": host.region_focus,
            "metrics": host.metrics,
            "notes": host.notes,
            "discussion_depth": host.discussion_depth,
        },
        "planned_sub_questions": list(analyst_plan.sub_questions),
        "planned_web_sub_questions": list(analyst_plan.web_sub_questions),
        "prior_sub_question_results": prior,
        "pending_step_index_1_based": pending_index + 1,
        "current_sub_question": original_sub_question,
        "duplicate_sql": duplicate_sql,
    }
    return json.dumps(payload, ensure_ascii=False)


async def run_duplicate_sub_question_rephrase(
    *,
    settings: Settings,
    question: str,
    host_plan: PlanningHostOutput,
    analyst_plan: PlanningAnalystOutput,
    sub_results: list[SubResult],
    pending_index: int,
    original_sub_question: str,
    duplicate_sql: str,
) -> tuple[str, str]:
    """Returns (proposed_sub_question, rationale) for HITL; TTS-valid or falls back to original."""
    base_user = _duplicate_rephrase_user_payload(
        question=question,
        host=host_plan,
        analyst_plan=analyst_plan,
        sub_results=sub_results,
        pending_index=pending_index,
        original_sub_question=original_sub_question,
        duplicate_sql=duplicate_sql,
    )

    async def _call(*, strict_extra: str) -> _DuplicateRephraseOut:
        content = await _invoke_json_object(
            settings=settings,
            system_prompt=duplicate_sub_question_system_prompt(),
            user_prompt=base_user + strict_extra,
        )
        data = json.loads(content)
        return _DuplicateRephraseOut.model_validate(data)

    try:
        out = await _call(strict_extra="")
    except Exception as first_e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Duplicate sub-question rephrase failed: {first_e}",
        ) from first_e

    prop = out.replacement_sub_question.strip()
    rat = out.rationale.strip()
    if is_valid_tts_sub_question(prop):
        return prop, rat

    try:
        out2 = await _call(strict_extra=duplicate_sub_question_strict_suffix())
        prop2 = out2.replacement_sub_question.strip()
        rat2 = out2.rationale.strip()
        if is_valid_tts_sub_question(prop2):
            return prop2, rat2
    except Exception:
        pass

    fallback_rationale = (
        f"{rat} The suggested wording may need editing for the query engine — "
        "please approve or edit the question below."
    )
    return original_sub_question.strip(), fallback_rationale

