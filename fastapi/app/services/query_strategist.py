"""
Aggregation Agent (Query Strategist).

Sits between text_to_sql and execute_sql. Decides whether to pass the generated
SQL through unchanged or rewrite it into a compact aggregated form that
downstream LLM agents can reason over effectively.

Design invariants:
- Strong pass-through bias: when in doubt, do not rewrite.
- Never block the pipeline: any failure falls back to the original SQL.
- Rewritten SQL is re-validated through validate_and_normalize_sql.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.llm.chat_model import build_chat_model
from app.prompts.llm_agents import (
    AGGREGATION_AGENT_SYSTEM,
    aggregation_agent_user_prompt,
)
from app.services.qa_pipeline import validate_and_normalize_sql

logger = logging.getLogger(__name__)

_AGG_FUNCS = re.compile(
    r"\b(SUM|AVG|COUNT|STDDEV|STDDEV_POP|STDDEV_SAMP|VARIANCE|VAR_POP|VAR_SAMP|MIN|MAX|MEDIAN)\s*\(",
    re.IGNORECASE,
)
_LIMIT_RE = re.compile(r"\bLIMIT\s+(\d+)", re.IGNORECASE)
_GROUP_BY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_TREND_KEYWORDS = re.compile(
    r"\b(trend|temporal|over\s+time|month[- ]over[- ]month|MoM|YoY|volatil|seasonal|fluctuat|growth|decline)\b",
    re.IGNORECASE,
)
_JSON_BLOCK = re.compile(r"\{[\s\S]*\}", re.MULTILINE)


class AggregationDecision(BaseModel):
    """Structured output from the Aggregation Agent LLM call."""

    model_config = ConfigDict(extra="ignore")

    action: Literal["pass_through", "rewrite"] = "pass_through"
    sql: str
    reason: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    risk_flags: list[str] = Field(default_factory=list)


@dataclass
class QueryStrategyResult:
    """Return value of apply_query_strategy — consumed by run_sub_questions_slice."""

    sql: str
    was_rewritten: bool
    original_sql: str
    decision: AggregationDecision | None = None


def _count_group_by_columns(sql: str) -> int:
    """Rough estimate of the number of GROUP BY columns in the outermost clause."""
    upper = sql.upper()
    pos = upper.rfind("GROUP BY")
    if pos == -1:
        return 0
    after = sql[pos + 8 :]
    # Stop at ORDER BY / LIMIT / HAVING / end-of-string / closing paren at depth 0
    end_markers = re.compile(r"\b(ORDER\s+BY|LIMIT|HAVING|UNION|INTERSECT|EXCEPT)\b", re.IGNORECASE)
    m = end_markers.search(after)
    segment = after[: m.start()] if m else after
    depth = 0
    cols = 1 if segment.strip() else 0
    for ch in segment:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                break
        elif ch == "," and depth == 0:
            cols += 1
    return cols


def should_review_query(sql: str, sub_question: str) -> bool:
    """
    Pure-Python heuristic gate. Returns True when the query MIGHT produce an
    explosion of rows and warrants an LLM review. Biased toward False (skip).
    """
    limit_match = _LIMIT_RE.search(sql)
    if limit_match:
        limit_val = int(limit_match.group(1))
        if limit_val <= 1000:
            return False

    has_agg = bool(_AGG_FUNCS.search(sql))
    has_group = bool(_GROUP_BY_RE.search(sql))

    # Scalar aggregation (e.g. SELECT COUNT(*) FROM ...) — safe.
    if has_agg and not has_group:
        return False

    # High-cardinality GROUP BY without LIMIT
    if has_group and not limit_match:
        n_cols = _count_group_by_columns(sql)
        if n_cols >= 3:
            return True

    # No aggregation and no LIMIT — raw row dump
    if not has_agg and not limit_match:
        return True

    # Intent signals trend / temporal analysis
    if _TREND_KEYWORDS.search(sub_question):
        if not limit_match:
            return True

    return False


async def run_aggregation_agent(
    *,
    settings: Settings,
    original_sql: str,
    sub_question: str,
    user_intent: str,
) -> AggregationDecision:
    """
    Single LLM call to decide pass-through vs rewrite.
    On any failure returns a pass-through fallback.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    user_content = aggregation_agent_user_prompt(
        original_sql=original_sql,
        sub_question=sub_question,
        user_intent=user_intent,
    )

    try:
        llm = build_chat_model(settings).bind(response_format={"type": "json_object"})
        resp = await llm.ainvoke([
            SystemMessage(content=AGGREGATION_AGENT_SYSTEM),
            HumanMessage(content=user_content),
        ])
        text = resp.content
        if isinstance(text, list):
            text = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in text
            )
        text = (text or "").strip()
        if not text:
            raise ValueError("Empty LLM response")

        m = _JSON_BLOCK.search(text)
        if not m:
            raise ValueError("No JSON object found in LLM response")

        obj = json.loads(m.group(0))
        decision = AggregationDecision.model_validate(obj)
        return decision

    except Exception:
        logger.warning(
            "Aggregation Agent LLM call failed; falling back to pass-through",
            exc_info=True,
        )
        return AggregationDecision(
            action="pass_through",
            sql=original_sql,
            reason="LLM call failed — pass-through fallback",
            confidence=0.0,
            risk_flags=[],
        )


async def apply_query_strategy(
    *,
    settings: Settings,
    original_sql: str,
    sub_question: str,
    user_intent: str,
) -> QueryStrategyResult:
    """
    Orchestrator: heuristic gate -> optional LLM call -> validate rewrite -> fallback.
    """
    if not should_review_query(original_sql, sub_question):
        return QueryStrategyResult(
            sql=original_sql,
            was_rewritten=False,
            original_sql=original_sql,
            decision=None,
        )

    decision = await run_aggregation_agent(
        settings=settings,
        original_sql=original_sql,
        sub_question=sub_question,
        user_intent=user_intent,
    )

    if decision.action == "pass_through":
        return QueryStrategyResult(
            sql=original_sql,
            was_rewritten=False,
            original_sql=original_sql,
            decision=decision,
        )

    if decision.confidence < settings.query_strategy_confidence_threshold:
        logger.info(
            "Aggregation Agent confidence %.2f below threshold %.2f; pass-through",
            decision.confidence,
            settings.query_strategy_confidence_threshold,
        )
        return QueryStrategyResult(
            sql=original_sql,
            was_rewritten=False,
            original_sql=original_sql,
            decision=decision,
        )

    # Validate the rewritten SQL
    try:
        validated = validate_and_normalize_sql(decision.sql)
    except Exception:
        logger.warning(
            "Aggregation Agent rewrite failed validation; falling back to original SQL",
            exc_info=True,
        )
        return QueryStrategyResult(
            sql=original_sql,
            was_rewritten=False,
            original_sql=original_sql,
            decision=decision,
        )

    return QueryStrategyResult(
        sql=validated,
        was_rewritten=True,
        original_sql=original_sql,
        decision=decision,
    )


async def force_rewrite_after_execution(
    *,
    settings: Settings,
    original_sql: str,
    sub_question: str,
    user_intent: str,
    row_count: int,
) -> QueryStrategyResult:
    """
    Post-execution forced rewrite when the result exceeded the row threshold
    and the query was NOT already rewritten.
    """
    logger.info(
        "Post-execution rowcount guard triggered: %d rows exceed threshold %d",
        row_count,
        settings.query_strategy_row_threshold,
    )
    decision = await run_aggregation_agent(
        settings=settings,
        original_sql=original_sql,
        sub_question=sub_question,
        user_intent=user_intent,
    )

    if decision.action != "rewrite":
        logger.info("Forced rewrite: LLM still chose pass-through; continuing with original")
        return QueryStrategyResult(
            sql=original_sql,
            was_rewritten=False,
            original_sql=original_sql,
            decision=decision,
        )

    try:
        validated = validate_and_normalize_sql(decision.sql)
    except Exception:
        logger.warning(
            "Forced rewrite failed validation; continuing with original large result",
            exc_info=True,
        )
        return QueryStrategyResult(
            sql=original_sql,
            was_rewritten=False,
            original_sql=original_sql,
            decision=decision,
        )

    return QueryStrategyResult(
        sql=validated,
        was_rewritten=True,
        original_sql=original_sql,
        decision=decision,
    )
