from __future__ import annotations

import re
from app.models.schemas import AgentInsight, AgentPipelineStep, ExecuteSqlResponse

_FINANCE_FIELDS = re.compile(
    r"(revenue|sales|amount|margin|cost|price|value|turnover|profit|qty|quantity|volume)",
    re.IGNORECASE,
)
_MARKETING_Q = re.compile(r"(campaign|marketing|promo|brand|advert|channel|acquisition)", re.IGNORECASE)


def _row_count(exe: ExecuteSqlResponse) -> int:
    if exe.rowCount is not None:
        return int(exe.rowCount)
    return len(exe.data)


def _field_names(exe: ExecuteSqlResponse) -> str:
    return " ".join(f.name for f in exe.fields)


def build_pulsecast_payload(
    question: str,
    sql: str,
    exe: ExecuteSqlResponse,
    analyst_summary: str,
) -> tuple[list[AgentPipelineStep], list[AgentInsight]]:
    """
    BRD-aligned multi-agent framing for Pulsecast: same five roles as the UI
    (Host, Analyst, Marketing, Finance, Challenger). Copy is deterministic (no extra LLM).
    """
    rc = _row_count(exe)
    fields_blob = _field_names(exe)
    q_short = question.strip()
    if len(q_short) > 160:
        q_short = q_short[:157] + "…"

    pipeline: list[AgentPipelineStep] = [
        AgentPipelineStep(
            id="host",
            status="completed",
            phase="intent",
            detail="Captured the question and routing to data agents",
        ),
        AgentPipelineStep(
            id="analyst",
            status="completed",
            phase="text-to-sql & execution",
            detail="Generated SQL and retrieved rows",
        ),
        AgentPipelineStep(
            id="marketing",
            status="completed",
            phase="go-to-market lens",
            detail="Context on demand/campaign signals when relevant",
        ),
        AgentPipelineStep(
            id="finance",
            status="completed",
            phase="financial lens",
            detail="Numeric / reporting guardrails",
        ),
        AgentPipelineStep(
            id="challenger",
            status="completed",
            phase="validation",
            detail="Sample size and query-limit checks",
        ),
    ]

    host_text = (
        f"Here's what we're analyzing: {q_short}\n"
        "I'll hand this to the analyst team to pull live numbers from your database."
    )

    marketing_text = (
        "If these results tie to campaigns, regions, or promos, compare timings and "
        "channel mix before changing spend — the SQL slice is only what we just queried."
        if _MARKETING_Q.search(question)
        else "From a go-to-market angle, pair these metrics with campaign calendars and "
        "segment splits when you need a causal story—not only top-line aggregates."
    )

    if _FINANCE_FIELDS.search(fields_blob) or _FINANCE_FIELDS.search(sql):
        finance_text = (
            f"The result set has {rc} row(s) with numeric-looking fields. "
            "Use your finance definitions (GAAP/internal) before booking numbers; "
            "this answer reflects the query as run, not official close."
        )
    else:
        finance_text = (
            "No obvious revenue/margin columns jumped out in this result shape—"
            "if the business question is financial, consider refining the question or schema context."
        )

    if exe.limited:
        challenger_text = (
            "Heads-up: the warehouse capped rows (limited run). "
            "Confirm totals with an export or an explicit LIMIT that matches your governance rules."
        )
    elif rc <= 3:
        challenger_text = (
            f"Only {rc} row(s) came back—useful for a spot check; "
            "validate against a broader filter if this drives a big decision."
        )
    else:
        challenger_text = (
            f"{rc} rows returned. Sanity-check a few cells against source if this feeds executive action."
        )

    messages: list[AgentInsight] = [
        AgentInsight(role="HOST", text=host_text),
        AgentInsight(role="ANALYST", text=analyst_summary),
        AgentInsight(role="MARKETING", text=marketing_text),
        AgentInsight(role="FINANCE", text=finance_text),
        AgentInsight(role="CHALLENGER", text=challenger_text),
    ]

    return pipeline, messages
