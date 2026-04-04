from __future__ import annotations

from typing import Any, Literal

from typing_extensions import TypedDict

from app.models.schemas import (
    ExecuteSqlResponse,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)
from app.services.pulsecast_completion_types import CompletionStreamOutcome

DiscussionDepth = Literal["minimal", "linear", "moderated"]


class PulsecastState(TypedDict, total=False):
    """Unified LangGraph state for the outer Pulsecast workflow and the agents subgraph."""

    question: str
    host_plan: PlanningHostOutput
    analyst_plan: PlanningAnalystOutput
    sub_results: list[SubResult]
    primary_sql: str
    primary_exe: ExecuteSqlResponse
    completed_web_results: list[dict[str, Any]]
    user_declined_web_search: bool
    outcome: CompletionStreamOutcome

    # Agent subgraph (JSON-serializable agent outputs as model_dump dicts)
    ctx: dict[str, Any]
    deterministic_summary: str
    discussion_depth: DiscussionDepth
    max_rounds: int
    agents_needs_pre_discussion_pause: bool
    analyst_output: dict[str, Any]
    discussion_turns: list[dict[str, Any]]
    agents_prior: dict[str, dict[str, Any]]
    agents_pipeline: list[dict[str, Any]]
    current_round: int
