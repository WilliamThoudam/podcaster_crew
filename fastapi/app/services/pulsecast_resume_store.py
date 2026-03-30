from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, Literal, Union

from app.models.schemas import (
    AgentPipelineStep,
    ExecuteSqlResponse,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)
from app.services.pulsecast_llm_agents import (
    DiscussionState,
    DiscussionTurn,
    _AgentOut,
)

PauseKindChallenger = Literal["challenger_followup"]
PauseKindDuplicate = Literal["duplicate_sub_question"]
PauseKindWebSearch = Literal["web_search"]

# Union of snapshots stored under a resume_token (Challenger HITL vs duplicate-SQL vs web search HITL).
PulsecastPausedSnapshotUnion = Union[
    "PulsecastPausedSnapshot",
    "DuplicateSubQuestionPausedSnapshot",
    "WebSearchPausedSnapshot",
]


@dataclass
class DuplicateSubQuestionPausedSnapshot:
    """Resume after duplicate SQL: user approves/edits analyst-proposed replacement sub-question."""

    question: str
    host_plan: PlanningHostOutput
    analyst_plan: PlanningAnalystOutput
    sub_results: list[SubResult]
    pending_index: int
    original_sub_question: str
    duplicate_sql: str
    proposed_sub_question: str
    rationale: str | None
    primary_sql: str
    primary_exe: ExecuteSqlResponse
    openai_user: str | None

    pause_kind: PauseKindDuplicate = "duplicate_sub_question"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pause_kind": "duplicate_sub_question",
            "question": self.question,
            "host_plan": self.host_plan.model_dump(mode="json"),
            "analyst_plan": self.analyst_plan.model_dump(mode="json"),
            "sub_results": [sr.model_dump(mode="json") for sr in self.sub_results],
            "pending_index": self.pending_index,
            "original_sub_question": self.original_sub_question,
            "duplicate_sql": self.duplicate_sql,
            "proposed_sub_question": self.proposed_sub_question,
            "rationale": self.rationale,
            "primary_sql": self.primary_sql,
            "primary_exe": self.primary_exe.model_dump(mode="json"),
            "openai_user": self.openai_user,
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> DuplicateSubQuestionPausedSnapshot:
        sub_results = [SubResult.model_validate(x) for x in d["sub_results"]]
        return DuplicateSubQuestionPausedSnapshot(
            question=d["question"],
            host_plan=PlanningHostOutput.model_validate(d["host_plan"]),
            analyst_plan=PlanningAnalystOutput.model_validate(d["analyst_plan"]),
            sub_results=sub_results,
            pending_index=int(d["pending_index"]),
            original_sub_question=d["original_sub_question"],
            duplicate_sql=d["duplicate_sql"],
            proposed_sub_question=d["proposed_sub_question"],
            rationale=d.get("rationale"),
            primary_sql=d["primary_sql"],
            primary_exe=ExecuteSqlResponse.model_validate(d["primary_exe"]),
            openai_user=d.get("openai_user"),
        )


@dataclass
class PulsecastPausedSnapshot:
    """Server-side state to resume after sql_approval_required (HITL)."""

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
    openai_user: str | None

    pause_kind: PauseKindChallenger = "challenger_followup"

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "pause_kind": "challenger_followup",
            "pipeline": [p.model_dump(mode="json") for p in self.pipeline],
            "analyst": self.discussion.analyst.model_dump(mode="json"),
            "discussion_turns": [t.model_dump(mode="json") for t in self.discussion.turns],
            "question": self.question,
            "generated_sql": self.generated_sql,
            "primary_exe": self.primary_exe.model_dump(mode="json"),
            "deterministic_summary": self.deterministic_summary,
            "sub_results": [sr.model_dump(mode="json") for sr in self.sub_results],
            "host_plan": self.host_plan.model_dump(mode="json"),
            "analyst_plan": self.analyst_plan.model_dump(mode="json"),
            "proposed_sub_question": self.proposed_sub_question,
            "rationale": self.rationale,
            "openai_user": self.openai_user,
        }

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> PulsecastPausedSnapshot:
        try:
            analyst = _AgentOut.model_validate(d["analyst"])
            turns_raw = d["discussion_turns"]
            turns = [DiscussionTurn.model_validate(x) for x in turns_raw]
        except KeyError as e:
            raise ValueError(f"Invalid paused snapshot (missing field: {e.args[0]})") from e
        discussion = DiscussionState(analyst=analyst, turns=turns)

        pipeline = [AgentPipelineStep.model_validate(x) for x in d["pipeline"]]
        sub_results = [SubResult.model_validate(x) for x in d["sub_results"]]

        return PulsecastPausedSnapshot(
            pipeline=pipeline,
            discussion=discussion,
            question=d["question"],
            generated_sql=d["generated_sql"],
            primary_exe=ExecuteSqlResponse.model_validate(d["primary_exe"]),
            deterministic_summary=d["deterministic_summary"],
            sub_results=sub_results,
            host_plan=PlanningHostOutput.model_validate(d["host_plan"]),
            analyst_plan=PlanningAnalystOutput.model_validate(d["analyst_plan"]),
            proposed_sub_question=d["proposed_sub_question"],
            rationale=d.get("rationale"),
            openai_user=d.get("openai_user"),
        )


@dataclass
class WebSearchPausedSnapshot:
    """Resume after web_search_approval_required (HITL).

    Two stages:
    - stage="pre": pause happens BEFORE any agent discussion starts (Option A web_search phase).
    - stage="mid": pause happens DURING agent discussion (existing WEB_CRAWLER behavior).
    """

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
    openai_user: str | None

    stage: Literal["pre", "mid"] = "mid"

    # stage="pre" has no agent discussion yet; keep optional.
    pipeline: list[AgentPipelineStep] | None = None
    discussion: DiscussionState | None = None

    pause_kind: PauseKindWebSearch = "web_search"

    def to_json_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "pause_kind": "web_search",
            "stage": self.stage,
            "question": self.question,
            "generated_sql": self.generated_sql,
            "primary_exe": self.primary_exe.model_dump(mode="json"),
            "deterministic_summary": self.deterministic_summary,
            "sub_results": [sr.model_dump(mode="json") for sr in self.sub_results],
            "host_plan": self.host_plan.model_dump(mode="json"),
            "analyst_plan": self.analyst_plan.model_dump(mode="json"),
            "proposed_search_query": self.proposed_search_query,
            "search_queries": list(self.search_queries),
            "pending_search_index": self.pending_search_index,
            "completed_web_results": list(self.completed_web_results),
            "rationale": self.rationale,
            "openai_user": self.openai_user,
        }
        if self.pipeline is not None:
            d["pipeline"] = [p.model_dump(mode="json") for p in self.pipeline]
        if self.discussion is not None:
            d["analyst"] = self.discussion.analyst.model_dump(mode="json")
            d["discussion_turns"] = [t.model_dump(mode="json") for t in self.discussion.turns]
        return d

    @staticmethod
    def from_json_dict(d: dict[str, Any]) -> WebSearchPausedSnapshot:
        stage = str(d.get("stage") or "mid").strip().lower()
        if stage not in ("pre", "mid"):
            stage = "mid"

        discussion: DiscussionState | None = None
        if "analyst" in d and "discussion_turns" in d:
            analyst = _AgentOut.model_validate(d["analyst"])
            turns = [DiscussionTurn.model_validate(x) for x in d["discussion_turns"]]
            discussion = DiscussionState(analyst=analyst, turns=turns)

        pipeline: list[AgentPipelineStep] | None = None
        if "pipeline" in d and isinstance(d.get("pipeline"), list):
            pipeline = [AgentPipelineStep.model_validate(x) for x in d["pipeline"]]

        sub_results = [SubResult.model_validate(x) for x in d["sub_results"]]
        proposed = str(d.get("proposed_search_query") or "").strip()
        raw_sqs = d.get("search_queries")
        if isinstance(raw_sqs, list) and len(raw_sqs) > 0:
            search_queries = [str(x).strip() for x in raw_sqs if str(x).strip()]
        else:
            search_queries = [proposed] if proposed else []
        pending = int(d.get("pending_search_index", 0))
        cw = d.get("completed_web_results")
        completed_web_results: list[dict[str, Any]] = cw if isinstance(cw, list) else []
        return WebSearchPausedSnapshot(
            stage=stage,  # type: ignore[arg-type]
            pipeline=pipeline,
            discussion=discussion,
            question=d["question"],
            generated_sql=d["generated_sql"],
            primary_exe=ExecuteSqlResponse.model_validate(d["primary_exe"]),
            deterministic_summary=d["deterministic_summary"],
            sub_results=sub_results,
            host_plan=PlanningHostOutput.model_validate(d["host_plan"]),
            analyst_plan=PlanningAnalystOutput.model_validate(d["analyst_plan"]),
            proposed_search_query=d["proposed_search_query"],
            search_queries=search_queries,
            pending_search_index=pending,
            completed_web_results=completed_web_results,
            rationale=d.get("rationale"),
            openai_user=d.get("openai_user"),
        )


def deserialize_paused_snapshot(d: dict[str, Any]) -> PulsecastPausedSnapshotUnion:
    kind = d.get("pause_kind")
    if kind == "duplicate_sub_question":
        return DuplicateSubQuestionPausedSnapshot.from_json_dict(d)
    if kind == "web_search":
        return WebSearchPausedSnapshot.from_json_dict(d)
    return PulsecastPausedSnapshot.from_json_dict(d)


class PulsecastResumeStore:
    """In-memory TTL map: resume_token -> serialized paused snapshot."""

    def __init__(self, ttl_seconds: float = 3600.0) -> None:
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}

    def _purge_expired(self) -> None:
        now = time.monotonic()
        dead = [k for k, (exp, _) in self._entries.items() if exp <= now]
        for k in dead:
            del self._entries[k]

    def issue_token(self, snapshot: PulsecastPausedSnapshotUnion) -> str:
        self._purge_expired()
        token = secrets.token_urlsafe(32)
        self._entries[token] = (time.monotonic() + self._ttl, snapshot.to_json_dict())
        return token

    def pop(self, token: str) -> PulsecastPausedSnapshotUnion | None:
        self._purge_expired()
        item = self._entries.pop(token, None)
        if item is None:
            return None
        exp, payload = item
        if time.monotonic() > exp:
            return None
        try:
            return deserialize_paused_snapshot(payload)
        except (KeyError, ValueError, TypeError):
            return None


resume_store = PulsecastResumeStore()
