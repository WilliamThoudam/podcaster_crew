from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any

from app.models.schemas import (
    AgentPipelineStep,
    ExecuteSqlResponse,
    PlanningAnalystOutput,
    PlanningHostOutput,
    SubResult,
)
from app.services.pulsecast_llm_agents import _AgentOut


@dataclass
class PulsecastPausedSnapshot:
    """Server-side state to resume after sql_approval_required (HITL)."""

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
    openai_user: str | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "v": 1,
            "pipeline": [p.model_dump(mode="json") for p in self.pipeline],
            "prior": {k: v.model_dump(mode="json") for k, v in self.prior.items()},
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
        if d.get("v") != 1:
            raise ValueError("Unsupported paused snapshot version")
        pipeline = [AgentPipelineStep.model_validate(x) for x in d["pipeline"]]
        prior = {k: _AgentOut.model_validate(v) for k, v in d["prior"].items()}
        sub_results = [SubResult.model_validate(x) for x in d["sub_results"]]
        return PulsecastPausedSnapshot(
            pipeline=pipeline,
            prior=prior,
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

    def issue_token(self, snapshot: PulsecastPausedSnapshot) -> str:
        self._purge_expired()
        token = secrets.token_urlsafe(32)
        self._entries[token] = (time.monotonic() + self._ttl, snapshot.to_json_dict())
        return token

    def pop(self, token: str) -> PulsecastPausedSnapshot | None:
        self._purge_expired()
        item = self._entries.pop(token, None)
        if item is None:
            return None
        exp, payload = item
        if time.monotonic() > exp:
            return None
        try:
            return PulsecastPausedSnapshot.from_json_dict(payload)
        except (KeyError, ValueError, TypeError):
            return None


resume_store = PulsecastResumeStore()
