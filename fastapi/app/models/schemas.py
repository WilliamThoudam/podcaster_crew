from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural language question")
    user_id: int | None = None
    user_db_id: int | None = None
    db_type: str | None = None
    schema_name: str | None = None
    session_id: str | None = None
    model: str | None = None
    max_nodes: str | None = None
    is_retry: bool = False


class TextToSqlResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_sql: str | None = None
    error: str | None = None


class ExecuteSqlField(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    type: str | None = None
    scale: int | None = None
    nullable: bool | None = None


class ExecuteSqlResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    success: bool = False
    data: list[dict[str, Any]] = Field(default_factory=list)
    rowCount: int | None = None
    fields: list[ExecuteSqlField] = Field(default_factory=list)
    executionTime: int | None = None
    query: str | None = None
    originalQuery: str | None = None
    limited: bool | None = None
    maxRecords: int | None = None
    note: str | None = None


class AgentPipelineStep(BaseModel):
    """BRD agent lane + phase for Pulsecast Agent Status / future UI sync."""

    id: Literal["host", "analyst", "marketing", "finance", "challenger"]
    status: Literal["completed", "skipped"] = "completed"
    phase: str
    detail: str | None = None


class AgentInsight(BaseModel):
    """One chat bubble aligned with podcast roles (HOST … CHALLENGER)."""

    role: Literal["HOST", "ANALYST", "MARKETING", "FINANCE", "CHALLENGER"]
    text: str


class QAResponse(BaseModel):
    generated_sql: str
    answer: str
    execute: ExecuteSqlResponse
    text_to_sql_error: str | None = None
    pipeline: list[AgentPipelineStep] = Field(
        default_factory=list,
        description="Ordered steps matching Host → Analyst → Marketing → Finance → Challenger.",
    )
    agent_messages: list[AgentInsight] = Field(
        default_factory=list,
        description="Multi-agent narration; Analyst entry mirrors `answer` for SQL-led turns.",
    )
