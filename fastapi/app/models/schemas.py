from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenAIChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAIResponseFormat(BaseModel):
    type: Literal["json_object", "text"] = "text"


class OpenAIChatCompletionRequest(BaseModel):
    model: str = "pulsecast-qa"
    messages: list[OpenAIChatMessage] = Field(default_factory=list, min_length=1)
    stream: bool = False
    temperature: float | None = None
    response_format: OpenAIResponseFormat | None = None

    # Optional compatibility fields (accepted, currently not used)
    max_tokens: int | None = None
    n: int | None = None
    user: str | None = None


class PulsecastChatResumeRequest(BaseModel):
    """Resume a paused stream after sql_approval_required (HITL)."""

    resume_token: str = Field(..., min_length=1)
    approved: bool
    edited_question: str | None = None
    stream: bool = True
    model: str = "pulsecast-qa"


class TextToSqlResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_sql: str | None = None
    error: str | None = None


class ExecuteSqlResponse(BaseModel):
    """MCP execute_sql: only success + data are used; extra keys from the tool are ignored."""

    model_config = ConfigDict(extra="ignore")

    success: bool = False
    data: list[dict[str, Any]] = Field(default_factory=list)


class PlanningHostOutput(BaseModel):
    """High-level framing from the HOST agent used to steer Analyst planning."""

    primary_focus: str
    time_window: str | None = None
    region_focus: str | None = None
    metrics: list[str] = Field(default_factory=list)
    notes: str | None = None


class PlanningAnalystOutput(BaseModel):
    """Analyst-produced decomposition into independently SQL-answerable sub-questions."""

    sub_questions: list[str] = Field(default_factory=list)
    rationale: str | None = None


class SubResult(BaseModel):
    """One planned sub-question with its generated SQL and execution result."""

    sub_question: str
    generated_sql: str
    execute: ExecuteSqlResponse


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


class OpenAIChatCompletionChoice(BaseModel):
    index: int = 0
    message: OpenAIChatMessage
    finish_reason: str | None = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)


class OpenAIChatCompletionDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class OpenAIChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: OpenAIChatCompletionDelta
    finish_reason: str | None = None


class OpenAIChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[OpenAIChatCompletionChunkChoice]


class OpenAIModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 0
    owned_by: str = "pulsecast"


class OpenAIModelsListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[OpenAIModelCard]
