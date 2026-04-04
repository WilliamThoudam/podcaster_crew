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
    #: When resuming discussion_approval_required, merge this text into the paused canonical question (same session, no replan).
    discussion_refinement: str | None = None
    stream: bool = True
    model: str = "pulsecast-qa"
    user: str | None = None


class PulsecastChatRefineRequest(BaseModel):
    """Follow-up turn on an existing in-memory Pulsecast session (same id as OpenAI `user`)."""

    session_id: str = Field(..., min_length=1)
    refinement: str = Field(..., min_length=1)
    stream: bool = True
    model: str = "pulsecast-qa"


class PulsecastStreamControlRequest(BaseModel):
    """Pause or resume outbound SSE for an active chat completion stream."""

    job_id: str = Field(..., min_length=1)
    paused: bool
    user: str | None = None


class TextToSqlResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_sql: str | None = None
    error: str | None = None


class ExecuteSqlResponse(BaseModel):
    """MCP execute_sql: only success + data are used; extra keys from the tool are ignored."""

    model_config = ConfigDict(extra="ignore")

    success: bool = False
    data: list[dict[str, Any]] = Field(default_factory=list)


DiscussionDepth = Literal["minimal", "linear", "moderated"]


class PlanningHostOutput(BaseModel):
    """High-level framing from the HOST agent used to steer Analyst planning."""

    primary_focus: str
    time_window: str | None = None
    region_focus: str | None = None
    metrics: list[str] = Field(default_factory=list)
    notes: str | None = None
    discussion_depth: DiscussionDepth = "moderated"


class PlanningAnalystOutput(BaseModel):
    """Analyst-produced decomposition: warehouse sub_questions (text-to-SQL) plus optional web_sub_questions."""

    sub_questions: list[str] = Field(
        default_factory=list,
        description="Mart-style questions executed via text-to-SQL (measures, dimensions, time).",
    )
    web_sub_questions: list[str] = Field(
        default_factory=list,
        description="Public-web intents (news, recalls, external reports); not sent to text-to-SQL.",
    )
    rationale: str | None = None


class SubResult(BaseModel):
    """One planned sub-question with its generated SQL and execution result."""

    sub_question: str
    generated_sql: str
    original_sql: str | None = None
    execute: ExecuteSqlResponse


class AgentPipelineStep(BaseModel):
    """BRD agent lane + phase for Pulsecast Agent Status / future UI sync."""

    id: Literal["host", "analyst", "marketing", "finance", "forecaster", "web_crawler", "challenger"]
    status: Literal["completed", "skipped"] = "completed"
    phase: str
    detail: str | None = None


class AgentInsight(BaseModel):
    """One chat bubble aligned with podcast roles (HOST … CHALLENGER)."""

    role: Literal["HOST", "ANALYST", "MARKETING", "FINANCE", "FORECASTER", "WEB_CRAWLER", "CHALLENGER"]
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
