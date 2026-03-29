from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(frozen=True)
class CompletionStreamComplete:
    content: str


@dataclass(frozen=True)
class CompletionStreamPaused:
    resume_token: str
    proposed_sub_question: str
    rationale: str | None


CompletionStreamOutcome = Union[CompletionStreamComplete, CompletionStreamPaused]
