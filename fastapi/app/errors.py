from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UpstreamServiceError(Exception):
    service: str
    message: str
    upstream_status_code: int | None = None
    upstream_body: str | None = None

    def __str__(self) -> str:  # pragma: no cover
        code = "" if self.upstream_status_code is None else f" ({self.upstream_status_code})"
        return f"{self.service}{code}: {self.message}"

