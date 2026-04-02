from __future__ import annotations

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import Settings
from app.errors import UpstreamServiceError


class MergeBiQueryResponse(BaseModel):
    """Conversational BI merge service: single canonical question string."""

    model_config = ConfigDict(extra="ignore")

    canonical_query: str


async def merge_conversational_bi_query(*, settings: Settings, queries: list[str]) -> str:
    payload = {"queries": queries}
    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(settings.merge_bi_query_url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as e:
            body_text = None
            try:
                body_text = e.response.text
            except Exception:
                body_text = None
            raise UpstreamServiceError(
                service="merge_bi_query",
                message="Upstream rejected the request",
                upstream_status_code=e.response.status_code,
                upstream_body=body_text,
            ) from e
        except httpx.RequestError as e:
            raise UpstreamServiceError(
                service="merge_bi_query",
                message=f"Upstream unavailable: {e}",
                upstream_status_code=None,
                upstream_body=None,
            ) from e

    try:
        parsed = MergeBiQueryResponse.model_validate(body)
    except ValidationError as e:
        raise UpstreamServiceError(
            service="merge_bi_query",
            message=f"Invalid merge response: {e}",
            upstream_status_code=resp.status_code,
            upstream_body=None,
        ) from e

    canonical = parsed.canonical_query.strip()
    if not canonical:
        raise UpstreamServiceError(
            service="merge_bi_query",
            message="Merge service returned an empty canonical_query",
            upstream_status_code=resp.status_code,
            upstream_body=None,
        )

    return canonical
