from __future__ import annotations

import httpx

from app.config import Settings
from app.errors import UpstreamServiceError
from app.models.schemas import TextToSqlResponse


async def generate_sql(
    *,
    settings: Settings,
    question: str,
    messages: list[dict[str, str]] | None = None,
    user_id: int,
    user_db_id: int,
    db_type: str,
    schema_name: str,
    session_id: str | None,
    model: str,
    max_nodes: str,
    is_retry: bool,
    trace_context: dict | None = None,
) -> TextToSqlResponse:
    url = settings.text_to_sql_base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict = {
        "messages": messages or [{"role": "user", "content": question}],
        "max_nodes": max_nodes,
        "stream": False,
        "user_db_id": user_db_id,
        "user_id": user_id,
        "db_type": db_type,
        "schema_name": schema_name,
        "model": model,
        "is_retry": is_retry,
    }
    if trace_context:
        payload["trace_context"] = trace_context
    if session_id:
        payload["session_id"] = session_id

    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as e:
            body_text = None
            try:
                body_text = e.response.text
            except Exception:
                body_text = None
            raise UpstreamServiceError(
                service="text_to_sql",
                message="Upstream rejected the request",
                upstream_status_code=e.response.status_code,
                upstream_body=body_text,
            ) from e
        except httpx.RequestError as e:
            raise UpstreamServiceError(
                service="text_to_sql",
                message=f"Upstream unavailable: {e}",
                upstream_status_code=None,
                upstream_body=None,
            ) from e

    # Support both top-level keys and nested shapes if upstream changes
    if "generated_sql" in body or "error" in body:
        return TextToSqlResponse.model_validate(body)

    if "choices" in body and body["choices"]:
        # OpenAI-style fallback (not expected for this service)
        content = body["choices"][0].get("message", {}).get("content", "")
        return TextToSqlResponse(generated_sql=content or None, error=None)

    return TextToSqlResponse(generated_sql=None, error=str(body)[:500])
