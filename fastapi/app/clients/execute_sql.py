from __future__ import annotations

import httpx

from app.config import Settings
from app.errors import UpstreamServiceError
from app.models.schemas import ExecuteSqlResponse


async def execute_sql(
    *,
    settings: Settings,
    sql: str,
    user_id: int,
    user_db_id: int,
    db_type: str,
) -> ExecuteSqlResponse:
    payload = {
        "sql": sql,
        "user_id": user_id,
        "user_db_id": user_db_id,
        "db_type": db_type,
    }
    timeout = httpx.Timeout(settings.http_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.post(settings.execute_sql_url, json=payload)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as e:
            body_text = None
            try:
                body_text = e.response.text
            except Exception:
                body_text = None
            raise UpstreamServiceError(
                service="execute_sql",
                message="Upstream rejected the request",
                upstream_status_code=e.response.status_code,
                upstream_body=body_text,
            ) from e
        except httpx.RequestError as e:
            raise UpstreamServiceError(
                service="execute_sql",
                message=f"Upstream unavailable: {e}",
                upstream_status_code=None,
                upstream_body=None,
            ) from e

    return ExecuteSqlResponse.model_validate(body)
