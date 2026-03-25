from __future__ import annotations

import httpx

from app.config import Settings
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
        resp = await client.post(settings.execute_sql_url, json=payload)
        resp.raise_for_status()
        body = resp.json()

    return ExecuteSqlResponse.model_validate(body)
