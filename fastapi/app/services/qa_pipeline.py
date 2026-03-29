from __future__ import annotations

"""
Non-SSE QA endpoint: text-to-SQL + execute + `run_llm_agents`.
Uses the same LangChain-backed LLM stack as the chat completion graph via `pulsecast_llm_agents`.
"""

import re
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.clients.execute_sql import execute_sql as execute_sql_client
from app.clients.text_to_sql import generate_sql
from app.config import Settings
from app.models.schemas import ExecuteSqlResponse, QARequest, QAResponse, TextToSqlResponse
from app.services.pulsecast_llm_agents import run_llm_agents

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|MERGE|TRUNCATE|ALTER|DROP|CREATE|GRANT|REVOKE|CALL|EXECUTE)\b",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_sql(sql: str) -> str:
    return sql.strip()


def validate_and_normalize_sql(sql: str) -> str:
    if not sql or not sql.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Generated SQL is empty",
        )
    s = _normalize_sql(sql)
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) != 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Exactly one SQL statement is required (no multiple statements)",
        )
    single = parts[0]
    if _FORBIDDEN.search(single):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only read-only SELECT-style queries are allowed",
        )
    head = single.lstrip(" \t\n\r\f\v(").upper()
    if not head.startswith("SELECT") and not head.startswith("WITH"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="SQL must start with SELECT or WITH (CTE)",
        )
    return single


def _format_value(v: Any, max_len: int = 80) -> str:
    s = str(v) if v is not None else "NULL"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def build_answer_summary(exec_result: ExecuteSqlResponse, sample_rows: int = 5) -> str:
    """Summarise using only the execute_sql `data` array (row count = len(data))."""
    parts: list[str] = []
    data = exec_result.data or []
    rc = len(data)
    parts.append(f"Returned {rc} row(s); evidence is only the rows in `data`.")

    if data:
        parts.append("Sample rows from data:")
        col_keys = list(data[0].keys())
        for row in data[:sample_rows]:
            cells = ", ".join(f"{k}={_format_value(row.get(k))}" for k in col_keys[:6])
            if len(col_keys) > 6:
                cells += ", …"
            parts.append(f"  • {cells}")

    return "\n".join(parts)


async def run_qa(settings: Settings, req: QARequest) -> QAResponse:
    user_id = req.user_id if req.user_id is not None else settings.default_user_id
    user_db_id = req.user_db_id if req.user_db_id is not None else settings.default_user_db_id
    db_type = req.db_type or settings.default_db_type
    schema_name = req.schema_name or settings.default_schema_name
    model = req.model or settings.default_model
    max_nodes = req.max_nodes or settings.default_max_nodes

    tts: TextToSqlResponse
    try:
        tts = await generate_sql(
            settings=settings,
            question=req.question.strip(),
            user_id=user_id,
            user_db_id=user_db_id,
            db_type=db_type,
            schema_name=schema_name,
            session_id=req.session_id,
            model=model,
            max_nodes=max_nodes,
            is_retry=req.is_retry,
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Text-to-SQL service error: {e}",
        ) from e

    if tts.error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"text_to_sql_error": tts.error, "generated_sql": tts.generated_sql},
        )

    sql = validate_and_normalize_sql((tts.generated_sql or "").strip())

    try:
        exe = await execute_sql_client(
            settings=settings,
            sql=sql,
            user_id=user_id,
            user_db_id=user_db_id,
            db_type=db_type,
        )
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Execute SQL service error: {e}",
        ) from e

    if not exe.success:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"generated_sql": sql, "execute": exe.model_dump()},
        )

    deterministic = build_answer_summary(exe)
    agents_out = await run_llm_agents(
        settings=settings,
        question=req.question.strip(),
        generated_sql=sql,
        exe=exe,
        deterministic_summary=deterministic,
        allow_sql_approval_pause=False,
    )
    answer = agents_out.answer
    pipeline = agents_out.pipeline
    agent_messages = agents_out.messages
    return QAResponse(
        generated_sql=sql,
        answer=answer,
        execute=exe,
        text_to_sql_error=None,
        pipeline=pipeline,
        agent_messages=agent_messages,
    )
