from __future__ import annotations

"""
Shared helpers for the Pulsecast chat completion pipeline: SQL validation and execute summaries.
"""

import re
from typing import Any

from fastapi import HTTPException, status

from app.models.schemas import ExecuteSqlResponse

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
