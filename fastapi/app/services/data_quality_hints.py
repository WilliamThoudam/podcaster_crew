"""
Heuristic quality hints for executed sub_question results (Pulsecast-only).

Reads row JSON already returned by execute_sql — does not call text-to-SQL or modify SQL.
"""

from __future__ import annotations

import math
from typing import Any

from app.models.schemas import SubResult

_MAX_ROWS_SCAN = 500
_MAGNITUDE_THRESHOLD = 1e14
_MAX_COLUMNS = 48


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _abs_numeric(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return float(abs(v))
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return abs(v)
    return None


def sub_result_quality_hints(sub_results: list[SubResult]) -> list[dict[str, Any]]:
    """Per sub_result: null/empty rates and suspicious numeric magnitudes over a capped row scan."""
    out: list[dict[str, Any]] = []
    for idx, sr in enumerate(sub_results):
        rows = sr.execute.data or []
        scan = rows[:_MAX_ROWS_SCAN]
        n = len(scan)
        if n == 0:
            out.append(
                {
                    "sub_question_index": idx,
                    "sub_question": (sr.sub_question or "")[:240],
                    "rows_scanned": 0,
                    "column_flags": [],
                    "overall_flags": ["empty_result"],
                }
            )
            continue

        keys: set[str] = set()
        for r in scan[: min(50, n)]:
            keys.update(r.keys())
        col_list = sorted(keys)[:_MAX_COLUMNS]

        column_flags: list[dict[str, Any]] = []
        overall_flags: list[str] = []
        for col in col_list:
            empty = sum(1 for r in scan if _is_empty(r.get(col)))
            null_frac = empty / n
            max_mag = 0.0
            for r in scan:
                a = _abs_numeric(r.get(col))
                if a is not None and a > max_mag:
                    max_mag = a
            susp_mag = max_mag >= _MAGNITUDE_THRESHOLD
            flags: list[str] = []
            if null_frac > 0.5:
                flags.append("high_null_or_empty_rate")
            if susp_mag:
                flags.append("suspicious_magnitude")
            column_flags.append(
                {
                    "column": col,
                    "null_or_empty_fraction": round(null_frac, 4),
                    "max_abs_numeric_sample": max_mag if max_mag > 0 else None,
                    "suspicious_magnitude": susp_mag,
                    "flags": flags,
                }
            )
            if null_frac > 0.5:
                overall_flags.append(f"high_null_rate:{col}")
            if susp_mag:
                overall_flags.append(f"suspicious_magnitude:{col}")

        overall_flags = list(dict.fromkeys(overall_flags))
        out.append(
            {
                "sub_question_index": idx,
                "sub_question": (sr.sub_question or "")[:240],
                "rows_scanned": n,
                "column_flags": column_flags,
                "overall_flags": overall_flags,
            }
        )
    return out
