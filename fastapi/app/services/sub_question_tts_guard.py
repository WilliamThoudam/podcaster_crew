"""
Reject follow-up sub-questions that are not plain analytic English for text-to-SQL.

Invalid: conversational/meta asks, pasted SQL / pseudo-SQL, obvious query syntax.
"""

from __future__ import annotations

import re

# Minimum length for a concrete analytic ask (after strip).
_MIN_LEN = 20

# Phrases that indicate assistant-directed or data-availability checks, not warehouse questions.
_META_ASK = re.compile(
    r"(?ix)"
    r"(^\s*(?:can|could|would)\s+you\b)"
    r"|(\bplease\s+confirm\b)"
    r"|(\bcan\s+you\s+confirm\b)"
    r"|(\bcould\s+you\s+confirm\b)"
    r"|(\bconfirm\s+(?:the\s+)?(?:availability|existence)\b)"
    r"|(\bconfirm\s+whether\b)"
    r"|(\bconfirm\s+if\b)"
    r"|(\bverify\s+(?:the\s+)?(?:availability|data)\b)"
    r"|(\bverify\s+whether\b)"
    r"|(\bcheck\s+if\s+(?:we|there)\b)"
    r"|(\bcheck\s+whether\s+(?:we|there)\b)"
    r"|(\bdo\s+we\s+have\s+(?:any\s+)?data\b)"
    r"|(\bis\s+there\s+(?:any\s+)?data\b)"
    r"|(\bdata\s+availability\b)"
    r"|(\bavailability\s+of\s+(?:the\s+)?(?:sales\s+)?data\b)"
    r"|(\bany\s+data\s+available\b)"
    r"|(\bis\s+data\s+available\b)"
    r"|(\bensure\s+(?:that\s+)?(?:we\s+have|there\s+is)\b)"
)

# Plain-language sub-questions only: reject pasted/generated SQL (Challenger / edits must not bypass this).
_SQL_SYNTAX = re.compile(
    r"(?is)"
    r"(?:^\s*SELECT\s+|\n\s*SELECT\s+)"
    r"|(?:\b(?:INSERT|UPDATE|DELETE|MERGE)\s+)"
    r"|(?:\b(?:INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN\b)"
    # JOIN table [alias] ON | JOIN ... USING (avoid English "join our …")
    r"|(?:\bJOIN\b\s+[A-Za-z_][A-Za-z0-9_.]*\s+(?:[A-Za-z_][A-Za-z0-9_]*\s+)?(?:ON\b|USING\b))"
    r"|(?:\b(?:GROUP|ORDER)\s+BY\b)"
    r"|(?:\bHAVING\b)"
    # join-condition pattern, not the word "on" in prose
    r"|(?:\bON\s+[A-Za-z_][A-Za-z0-9_.]*\s*\.\s*[A-Za-z_][A-Za-z0-9_.]*\s*=\s*)"
    r"|(?:\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\()"
)


def _looks_like_sql(text: str) -> bool:
    return bool(_SQL_SYNTAX.search(text or ""))


# At least one token suggesting an analytic slice (warehouse-style question).
_ANALYTIC_HINT = re.compile(
    r"(?i)\b("
    r"sales|revenue|margin|profit|volume|quantity|qty|amount|units?|count|total|sum|average|avg|mean|"
    r"by\s+\w+|per\s+\w+|group(?:ed)?\s+by|breakdown|trend|growth|decline|yoy|qoq|quarter|month|year|week|"
    r"region|country|channel|product|category|segment|customer|manufacturer|brand|sku|panel|calendar"
    r")\b",
)


def is_valid_tts_sub_question(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < _MIN_LEN:
        return False
    if _looks_like_sql(s):
        return False
    if _META_ASK.search(s):
        return False
    # Short strings that passed meta filter should still smell like analytics.
    if len(s) < 48 and not _ANALYTIC_HINT.search(s):
        return False
    return True
