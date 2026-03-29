"""
Reject Challenger follow-up strings that are conversational/meta for text-to-SQL (not chat).
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
    if _META_ASK.search(s):
        return False
    # Short strings that passed meta filter should still smell like analytics.
    if len(s) < 48 and not _ANALYTIC_HINT.search(s):
        return False
    return True
