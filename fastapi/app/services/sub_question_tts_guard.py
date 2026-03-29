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

# Gaps that are not answerable from a typical mart (news, recalls-as-media, comms, PR). Second-line guard
# after model prompt rules — keeps Challenger/HITL from opening SQL approval for faux warehouse asks.
_NON_WAREHOUSE_ASK = re.compile(
    r"(?ix)"
    r"(\bnews\s+(?:article|articles|coverage|story|stories|headline|headlines)\b)"
    r"|(\bmedia\s+coverage\b)"
    r"|(\bpress\s+(?:release|releases|coverage)\b)"
    r"|(\bpublic\s+(?:announcement|statement)\b)"
    r"|(\bproduct\s+recall\b|\bsafety\s+recall\b|\brecall\s+(?:notice|announcement|campaign)\b)"
    r"|(\b(?:fda|nhtsa|cpsc)\s+recall\b)"
    r"|(\bregulatory\s+(?:filing|announcement|action)\b)"
    r"|(\bcompetitor\s+(?:news|announcement|press)\b)"
    r"|(\binternal\s+comm(?:s|unications)?\b)"
    r"|(\bslack\b|\bemail\s+thread\b|\bteams\s+message\b)"
    r"|(\bpr\s+event\b|\bpress\s+event\b)"
    r"|(\bweb\s+search\s+results?\b)"
    r"|(\b(?:fetch|list|pull)\s+(?:the\s+)?(?:latest\s+)?(?:news|articles|press)\b)"
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


def looks_like_non_warehouse_sub_question(text: str) -> bool:
    """True if the text reads like news/comms/PR/recall asks, not a warehouse mart query."""
    return bool(_NON_WAREHOUSE_ASK.search(text or ""))


def is_valid_tts_sub_question(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < _MIN_LEN:
        return False
    if _looks_like_sql(s):
        return False
    if _META_ASK.search(s):
        return False
    if looks_like_non_warehouse_sub_question(s):
        return False
    # Short strings that passed meta filter should still smell like analytics.
    if len(s) < 48 and not _ANALYTIC_HINT.search(s):
        return False
    return True
