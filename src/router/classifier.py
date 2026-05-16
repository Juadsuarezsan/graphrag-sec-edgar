"""Query classifier — chooses lookup / multi_hop / aggregation / hybrid path."""
from __future__ import annotations

import re

LOOKUP_HINTS = [r"\bwho is\b", r"\bwhat is\b", r"\bwhen\b", r"\bwhere\b"]
MULTI_HOP_HINTS = [
    r"\bdepend(s|ed|ing)? on\b",
    r"\bsupplied by\b",
    r"\b(signed|approved) by\b",
    r"\bcompetitor(s)? of\b",
    r"\bsame (supplier|board)\b",
    r"\bconnected to\b",
    r"\bvia\b",
]
AGG_HINTS = [r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bover \$?\d", r"\bin (the )?(s&p|emea|us)\b"]


def classify(question: str) -> str:
    q = question.lower()
    lk = sum(1 for p in LOOKUP_HINTS if re.search(p, q))
    mh = sum(1 for p in MULTI_HOP_HINTS if re.search(p, q))
    ag = sum(1 for p in AGG_HINTS if re.search(p, q))
    if ag > 0 and mh == 0:
        return "aggregation"
    if mh > 0:
        return "multi_hop"
    if lk > 0 and mh == 0 and ag == 0:
        return "lookup"
    return "hybrid"
