"""Regex query classifier: lookup / multi_hop / aggregation / hybrid.

Kept as a fast pre-filter and for the eval's routing metric; the planner's
structural classification (:func:`src.engine.planner.classify_plan`) is what
the engine reports.
"""

from __future__ import annotations

import re
from typing import Literal

QueryKind = Literal["lookup", "multi_hop", "aggregation", "hybrid"]

LOOKUP_HINTS = [r"\bwho is\b", r"\bwhat is\b", r"\bwhen\b", r"\bwhere\b", r"\bwhich company does\b"]
MULTI_HOP_HINTS = [
    r"\bdepend(s|ed|ing)? on\b",
    r"\bsupplied by\b",
    r"\bsuppliers? of\b",
    r"\bcompetitors? of\b",
    r"\bcompete(s)? with\b",
    r"\bsame (supplier|board|ceo|risk)\b",
    r"\bshare[sd]? (a|the|an) \w+\b",
    r"\bparent company of\b",
    r"\bconnected to\b",
    r"\bvia\b",
    r"\bof (the )?(companies|competitors|suppliers|subsidiaries) (that|which|supplied|led)\b",
]
AGG_HINTS = [r"\bhow many\b", r"\bcount\b", r"\btotal\b", r"\bnumber of\b"]


def classify(question: str) -> QueryKind:
    """Classify with hint counts; aggregation wins, then multi-hop, then lookup."""
    q = question.lower()
    lk = sum(1 for p in LOOKUP_HINTS if re.search(p, q))
    mh = sum(1 for p in MULTI_HOP_HINTS if re.search(p, q))
    ag = sum(1 for p in AGG_HINTS if re.search(p, q))
    if ag > 0:
        return "aggregation"
    if mh > 0:
        return "multi_hop"
    if lk > 0:
        return "lookup"
    return "hybrid"
