"""Metrics: hit rate, answer precision, extraction P/R/F1, latency percentiles.

LLM-as-judge rubric (RAGAS-style faithfulness) is declared in
:data:`FAITHFULNESS_RUBRIC` but only runs when ``ANTHROPIC_API_KEY`` is set;
stored results mark the cell as pending otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.api.schemas import GraphAnswer
from src.evaluation.gold import GoldItem
from src.extraction.schemas import ExtractionResult

#: Numbered criteria for the faithfulness judge (pending an API key).
FAITHFULNESS_RUBRIC: tuple[str, ...] = (
    "1. Every factual claim in the answer is supported by a cited node or edge in the context.",
    "2. No entity is named that is absent from the retrieved subgraph.",
    "3. Counts and aggregates equal the number of matching nodes in the context.",
    "4. The answer does not contradict any relationship in the context.",
    "5. Score = supported_claims / total_claims, in [0, 1]; empty answers score 0.",
)


@dataclass(frozen=True)
class HitResult:
    """Per-question scoring."""

    hit: bool
    precision: float
    recall: float
    routing_ok: bool


def score_answer(item: GoldItem, answer: GraphAnswer, expected_kind: str) -> HitResult:
    """Score one answer against gold.

    * lookup / multi-hop: **hit** when every gold id is in ``answer_ids`` (and the
      expected text, if any, appears in the answer); precision/recall over ids.
    * aggregation: **hit** when ``count`` equals ``expected_count``.
    """
    got = set(answer.answer_ids)
    exp = set(item.expected_ids)
    inter = got & exp
    precision = len(inter) / len(got) if got else 0.0
    recall = len(inter) / len(exp) if exp else 1.0
    if item.category == "aggregation":
        hit = answer.count is not None and answer.count == item.expected_count
    else:
        hit = exp <= got
        if item.expected_text and item.expected_text not in answer.answer:
            hit = False
    return HitResult(
        hit=hit,
        precision=precision,
        recall=recall,
        routing_ok=answer.classified_kind == expected_kind,
    )


def percentile(values: list[int | float], pct: float) -> float:
    """Nearest-rank percentile (``pct`` in [0, 100]); 0 for empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return float(ordered[k])


@dataclass(frozen=True)
class PRF:
    """Precision / recall / F1 with support."""

    precision: float
    recall: float
    f1: float
    n_gold: int
    n_pred: int

    def as_dict(self) -> dict[str, float | int]:
        """Serializable view."""
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
        }


def _prf(gold: set[object], pred: set[object]) -> PRF:
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF(p, r, f1, len(gold), len(pred))


def extraction_prf(
    gold: ExtractionResult, pred: ExtractionResult, focal_id: str | None = None
) -> tuple[PRF, PRF]:
    """Entity and relationship P/R/F1 between two extraction results.

    Entities match on ``(id, type)``; relationships on ``(from, to, type)``.
    The focal filer is excluded from the entity sets because every extractor
    receives it as input.
    """
    ge = {(e.id, e.type) for e in gold.entities if e.id != focal_id}
    pe = {(e.id, e.type) for e in pred.entities if e.id != focal_id}
    gr = {(r.from_id, r.to_id, r.type) for r in gold.relationships}
    pr = {(r.from_id, r.to_id, r.type) for r in pred.relationships}
    return _prf(set(ge), set(pe)), _prf(set(gr), set(pr))
