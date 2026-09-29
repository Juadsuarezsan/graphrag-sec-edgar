"""Gold-set loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["lookup", "two_hop", "three_hop", "aggregation"]

#: Category → engine kind expected by the routing metric.
CATEGORY_TO_KIND: dict[str, str] = {
    "lookup": "lookup",
    "two_hop": "multi_hop",
    "three_hop": "multi_hop",
    "aggregation": "aggregation",
}
EXPECTED_COUNTS: dict[str, int] = {"lookup": 30, "two_hop": 30, "three_hop": 20, "aggregation": 20}


class GoldItem(BaseModel):
    """One evaluation question with hand-written ground truth."""

    id: str
    category: Category
    question: str = Field(min_length=5)
    expected_ids: list[str] = Field(default_factory=list)
    expected_count: int | None = None
    expected_text: str | None = None
    hops: int = Field(ge=0, le=4)
    notes: str = ""
    ground_truth: str = "manual"

    @model_validator(mode="after")
    def _aggregation_has_count(self) -> GoldItem:
        if self.category == "aggregation" and self.expected_count is None:
            raise ValueError(f"{self.id}: aggregation item needs expected_count")
        if self.category != "aggregation" and not self.expected_ids:
            raise ValueError(f"{self.id}: needs expected_ids")
        return self


def load_gold(path: Path) -> list[GoldItem]:
    """Read a JSONL gold file, validating every row."""
    items: list[GoldItem] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(GoldItem.model_validate(json.loads(line)))
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate ids in gold set")
    return items


def category_counts(items: list[GoldItem]) -> dict[str, int]:
    """Number of items per category."""
    out: dict[str, int] = {}
    for it in items:
        out[it.category] = out.get(it.category, 0) + 1
    return out
