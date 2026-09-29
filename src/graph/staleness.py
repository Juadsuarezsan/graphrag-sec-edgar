"""Staleness detector: flags nodes whose evidence is older than a threshold.

Every node carries ``updated_at`` (last merge) and ``source_filing_date``
(date of the filing that produced it). A node is stale when the most
reliable of those dates is more than ``max_age_days`` before ``now``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from src.graph.base import GraphStore, Node

DEFAULT_MAX_AGE_DAYS = 365


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return None


def node_age_days(node: Node, now: date) -> int | None:
    """Age in days from ``source_filing_date`` (preferred) or ``updated_at``.

    Returns ``None`` when neither date is parseable.
    """
    ref = _parse_date(node.get("source_filing_date")) or _parse_date(node.get("updated_at"))
    return (now - ref).days if ref else None


def is_stale(node: Node, now: date, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> bool:
    """``True`` when the node's evidence is older than ``max_age_days``.

    Nodes without any parseable date are treated as stale: unknown provenance
    must never pass as fresh.
    """
    age = node_age_days(node, now)
    return age is None or age > max_age_days


@dataclass
class StaleNode:
    """One stale node in a report."""

    id: str
    type: str
    name: str
    age_days: int | None
    source_filing_date: str | None
    updated_at: str


@dataclass
class StalenessReport:
    """Summary of a staleness scan."""

    as_of: str
    max_age_days: int
    total_nodes: int
    stale_count: int
    stale_nodes: list[StaleNode] = field(default_factory=list)

    @property
    def stale_ratio(self) -> float:
        """Fraction of stale nodes (0 when the graph is empty)."""
        return self.stale_count / self.total_nodes if self.total_nodes else 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "as_of": self.as_of,
            "max_age_days": self.max_age_days,
            "total_nodes": self.total_nodes,
            "stale_count": self.stale_count,
            "stale_ratio": round(self.stale_ratio, 4),
            "stale_nodes": [vars(n) for n in self.stale_nodes],
        }


def scan(store: GraphStore, now: date, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> StalenessReport:
    """Scan every node and return the stale ones (read-only)."""
    stale: list[StaleNode] = []
    nodes = store.nodes()
    for n in nodes:
        if is_stale(n, now, max_age_days):
            stale.append(
                StaleNode(
                    id=n["id"],
                    type=n["type"],
                    name=n["name"],
                    age_days=node_age_days(n, now),
                    source_filing_date=n.get("source_filing_date"),
                    updated_at=n["updated_at"],
                )
            )
    stale.sort(key=lambda s: (-(s.age_days or 10**9), s.id))
    return StalenessReport(
        as_of=now.isoformat(),
        max_age_days=max_age_days,
        total_nodes=len(nodes),
        stale_count=len(stale),
        stale_nodes=stale,
    )


def mark(store: GraphStore, now: date, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> StalenessReport:
    """Scan and persist ``properties.stale`` on every node (keeps ``updated_at``)."""
    report = scan(store, now, max_age_days)
    stale_ids = {s.id for s in report.stale_nodes}
    for n in store.nodes():
        store.upsert_node(
            id=n["id"],
            type=n["type"],
            name=n["name"],
            properties={"stale": n["id"] in stale_ids},
            updated_at=n["updated_at"],
        )
    return report
