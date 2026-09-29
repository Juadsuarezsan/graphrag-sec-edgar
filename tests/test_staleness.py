"""Staleness detector with fixed dates."""

from __future__ import annotations

from datetime import date

from src.graph import staleness
from src.graph.base import Node
from src.graph.store import InMemoryGraph

NOW = date(2026, 9, 29)


def _graph() -> InMemoryGraph:
    g = InMemoryGraph()
    g.upsert_node(
        id="fresh",
        type="Company",
        name="Fresh",
        source_filing_date="2026-01-15",
        updated_at="2026-09-01",
    )
    g.upsert_node(
        id="edge365",
        type="Company",
        name="Edge",
        source_filing_date="2025-09-29",
        updated_at="2026-09-01",
    )
    g.upsert_node(
        id="stale",
        type="Company",
        name="Stale",
        source_filing_date="2025-09-28",
        updated_at="2026-09-01",
    )
    g.upsert_node(id="updated_only", type="Person", name="Upd", updated_at="2024-01-01")
    return g


NODATE: Node = {
    "id": "nodate",
    "type": "Risk",
    "name": "NoDate",
    "properties": {},
    "embedding": [],
    "updated_at": "not-a-date",
    "source_filing_date": None,
}


def test_is_stale_boundaries() -> None:
    g = _graph()
    assert not staleness.is_stale(g.get_node("fresh"), NOW)  # type: ignore[arg-type]
    assert not staleness.is_stale(g.get_node("edge365"), NOW)  # type: ignore[arg-type]  exactly 365 days
    assert staleness.is_stale(g.get_node("stale"), NOW)  # type: ignore[arg-type]  366 days
    assert staleness.is_stale(NODATE, NOW)  # unknown provenance is never fresh
    assert staleness.is_stale(g.get_node("updated_only"), NOW)  # type: ignore[arg-type]
    assert staleness.node_age_days(NODATE, NOW) is None


def test_scan_and_mark() -> None:
    g = _graph()
    report = staleness.scan(g, NOW)
    assert report.total_nodes == 4 and report.stale_count == 2
    assert [s.id for s in report.stale_nodes] == ["updated_only", "stale"]  # oldest first
    assert report.as_dict()["stale_ratio"] == 0.5
    staleness.mark(g, NOW)
    assert g.get_node("stale")["properties"]["stale"] is True  # type: ignore[index]
    assert g.get_node("fresh")["properties"]["stale"] is False  # type: ignore[index]
    assert g.get_node("stale")["updated_at"] == "2026-09-01"  # type: ignore[index]  mark keeps updated_at
    assert staleness.scan(g, NOW, max_age_days=30).stale_count == 4
    assert staleness.scan(InMemoryGraph(), NOW).stale_ratio == 0.0
