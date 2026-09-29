"""Contract tests shared by the in-memory and NetworkX stores, plus persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.graph.base import GraphStore, UnknownNodeError
from src.graph.fixture import build_demo_graph, fixture_summary, populate
from src.graph.networkx_store import NetworkXGraphStore
from src.graph.store import InMemoryGraph


@pytest.fixture(params=["memory", "networkx"])
def store(request: pytest.FixtureRequest) -> GraphStore:
    return InMemoryGraph() if request.param == "memory" else NetworkXGraphStore()


def _small(store: GraphStore) -> GraphStore:
    store.upsert_node(
        id="a",
        type="Company",
        name="A Corp",
        properties={"in_sp500": True},
        source_filing_date="2025-01-01",
    )
    store.upsert_node(id="b", type="Company", name="B Corp", properties={"in_sp500": False})
    store.upsert_node(id="p", type="Person", name="P Person", properties={"role": "CEO"})
    store.upsert_edge(from_id="p", to_id="a", type="CEO_OF", properties={"since": "2011"})
    store.upsert_edge(from_id="a", to_id="b", type="SUPPLIED_BY")
    return store


def test_merge_semantics(store: GraphStore) -> None:
    _small(store)
    store.upsert_node(id="a", type="Company", name="A Corp", properties={"sector": "Tech"})
    node = store.get_node("a")
    assert node is not None
    assert node["properties"] == {"in_sp500": True, "sector": "Tech"}
    assert node["source_filing_date"] == "2025-01-01"
    store.upsert_edge(from_id="a", to_id="b", type="SUPPLIED_BY", properties={"component": "chips"})
    assert store.n_nodes == 3 and store.n_edges == 2
    supplied = next(e for e in store.edges() if e["type"] == "SUPPLIED_BY")
    assert supplied["properties"] == {"component": "chips"}
    assert isinstance(store, GraphStore)


def test_unknown_node_raises(store: GraphStore) -> None:
    _small(store)
    with pytest.raises(UnknownNodeError):
        store.upsert_edge(from_id="a", to_id="zzz", type="COMPETES_WITH")
    assert store.get_node("zzz") is None
    assert store.neighbors("zzz") == []


def test_neighbors_direction_and_type(store: GraphStore) -> None:
    _small(store)
    assert {n["id"] for n, _ in store.neighbors("a")} == {"p", "b"}
    assert [n["id"] for n, _ in store.neighbors("a", direction="in")] == ["p"]
    assert [n["id"] for n, _ in store.neighbors("a", direction="out")] == ["b"]
    assert [e["type"] for _, e in store.neighbors("a", edge_type="CEO_OF")] == ["CEO_OF"]


def test_traverse_and_count(store: GraphStore) -> None:
    _small(store)
    sub = store.traverse(seed_ids=["p", "ghost"], max_hops=2)
    hops = {n["id"]: n["hops_from_seed"] for n in sub["nodes"]}
    assert hops == {"p": 0, "a": 1, "b": 2}
    assert len(sub["edges"]) == 2
    only = store.traverse(seed_ids=["p"], max_hops=2, edge_types=["CEO_OF"])
    assert {n["id"] for n in only["nodes"]} == {"p", "a"}
    assert store.count_by_type(type="Company") == 2
    assert store.count_by_type(type="Company", where={"in_sp500": True}) == 1


def test_vector_search_prefers_matching_name(store: GraphStore) -> None:
    _small(store)
    res = store.vector_search(query="A Corp Company", k=2)
    assert res[0]["id"] == "a" and res[0]["similarity"] >= res[1]["similarity"]
    assert [r["id"] for r in store.vector_search(query="P Person", k=5, type_filter="Person")] == [
        "p"
    ]
    store.clear()
    assert store.n_nodes == 0 and store.n_edges == 0


def test_networkx_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "g.json"
    g = NetworkXGraphStore(path)
    populate(g)
    saved = g.save()
    assert saved == path and path.exists()
    g2 = NetworkXGraphStore(path)
    assert g2.n_nodes == g.n_nodes and g2.n_edges == g.n_edges
    apple = g2.get_node("apple")
    assert apple is not None and apple["properties"]["ticker"] == "AAPL"
    assert apple["source_filing_date"] == "2025-10-31"
    with pytest.raises(ValueError):
        NetworkXGraphStore().save()


def test_fixture_summary_counts() -> None:
    g = build_demo_graph()
    s = fixture_summary()
    assert s["nodes"] == g.n_nodes >= 250 and s["edges"] == g.n_edges >= 400
    assert set(s["by_node_type"]) == {
        "Company",
        "Person",
        "Subsidiary",
        "Product",
        "Risk",
        "Market",
    }
    assert set(s["by_edge_type"]) == {
        "CEO_OF",
        "DIRECTOR_OF",
        "SUBSIDIARY_OF",
        "OFFERS_PRODUCT",
        "MENTIONS_RISK",
        "OPERATES_IN",
        "SUPPLIED_BY",
        "COMPETES_WITH",
    }
    directors = [n for n in g.nodes() if n["properties"].get("role") == "Director"]
    assert directors and all(n["properties"].get("synthetic") is True for n in directors)
