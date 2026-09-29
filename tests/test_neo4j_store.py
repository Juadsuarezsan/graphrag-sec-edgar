"""Neo4jGraphStore contract against a recorded fake driver (no Neo4j needed)."""

from __future__ import annotations

from typing import Any

import pytest

from src.graph.base import UnknownNodeError
from src.graph.neo4j_store import NODE_LABEL, VECTOR_INDEX, Neo4jGraphStore, Neo4jTransientError


class FakeSession:
    def __init__(self, driver: FakeDriver) -> None:
        self._driver = driver

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self._driver.queries.append((query, params))
        if self._driver.fail_times > 0:
            self._driver.fail_times -= 1
            raise OSError("connection reset")
        for needle, rows in self._driver.responses:
            if needle in query:
                return rows
        return []


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.responses: list[tuple[str, list[dict[str, Any]]]] = []
        self.fail_times = 0
        self.closed = False

    def session(self, *, database: str | None = None) -> FakeSession:
        return FakeSession(self)

    def close(self) -> None:
        self.closed = True


def _node_row(nid: str, ntype: str = "Company", **props: Any) -> dict[str, Any]:
    base = {
        "id": nid,
        "type": ntype,
        "name": nid.title(),
        "updated_at": "2026-01-01",
        "source_filing_date": "2025-06-30",
        "embedding": [0.1, 0.2],
    }
    base.update({f"p_{k}": v for k, v in props.items()})
    return base


@pytest.fixture
def driver() -> FakeDriver:
    return FakeDriver()


@pytest.fixture
def store(driver: FakeDriver) -> Neo4jGraphStore:
    return Neo4jGraphStore(driver)


def test_ensure_schema_creates_constraint_and_vector_index(
    store: Neo4jGraphStore, driver: FakeDriver
) -> None:
    store.ensure_schema()
    joined = " ".join(q for q, _ in driver.queries)
    assert "CREATE CONSTRAINT" in joined and f"CREATE VECTOR INDEX {VECTOR_INDEX}" in joined
    assert driver.queries[1][1]["dims"] == store.embedder.dims


def test_upsert_node_uses_merge_and_flattens_properties(
    store: Neo4jGraphStore, driver: FakeDriver
) -> None:
    store.upsert_node(
        id="apple",
        type="Company",
        name="Apple",
        properties={"ticker": "AAPL"},
        source_filing_date="2025-10-31",
    )
    query, params = driver.queries[-1]
    assert query.startswith(f"MERGE (n:{NODE_LABEL} {{id: $id}})") and "SET n:Company" in query
    assert params["props"] == {"p_ticker": "AAPL"} and params["sfd"] == "2025-10-31"
    assert params["has_emb"] is False and len(params["embedding"]) == store.embedder.dims


def test_upsert_edge_merges_and_raises_when_endpoint_missing(
    store: Neo4jGraphStore, driver: FakeDriver
) -> None:
    driver.responses.append(("MERGE (a)-[r:SUPPLIED_BY]->(b)", [{"n": 1}]))
    store.upsert_edge(
        from_id="apple", to_id="tsmc", type="SUPPLIED_BY", properties={"component": "chips"}
    )
    assert driver.queries[-1][1]["props"] == {"component": "chips"}
    driver.responses = [("MERGE (a)-[r:COMPETES_WITH]->(b)", [{"n": 0}])]
    with pytest.raises(UnknownNodeError):
        store.upsert_edge(from_id="apple", to_id="ghost", type="COMPETES_WITH")


def test_reads_map_rows_to_nodes_and_edges(store: Neo4jGraphStore, driver: FakeDriver) -> None:
    driver.responses = [
        ("MATCH (n:Entity {id: $id}) RETURN n", [{"n": _node_row("apple", ticker="AAPL")}]),
        (
            "RETURN a.id AS from_id",
            [{"from_id": "apple", "to_id": "tsmc", "type": "SUPPLIED_BY", "props": {}}],
        ),
        ("MATCH (n:Company)", [{"n": 3}]),
        ("RETURN count(n) AS n", [{"n": 7}]),
        ("RETURN count(r) AS n", [{"n": 9}]),
    ]
    node = store.get_node("apple")
    assert (
        node is not None and node["properties"] == {"ticker": "AAPL"} and node["type"] == "Company"
    )
    assert store.edges()[0]["type"] == "SUPPLIED_BY"
    assert store.count_by_type(type="Company", where={"in_sp500": True}) == 3
    assert "n.p_in_sp500 = $w_in_sp500" in driver.queries[-1][0]
    assert store.n_nodes == 7 and store.n_edges == 9
    driver.responses = []
    assert store.get_node("missing") is None


def test_neighbors_vector_search_and_traverse(store: Neo4jGraphStore, driver: FakeDriver) -> None:
    driver.responses = [
        (
            "MATCH (a)-[r:SUPPLIED_BY]->(m)",
            [
                {
                    "m": _node_row("tsmc"),
                    "from_id": "apple",
                    "to_id": "tsmc",
                    "type": "SUPPLIED_BY",
                    "props": {"component": "chips"},
                }
            ],
        ),
        ("db.index.vector.queryNodes", [{"node": _node_row("apple"), "score": 0.93}]),
        ("MATCH (n:Entity {id: $id}) RETURN n", [{"n": _node_row("apple")}]),
        (
            "MATCH p = (s)",
            [
                {
                    "m": _node_row("tsmc"),
                    "hops": 1,
                    "rels": [
                        [{"from_id": "apple", "to_id": "tsmc", "type": "SUPPLIED_BY", "props": {}}]
                    ],
                }
            ],
        ),
    ]
    nbrs = store.neighbors("apple", edge_type="SUPPLIED_BY", direction="out")
    assert nbrs[0][0]["id"] == "tsmc" and nbrs[0][1]["properties"] == {"component": "chips"}
    hits = store.vector_search(query="apple", k=3, type_filter="Company")
    assert hits[0]["id"] == "apple" and hits[0]["similarity"] == 0.93
    assert driver.queries[-1][1]["k_over"] == 12
    sub = store.traverse(seed_ids=["apple"], max_hops=2, edge_types=["SUPPLIED_BY"])
    assert {n["id"]: n["hops_from_seed"] for n in sub["nodes"]} == {"apple": 0, "tsmc": 1}
    assert sub["edges"][0]["type"] == "SUPPLIED_BY"
    assert any("[:SUPPLIED_BY*1..2]" in q for q, _ in driver.queries)


def test_transient_errors_are_retried_then_raised(
    store: Neo4jGraphStore, driver: FakeDriver
) -> None:
    store._run.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    driver.fail_times = 2
    driver.responses = [("RETURN count(n) AS n", [{"n": 1}])]
    assert store.n_nodes == 1
    driver.fail_times = 5
    with pytest.raises(Neo4jTransientError):
        store.clear()
    store.close()
    assert driver.closed
