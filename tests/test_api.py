"""HTTP layer: health, query modes, 422 edge cases, graph, stale, node, schema, rate limit, CORS."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.config import get_settings


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["nodes"] > 250 and body["edges"] > 400
    assert body["llm_enabled"] is False and body["graph_backend"] == "memory"
    assert r.headers["X-Trace-Id"]


def test_query_modes(client: TestClient) -> None:
    r = client.post("/api/query", json={"question": "Who is the CEO of Apple?"})
    assert r.status_code == 200
    body = r.json()
    assert (
        body["answer_ids"] == ["tim_cook"]
        and body["mode"] == "graph"
        and body["synthesizer"] == "template"
    )
    assert body["metrics"]["trace_id"] and body["metrics"]["llm_calls"] == 0
    r = client.post(
        "/api/query",
        json={"question": "How many subsidiaries does Alphabet have?", "mode": "hybrid"},
    )
    assert r.json()["count"] == 4
    r = client.post(
        "/api/query", json={"question": "Who is the CEO of Apple?", "mode": "vector", "k": 3}
    )
    assert r.status_code == 200 and r.json()["mode"] == "vector"
    r = client.post("/api/query", json={"question": "Which products does Alphabet offer?"})
    assert "prod_gmail" in r.json()["answer_ids"]  # real GOOGL excerpt merged at startup


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": "ab"},
        {"question": "x" * 1001},
        {"question": "Who is the CEO of Apple?", "k": 0},
        {"question": "Who is the CEO of Apple?", "k": 26},
        {"question": "Who is the CEO of Apple?", "max_hops": 9},
        {"question": "Who is the CEO of Apple?", "mode": "sparql"},
        {"nope": 1},
    ],
)
def test_query_validation_422(client: TestClient, payload: dict[str, object]) -> None:
    r = client.post("/api/query", json=payload)
    assert r.status_code == 422


def test_malformed_json_422(client: TestClient) -> None:
    r = client.post(
        "/api/query", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert r.status_code == 422


def test_graph_snapshot_and_filters(client: TestClient) -> None:
    r = client.get("/api/graph", params={"limit": 50, "types": "Company,Person"})
    body = r.json()
    assert r.status_code == 200 and len(body["nodes"]) == 50 and body["truncated"] is True
    assert {n["type"] for n in body["nodes"]} <= {"Company", "Person"}
    assert all("embedding" not in n and "aliases" not in n["properties"] for n in body["nodes"])
    assert body["total_nodes"] > 250 and all(
        {"from", "to", "type"} <= set(e) for e in body["edges"]
    )
    assert client.get("/api/graph", params={"limit": 0}).status_code == 422


def test_stale_endpoint(client: TestClient) -> None:
    r = client.get("/api/stale")
    body = r.json()
    assert r.status_code == 200 and body["max_age_days"] == 365 and body["total_nodes"] > 0
    ids = {s["id"] for s in body["stale_nodes"]}
    assert (
        "samsung" in ids and "apple" not in ids
    )  # Samsung's filing date in the fixture is from 2024
    assert all(s["age_days"] is None or s["age_days"] > 365 for s in body["stale_nodes"])
    assert client.get("/api/stale", params={"max_age_days": 0}).status_code == 422
    graph = client.get("/api/graph", params={"types": "Company", "limit": 100}).json()
    assert any(n["stale"] for n in graph["nodes"]) and any(not n["stale"] for n in graph["nodes"])


def test_node_detail_and_schema(client: TestClient) -> None:
    r = client.get("/api/node/apple")
    body = r.json()
    assert (
        r.status_code == 200
        and body["name"] == "Apple Inc."
        and body["source_filing_date"] == "2025-10-31"
    )
    assert any(
        n["id"] == "tim_cook" and n["edge"] == "CEO_OF" and n["direction"] == "in"
        for n in body["neighbors"]
    )
    assert client.get("/api/node/ghost").status_code == 404
    schema = client.get("/api/schema").json()
    assert "Subsidiary" in schema["node_types"] and schema["edge_signatures"]["CEO_OF"] == [
        "Person",
        "Company",
    ]


def test_cors_restricted_to_configured_origins(client: TestClient) -> None:
    ok = client.options(
        "/api/query",
        headers={"Origin": "https://demo.example", "Access-Control-Request-Method": "POST"},
    )
    assert ok.headers.get("access-control-allow-origin") == "https://demo.example"
    bad = client.options(
        "/api/query",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in bad.headers
    assert get_settings().cors_origin_list == ["http://localhost:8000", "https://demo.example"]


@pytest.fixture
def strict_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("RATE_LIMIT", "2/minute")
    get_settings.cache_clear()
    from src.api.main import app

    app.state.limiter.reset()
    with TestClient(app) as c:
        yield c
    app.state.limiter.reset()


def test_rate_limit_returns_429(strict_client: TestClient) -> None:
    payload = {"question": "Who is the CEO of Apple?"}
    assert strict_client.post("/api/query", json=payload).status_code == 200
    assert strict_client.post("/api/query", json=payload).status_code == 200
    r = strict_client.post("/api/query", json=payload)
    assert r.status_code == 429 and "rate limit" in r.json()["detail"] and r.json()["trace_id"]


def test_demo_is_served(client: TestClient) -> None:
    r = client.get("/demo/")
    assert r.status_code == 200 and "<html" in r.text.lower()
