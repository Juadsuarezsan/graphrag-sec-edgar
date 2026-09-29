"""GraphRAG SEC EDGAR API.

Endpoints: ``GET /health``, ``POST /api/query``, ``GET /api/graph``,
``GET /api/stale``, ``GET /api/node/{id}``, ``GET /api/schema``. The static
demo is served under ``/demo``.
"""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.schemas import ErrorOut, GraphAnswer, GraphQuery, GraphSnapshot, HealthOut
from src.config import Settings, get_settings
from src.engine.graphrag import EngineError, GraphRAGEngine
from src.engine.synthesis import ClaudeSynthesizer, Synthesizer, TemplateSynthesizer
from src.extraction.rules import RuleBasedExtractor
from src.extraction.schemas import EDGE_SIGNATURES, EDGE_TYPES, NODE_TYPES
from src.graph import staleness
from src.graph.base import GraphStore, UnknownNodeError
from src.graph.fixture import populate
from src.graph.store import InMemoryGraph
from src.ingestion.pipeline import SAMPLE_DIR, ingest_sections, load_sample_filings
from src.observability import (
    LangSmithTracer,
    RequestMetrics,
    configure_logging,
    current_trace_id,
    new_trace_id,
)

API_VERSION = "0.6.0"
ROOT = Path(__file__).resolve().parents[2]


def build_store(settings: Settings) -> GraphStore:
    """Create the graph store selected by ``GRAPH_BACKEND`` and populate it."""
    if settings.graph_backend == "neo4j":
        from src.graph.neo4j_store import Neo4jGraphStore

        neo = Neo4jGraphStore.from_settings(
            settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password
        )
        neo.ensure_schema()
        if neo.n_nodes == 0:
            populate(neo)
        return neo
    if settings.graph_backend == "networkx":
        from src.graph.networkx_store import NetworkXGraphStore

        nx_store = NetworkXGraphStore(ROOT / settings.graph_path)
        if nx_store.n_nodes == 0:
            populate(nx_store)
            _ingest_samples(nx_store)
            nx_store.save()
        return nx_store
    store = InMemoryGraph()
    populate(store)
    _ingest_samples(store)
    return store


def _ingest_samples(store: GraphStore) -> None:
    extractor = RuleBasedExtractor()
    for filing, section in load_sample_filings(ROOT / SAMPLE_DIR):
        ingest_sections(filing, [section], extractor, store)


def build_synthesizer(settings: Settings) -> Synthesizer:
    """Claude when a key is configured, deterministic template otherwise."""
    if settings.anthropic_api_key:
        return ClaudeSynthesizer(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_s=settings.request_timeout_s,
        )
    return TemplateSynthesizer()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the store, engine, tracer and staleness flags once per process."""
    settings = get_settings()
    configure_logging(settings.log_level)
    store = build_store(settings)
    staleness.mark(store, datetime.now(UTC).date(), settings.stale_after_days)
    app.state.settings = settings
    app.state.store = store
    app.state.engine = GraphRAGEngine(store, synthesizer=build_synthesizer(settings))
    app.state.tracer = LangSmithTracer(settings)
    logger.info(
        "graph ready: backend={} nodes={} edges={}",
        settings.graph_backend,
        store.n_nodes,
        store.n_edges,
    )
    yield


_settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[])

app = FastAPI(
    title="GraphRAG SEC EDGAR",
    version=API_VERSION,
    description=(
        "Knowledge graph over S&P 500 10-K filings with multi-hop traversal and a vector baseline."
    ),
    lifespan=lifespan,
    responses={422: {"model": ErrorOut}, 429: {"model": ErrorOut}, 500: {"model": ErrorOut}},
)
app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
_demo_dir = ROOT / "demo"
if _demo_dir.is_dir():
    app.mount("/demo", StaticFiles(directory=str(_demo_dir), html=True), name="demo")


@app.middleware("http")
async def trace_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Bind a trace id, time the request and log one structured line."""
    trace_id = request.headers.get("x-trace-id") or new_trace_id()
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    response.headers["X-Trace-Id"] = trace_id
    logger.bind(trace_id=trace_id).info(
        "{} {} -> {} latency_ms={}",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


def _error(status: int, detail: str) -> JSONResponse:
    body = ErrorOut(detail=detail, trace_id=current_trace_id()).model_dump()
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(RateLimitExceeded)
async def _rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return _error(429, f"rate limit exceeded: {exc.detail}")


@app.exception_handler(UnknownNodeError)
async def _unknown_node(request: Request, exc: UnknownNodeError) -> JSONResponse:
    return _error(404, str(exc))


@app.exception_handler(EngineError)
async def _engine_error(request: Request, exc: EngineError) -> JSONResponse:
    logger.warning("engine error: {}", exc)
    return _error(422, str(exc))


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error: {}", exc)
    return _error(500, "internal error; see server logs with the trace id")


@app.get("/health", response_model=HealthOut)
async def health(request: Request) -> HealthOut:
    """Liveness + graph size."""
    s: Settings = request.app.state.settings
    store: GraphStore = request.app.state.store
    return HealthOut(
        status="ok",
        version=API_VERSION,
        graph_backend=s.graph_backend,
        embedder=store.embedder.name,
        nodes=store.n_nodes,
        edges=store.n_edges,
        llm_enabled=bool(s.anthropic_api_key),
    )


@app.post("/api/query", response_model=GraphAnswer)
@limiter.limit(lambda: get_settings().rate_limit)
async def query(request: Request, q: GraphQuery) -> GraphAnswer:
    """Answer a question in ``graph``, ``vector`` or ``hybrid`` mode."""
    engine: GraphRAGEngine = request.app.state.engine
    answer = await engine.answer(q.question, k=q.k, max_hops=q.max_hops, mode=q.mode)
    tracer: LangSmithTracer = request.app.state.tracer
    if tracer.enabled:
        tracer.export(
            _metrics_from(answer),
            inputs={"question": q.question, "mode": q.mode},
            outputs={"answer": answer.answer, "answer_ids": answer.answer_ids},
        )
    return answer


def _metrics_from(answer: GraphAnswer) -> RequestMetrics:
    m = RequestMetrics(trace_id=answer.metrics.trace_id, name="query")
    m.latency_ms = answer.metrics.latency_ms
    m.tokens_in = answer.metrics.tokens_in
    m.tokens_out = answer.metrics.tokens_out
    m.cost_usd = answer.metrics.cost_usd
    return m


@app.get("/api/graph", response_model=GraphSnapshot)
async def graph_snapshot(
    request: Request,
    limit: int = Query(default=400, ge=1, le=5000),
    types: str | None = Query(default=None, description="Comma-separated node types"),
) -> GraphSnapshot:
    """Nodes and edges for visualisation (embeddings omitted)."""
    store: GraphStore = request.app.state.store
    wanted = {t.strip() for t in types.split(",")} if types else None
    nodes = [n for n in store.nodes() if wanted is None or n["type"] in wanted]
    truncated = len(nodes) > limit
    nodes = nodes[:limit]
    keep = {n["id"] for n in nodes}
    edges = [e for e in store.edges() if e["from_id"] in keep and e["to_id"] in keep]
    return GraphSnapshot(
        nodes=[
            {
                "id": n["id"],
                "type": n["type"],
                "name": n["name"],
                "stale": bool(n["properties"].get("stale", False)),
                "source_filing_date": n["source_filing_date"],
                "properties": {k: v for k, v in n["properties"].items() if k != "aliases"},
            }
            for n in nodes
        ],
        edges=[{"from": e["from_id"], "to": e["to_id"], "type": e["type"]} for e in edges],
        total_nodes=store.n_nodes,
        total_edges=store.n_edges,
        truncated=truncated,
    )


@app.get("/api/stale")
async def stale(
    request: Request, max_age_days: int = Query(default=365, ge=1, le=3650)
) -> dict[str, Any]:
    """Nodes whose evidence is older than ``max_age_days`` (default 365)."""
    store: GraphStore = request.app.state.store
    return staleness.scan(store, datetime.now(UTC).date(), max_age_days).as_dict()


@app.get("/api/node/{node_id}")
async def node_detail(request: Request, node_id: str) -> dict[str, Any]:
    """One node with its neighbours; 404 when unknown."""
    store: GraphStore = request.app.state.store
    node = store.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"unknown node {node_id}")
    neighbors = [
        {
            "id": o["id"],
            "type": o["type"],
            "name": o["name"],
            "edge": e["type"],
            "direction": "out" if e["from_id"] == node_id else "in",
        }
        for o, e in store.neighbors(node_id)
    ]
    return {
        "id": node["id"],
        "type": node["type"],
        "name": node["name"],
        "properties": {k: v for k, v in node["properties"].items() if k != "aliases"},
        "updated_at": node["updated_at"],
        "source_filing_date": node["source_filing_date"],
        "neighbors": neighbors,
    }


@app.get("/api/schema")
async def schema() -> dict[str, Any]:
    """Node types, edge types and edge signatures (mirrors docs/graph_schema.md)."""
    return {
        "node_types": list(NODE_TYPES),
        "edge_types": list(EDGE_TYPES),
        "edge_signatures": {k: list(v) for k, v in EDGE_SIGNATURES.items()},
    }
