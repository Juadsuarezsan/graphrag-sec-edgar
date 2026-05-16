"""GraphRAG SEC EDGAR API."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from src.api.schemas import GraphAnswer, GraphQuery
from src.config import get_settings
from src.engine.graphrag import answer
from src.graph.fixture import build_demo_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_demo_graph()
    yield


app = FastAPI(
    title="GraphRAG SEC EDGAR",
    version="0.5.0",
    description="In-memory knowledge graph over S&P 500 fixture; routes lookup/multi_hop/aggregation.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    s = get_settings()
    g = app.state.graph
    return {
        "status": "ok", "version": "0.5.0", "stage": "substantive",
        "neo4j_uri": s.neo4j_uri, "nodes": str(g.n_nodes), "edges": str(g.n_edges),
    }


@app.post("/api/query", response_model=GraphAnswer)
async def query(q: GraphQuery) -> GraphAnswer:
    try:
        return await answer(app.state.graph, q.question, k=q.k, max_hops=q.max_hops)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/graph")
async def graph_snapshot() -> dict:
    g = app.state.graph
    return {
        "nodes": [{"id": n["id"], "type": n["type"], "name": n["name"],
                   "properties": n["properties"]}
                  for n in g._nodes.values()],
        "edges": [{"from": e["from"], "to": e["to"], "type": e["type"]} for e in g._edges],
    }


@app.get("/api/eval/run")
async def eval_endpoint() -> dict:
    from src.eval.runner import run_eval
    return await run_eval()
