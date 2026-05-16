"""GraphRAG SEC EDGAR — placeholder until v0.1.0 build out."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="GraphRAG SEC EDGAR",
    version="0.1.0",
    description="GraphRAG over SEC EDGAR — Neo4j + multi-hop Cypher + vector hybrid",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "version": "0.1.0",
        "stage": "scaffolding",
        "llm_enabled": "yes" if s.anthropic_api_key else "no",
    }

class GraphQuery(BaseModel):
    question: str
    max_hops: int = 2
    k: int = 5


class GraphAnswer(BaseModel):
    question: str
    answer: str = ""
    cited_nodes: list[dict] = []
    reasoning_path: list[str] = []
    classified_kind: str = ""  # lookup | multi_hop | aggregation | hybrid
    latency_ms: int = 0


@app.post("/api/query", response_model=GraphAnswer)
async def graph_query(q: GraphQuery) -> GraphAnswer:
    return GraphAnswer(question=q.question, answer="not_yet_implemented")
