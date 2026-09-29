"""HTTP schemas (request validation → 422, typed responses)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

QueryKind = Literal["lookup", "multi_hop", "aggregation", "hybrid"]
QueryMode = Literal["graph", "vector", "hybrid"]

MAX_QUESTION_CHARS = 1000


class GraphQuery(BaseModel):
    """Body of ``POST /api/query``."""

    question: str = Field(
        min_length=3, max_length=MAX_QUESTION_CHARS, description="Natural-language question"
    )
    max_hops: int = Field(default=3, ge=1, le=4)
    k: int = Field(default=5, ge=1, le=25)
    mode: QueryMode = "graph"


class CitedNode(BaseModel):
    """A node the answer relies on."""

    id: str
    type: str
    name: str
    similarity: float | None = None
    hops_from_seed: int | None = None


class PlanStep(BaseModel):
    """One traversal step of the executed plan."""

    edge_types: list[str]
    direction: Literal["in", "out", "both"]


class SubgraphOut(BaseModel):
    """Nodes/edges to render in the demo."""

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class QueryMetrics(BaseModel):
    """Per-request observability fields."""

    trace_id: str
    latency_ms: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0


class GraphAnswer(BaseModel):
    """Response of ``POST /api/query``."""

    question: str
    mode: QueryMode
    answer: str
    classified_kind: QueryKind
    answer_ids: list[str] = Field(default_factory=list)
    count: int | None = None
    seeds: list[str] = Field(default_factory=list)
    plan: list[PlanStep] = Field(default_factory=list)
    cited_nodes: list[CitedNode] = Field(default_factory=list)
    reasoning_path: list[str] = Field(default_factory=list)
    subgraph: SubgraphOut = Field(default_factory=SubgraphOut)
    synthesizer: str = "template"
    metrics: QueryMetrics


class HealthOut(BaseModel):
    """Response of ``GET /health``."""

    status: Literal["ok"]
    version: str
    graph_backend: str
    embedder: str
    nodes: int
    edges: int
    llm_enabled: bool


class GraphSnapshot(BaseModel):
    """Response of ``GET /api/graph``."""

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    total_nodes: int
    total_edges: int
    truncated: bool


class ErrorOut(BaseModel):
    """Uniform error body."""

    detail: str
    trace_id: str
