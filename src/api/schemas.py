from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QueryKind = Literal["lookup", "multi_hop", "aggregation", "hybrid"]
NodeType = Literal["Company", "Person", "Product", "Risk", "Subsidiary"]


class Entity(BaseModel):
    id: str
    type: NodeType
    name: str
    properties: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    from_id: str
    to_id: str
    type: str
    properties: dict = Field(default_factory=dict)


class GraphQuery(BaseModel):
    question: str
    max_hops: int = 2
    k: int = 5


class CitedNode(BaseModel):
    id: str
    type: str
    name: str
    similarity: float | None = None
    hops_from_seed: int | None = None


class GraphAnswer(BaseModel):
    question: str
    answer: str = ""
    classified_kind: QueryKind = "lookup"
    cited_nodes: list[CitedNode] = Field(default_factory=list)
    reasoning_path: list[str] = Field(default_factory=list)
    latency_ms: int = 0
