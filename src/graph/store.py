"""In-memory graph store. Same shape as Neo4j driver wrapper would expose.

Production swaps `neo4j.AsyncDriver` behind the same interface — the engine
+ classifier + eval all work without Neo4j running.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Iterable

_DIMS = 256
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def stub_embed(text: str) -> list[float]:
    tokens = _TOKEN_RE.findall(text.lower()) or ["<empty>"]
    acc = [0.0] * _DIMS
    for t in tokens:
        d = hashlib.sha256(t.encode()).digest()
        raw = (d * ((_DIMS // len(d)) + 1))[: _DIMS]
        for i, b in enumerate(raw):
            acc[i] += (b - 128) / 128.0
    return [x / len(tokens) for x in acc]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


class InMemoryGraph:
    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []
        self._adj: dict[str, list[tuple[str, dict]]] = {}

    def upsert_node(self, *, id: str, type: str, name: str, properties: dict | None = None,
                    embedding: list[float] | None = None) -> None:
        existing = self._nodes.get(id, {})
        merged = {**existing, "id": id, "type": type, "name": name,
                  "properties": {**existing.get("properties", {}), **(properties or {})}}
        if embedding is not None:
            merged["embedding"] = embedding
        elif "embedding" not in merged:
            merged["embedding"] = stub_embed(f"{name} {type}")
        self._nodes[id] = merged
        self._adj.setdefault(id, [])

    def upsert_edge(self, *, from_id: str, to_id: str, type: str, properties: dict | None = None) -> None:
        if from_id not in self._nodes or to_id not in self._nodes:
            raise KeyError(f"unknown node: {from_id} -> {to_id}")
        edge = {"from": from_id, "to": to_id, "type": type, "properties": properties or {}}
        self._edges.append(edge)
        self._adj.setdefault(from_id, []).append((to_id, edge))
        self._adj.setdefault(to_id, []).append((from_id, edge))

    def vector_search(self, *, query: str, k: int = 5, type_filter: str | None = None) -> list[dict[str, Any]]:
        qvec = stub_embed(query)
        scored = []
        for n in self._nodes.values():
            if type_filter and n["type"] != type_filter:
                continue
            sim = cosine(qvec, n.get("embedding", []))
            scored.append((sim, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{**n, "similarity": s} for s, n in scored[:k]]

    def traverse(self, *, seed_ids: list[str], max_hops: int = 2) -> dict[str, Any]:
        visited: dict[str, int] = {nid: 0 for nid in seed_ids}
        edges_used: list[dict] = []
        frontier = list(seed_ids)
        for hop in range(max_hops):
            next_frontier: list[str] = []
            for nid in frontier:
                for neighbor, edge in self._adj.get(nid, []):
                    if neighbor not in visited:
                        visited[neighbor] = hop + 1
                        next_frontier.append(neighbor)
                    if edge not in edges_used:
                        edges_used.append(edge)
            frontier = next_frontier
            if not frontier:
                break
        return {"nodes": [{**self._nodes[nid], "hops_from_seed": d} for nid, d in visited.items()],
                "edges": edges_used}

    def count_by_type(self, *, type: str, where: dict[str, Any] | None = None) -> int:
        n = 0
        for node in self._nodes.values():
            if node["type"] != type:
                continue
            if where and not all(node["properties"].get(k) == v for k, v in where.items()):
                continue
            n += 1
        return n

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        return len(self._edges)
