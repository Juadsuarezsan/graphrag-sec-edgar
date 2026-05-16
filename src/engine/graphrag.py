"""GraphRAG engine — routes queries, runs traversal/aggregation, synthesizes."""
from __future__ import annotations

import re
import time
from typing import Any

from src.api.schemas import CitedNode, GraphAnswer
from src.graph.store import InMemoryGraph
from src.router.classifier import classify


def _label_for_aggregation(q: str) -> str | None:
    q = q.lower()
    if "compan" in q or "s&p" in q:
        return "Company"
    if "people" in q or "ceo" in q or "person" in q:
        return "Person"
    if "risk" in q:
        return "Risk"
    return None


def _seed_nodes(g: InMemoryGraph, question: str, k: int = 5) -> list[dict]:
    return g.vector_search(query=question, k=k)


def _synthesize(question: str, nodes: list[dict], edges: list[dict], kind: str, count: int | None = None) -> str:
    if kind == "aggregation":
        return f"count = {count}" if count is not None else "aggregation: 0"
    if not nodes:
        return "No matching nodes found in the knowledge graph."
    names = [n["name"] for n in nodes[:5]]
    rel_part = ""
    if edges:
        rel_part = " · relationships: " + ", ".join(
            f"{e['from']}-{e['type']}->{e['to']}" for e in edges[:4]
        )
    return f"Cited: {', '.join(names)}.{rel_part}"


async def answer(g: InMemoryGraph, question: str, *, k: int = 5, max_hops: int = 2) -> GraphAnswer:
    t0 = time.perf_counter()
    kind = classify(question)

    if kind == "aggregation":
        label = _label_for_aggregation(question)
        count = g.count_by_type(type=label) if label else 0
        return GraphAnswer(
            question=question, answer=_synthesize(question, [], [], "aggregation", count=count),
            classified_kind="aggregation",
            reasoning_path=[f"aggregate:{label or '?'}"],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    seeds = _seed_nodes(g, question, k=k)
    if not seeds:
        return GraphAnswer(question=question, answer="No matches.",
                              classified_kind=kind,
                              latency_ms=int((time.perf_counter() - t0) * 1000))

    if kind == "lookup":
        cited = [CitedNode(id=n["id"], type=n["type"], name=n["name"],
                            similarity=n.get("similarity")) for n in seeds]
        return GraphAnswer(
            question=question, answer=_synthesize(question, seeds, [], "lookup"),
            classified_kind="lookup", cited_nodes=cited,
            reasoning_path=[n["id"] for n in seeds],
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )

    seed_ids = [n["id"] for n in seeds]
    sub = g.traverse(seed_ids=seed_ids, max_hops=max_hops)
    nodes = sub["nodes"]
    edges = sub["edges"]
    cited = [CitedNode(id=n["id"], type=n["type"], name=n["name"],
                        hops_from_seed=n.get("hops_from_seed", 0)) for n in nodes]
    return GraphAnswer(
        question=question,
        answer=_synthesize(question, nodes, edges, kind),
        classified_kind=kind if kind != "hybrid" else "multi_hop",
        cited_nodes=cited,
        reasoning_path=[n["id"] for n in sorted(nodes, key=lambda x: x.get("hops_from_seed", 0))],
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
