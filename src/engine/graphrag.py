"""GraphRAG engine: link → plan → traverse → aggregate → synthesize.

Three retrieval modes share one response schema so they can be compared in
``eval``:

* ``graph``  — entity linking + planned traversal (this system);
* ``vector`` — TF-IDF passage retrieval over the same corpus (baseline);
* ``hybrid`` — vector seeds added to the linked seeds, then traversal.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from loguru import logger

from src.api.schemas import (
    CitedNode,
    GraphAnswer,
    PlanStep,
    QueryKind,
    QueryMetrics,
    QueryMode,
    SubgraphOut,
)
from src.engine.linking import EntityLinker, Mention, normalize
from src.engine.planner import Constraint, QueryPlan, Step, build_plan, resolve_direction
from src.engine.synthesis import Synthesizer, TemplateSynthesizer
from src.engine.vector import Passage, VectorIndex
from src.graph.base import Edge, GraphStore, Node
from src.observability import RequestMetrics, new_trace_id


class EngineError(RuntimeError):
    """Base class for engine failures surfaced as HTTP 4xx/5xx."""


class UnsupportedQuestionError(EngineError):
    """The planner could not map the question to graph operations."""


@dataclass
class Execution:
    """Intermediate result of running a plan."""

    answer_ids: list[str]
    seeds: list[str]
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    path_sentences: list[str] = field(default_factory=list)
    hops: dict[str, int] = field(default_factory=dict)
    resolved: list[PlanStep] = field(default_factory=list)


def _edge_sentence(store: GraphStore, e: Edge) -> str:
    a, b = store.get_node(e["from_id"]), store.get_node(e["to_id"])
    an = a["name"] if a else e["from_id"]
    bn = b["name"] if b else e["to_id"]
    return f"{an} [{e['from_id']}] -{e['type']}-> {bn} [{e['to_id']}]."


_FILTERABLE_PROPS = ("jurisdiction", "sector", "category", "role", "hq_state")


def _property_match(n: Node, text: str) -> bool:
    """True when some filterable string property of ``n`` appears in the question."""
    for key in _FILTERABLE_PROPS:
        val = n["properties"].get(key)
        if isinstance(val, str) and len(val) >= 2 and f" {normalize(val)} " in f" {text} ":
            return True
    return False


def _flip(direction: str) -> Literal["in", "out", "both"]:
    if direction == "in":
        return "out"
    if direction == "out":
        return "in"
    return "both"


def _node_out(n: Node, hops: int | None) -> dict[str, Any]:
    return {
        "id": n["id"],
        "type": n["type"],
        "name": n["name"],
        "hops": hops,
        "stale": bool(n["properties"].get("stale", False)),
    }


def _subgraph_out(exec_: Execution) -> SubgraphOut:
    nodes = [_node_out(n, exec_.hops.get(n["id"])) for n in exec_.nodes.values()]
    edges = [{"from": e["from_id"], "to": e["to_id"], "type": e["type"]} for e in exec_.edges]
    return SubgraphOut(nodes=nodes[:150], edges=edges[:300])


class GraphRAGEngine:
    """Query engine over any :class:`GraphStore`.

    Args:
        store: Populated graph.
        synthesizer: Defaults to the deterministic template.
        extra_passages: Filing chunks to add to the vector index (optional).
    """

    def __init__(
        self,
        store: GraphStore,
        synthesizer: Synthesizer | None = None,
        extra_passages: list[Passage] | None = None,
    ) -> None:
        self.store = store
        self.synthesizer: Synthesizer = synthesizer or TemplateSynthesizer()
        self.linker = EntityLinker(store)
        self._extra_passages = list(extra_passages or [])
        self._vector: VectorIndex | None = None

    # -- indexes -----------------------------------------------------------
    @property
    def vector_index(self) -> VectorIndex:
        """Lazily built passage index (rebuilt with :meth:`refresh`)."""
        if self._vector is None:
            self._vector = VectorIndex.from_store(self.store, self._extra_passages)
        return self._vector

    def refresh(self) -> None:
        """Rebuild alias table and vector index after the graph changed."""
        self.linker.rebuild(self.store)
        self._vector = None

    # -- execution ---------------------------------------------------------
    def _expand(self, frontier: set[str], step: Step, exec_: Execution, hop: int) -> set[str]:
        types = {n["type"] for nid in frontier if (n := self.store.get_node(nid)) is not None}
        direction = resolve_direction(step, types)
        exec_.resolved.append(PlanStep(edge_types=list(step.edge_types), direction=direction))
        nxt: set[str] = set()
        for nid in frontier:
            for et in step.edge_types:
                for other, edge in self.store.neighbors(nid, edge_type=et, direction=direction):
                    nxt.add(other["id"])
                    exec_.nodes.setdefault(other["id"], other)
                    exec_.hops.setdefault(other["id"], hop)
                    if edge not in exec_.edges:
                        exec_.edges.append(edge)
        return nxt

    def _run_constraint(self, c: Constraint, exec_: Execution) -> set[str]:
        seed = self.store.get_node(c.seed_id)
        if seed is None:
            return set()
        exec_.nodes.setdefault(seed["id"], seed)
        exec_.hops.setdefault(seed["id"], 0)
        frontier = {c.seed_id}
        hop = 0
        for step in c.steps:
            hop += 1
            frontier = self._expand(frontier, step, exec_, hop)
            if step.share:
                hop += 1
                back_dir: Literal["in", "out", "both", "auto"] = (
                    "auto" if step.direction == "auto" else _flip(step.direction)
                )
                back = Step(step.edge_types, back_dir)
                frontier = self._expand(frontier, back, exec_, hop) - {c.seed_id}
            if not frontier:
                break
        return frontier

    def _filter(self, ids: set[str], plan: QueryPlan) -> list[str]:
        out: list[str] = []
        for nid in sorted(ids):
            n = self.store.get_node(nid)
            if n is None:
                continue
            props = n["properties"]
            if plan.answer_type and n["type"] != plan.answer_type:
                continue
            if plan.role_filter and n["type"] == "Person" and props.get("role") != plan.role_filter:
                continue
            if plan.in_sp500_only and n["type"] == "Company" and not props.get("in_sp500"):
                continue
            if (
                plan.sector
                and n["type"] == "Company"
                and str(props.get("sector", "")).lower() != plan.sector
            ):
                continue
            out.append(nid)
        return out

    def execute(self, plan: QueryPlan) -> Execution:
        """Run every constraint and intersect (or union for a single seed)."""
        exec_ = Execution(answer_ids=[], seeds=plan.seed_ids)
        if not plan.constraints:
            return self._execute_seedless(plan, exec_)
        result: set[str] | None = None
        for c in plan.constraints:
            frontier = self._run_constraint(c, exec_)
            if not c.steps:
                # bare mention with no relation: the seed itself is the answer (lookup by name)
                frontier = {c.seed_id} if self.store.get_node(c.seed_id) else set()
            result = frontier if result is None else (result & frontier)
        ids = set(result or set())
        if any(c.steps for c in plan.constraints):
            ids -= set(plan.seed_ids)
        exec_.answer_ids = self._filter(ids, plan)
        exec_.path_sentences = [
            _edge_sentence(self.store, e)
            for e in exec_.edges
            if e["from_id"] in exec_.nodes and e["to_id"] in exec_.nodes
        ]
        return exec_

    def _execute_seedless(self, plan: QueryPlan, exec_: Execution) -> Execution:
        """No entity in the question: filter nodes of the answer type by properties/degree."""
        if plan.answer_type is None:
            return exec_
        text = normalize(plan.question)
        edge_types = tuple(et for st in plan.seedless_steps for et in st.edge_types)
        ids: set[str] = set()
        for n in self.store.nodes():
            if n["type"] != plan.answer_type:
                continue
            if plan.min_degree is not None:
                nbrs = [
                    e
                    for et in (edge_types or (None,))
                    for _o, e in self.store.neighbors(n["id"], edge_type=et)
                ]
                if len(nbrs) < plan.min_degree:
                    continue
                for _o, e in (
                    (o, e)
                    for et in (edge_types or (None,))
                    for o, e in self.store.neighbors(n["id"], edge_type=et)
                ):
                    exec_.nodes.setdefault(_o["id"], _o)
                    if e not in exec_.edges:
                        exec_.edges.append(e)
            elif not _property_match(n, text):
                continue
            ids.add(n["id"])
            exec_.nodes.setdefault(n["id"], n)
            exec_.hops.setdefault(n["id"], 0)
        exec_.answer_ids = self._filter(ids, plan)
        exec_.path_sentences = [_edge_sentence(self.store, e) for e in exec_.edges]
        return exec_

    # -- public API --------------------------------------------------------
    def answer_sync(
        self, question: str, *, k: int = 5, max_hops: int = 3, mode: QueryMode = "graph"
    ) -> GraphAnswer:
        """Answer a question in the given mode (blocking)."""
        trace_id = new_trace_id()
        metrics = RequestMetrics(name=f"query:{mode}", trace_id=trace_id)
        mentions = self.linker.link(question)
        if mode == "vector":
            answer = self._answer_vector(question, k, metrics)
        else:
            if mode == "hybrid":
                mentions = self._hybrid_mentions(question, mentions, k)
            answer = self._answer_graph(question, mentions, max_hops, metrics, mode)
        metrics.extra = {"kind": answer.classified_kind, "n_answer": len(answer.answer_ids)}
        metrics.finish()
        answer.metrics = QueryMetrics(
            trace_id=trace_id,
            latency_ms=metrics.latency_ms,
            tokens_in=metrics.tokens_in,
            tokens_out=metrics.tokens_out,
            cost_usd=metrics.cost_usd,
            llm_calls=metrics.llm_calls,
        )
        return answer

    async def answer(
        self, question: str, *, k: int = 5, max_hops: int = 3, mode: QueryMode = "graph"
    ) -> GraphAnswer:
        """Async wrapper: runs the CPU-bound engine in a worker thread."""
        return await asyncio.to_thread(
            self.answer_sync, question, k=k, max_hops=max_hops, mode=mode
        )

    def _hybrid_mentions(self, question: str, mentions: list[Mention], k: int) -> list[Mention]:
        if mentions:
            return mentions
        # No lexical link: fall back to vector seeds of graph nodes
        seeds = self.store.vector_search(query=question, k=min(k, 3))
        return [Mention(node_id=s["id"], alias=s["name"].lower(), start=0, end=0) for s in seeds]

    def _answer_graph(
        self,
        question: str,
        mentions: list[Mention],
        max_hops: int,
        metrics: RequestMetrics,
        mode: QueryMode,
    ) -> GraphAnswer:
        seed_types = {
            m.node_id: n["type"]
            for m in mentions
            if (n := self.store.get_node(m.node_id)) is not None
        }
        plan = build_plan(question, mentions, seed_types)
        if plan.n_hops > max_hops:
            plan.notes.append(f"plan needs {plan.n_hops} hops > max_hops={max_hops}; truncated")
            for c in plan.constraints:
                c.steps = c.steps[:max_hops]
        exec_ = self.execute(plan)
        count: int | None = len(exec_.answer_ids) if plan.aggregate else None
        prop_value: str | None = None
        answer_nodes = [
            n for nid in exec_.answer_ids if (n := self.store.get_node(nid)) is not None
        ]
        if plan.property_lookup and plan.constraints:
            seed = self.store.get_node(plan.constraints[0].seed_id)
            if seed is not None:
                raw = seed["properties"].get(plan.property_lookup)
                if raw is not None:
                    prop_value = f"{seed['name']} [{seed['id']}] {plan.property_lookup} = {raw}"
                    answer_nodes = [seed]
                    exec_.answer_ids = [seed["id"]]
        text = self.synthesizer.synthesize(
            question,
            answer_nodes,
            count=count,
            property_value=prop_value,
            path_sentences=exec_.path_sentences,
            metrics=metrics,
        )
        cited = [
            CitedNode(
                id=n["id"], type=n["type"], name=n["name"], hops_from_seed=exec_.hops.get(n["id"])
            )
            for n in sorted(
                exec_.nodes.values(), key=lambda x: (exec_.hops.get(x["id"], 99), x["id"])
            )
        ]
        kind = cast(QueryKind, plan.kind)
        logger.debug("plan={} answer_ids={}", plan.as_dict(), exec_.answer_ids)
        return GraphAnswer(
            question=question,
            mode=mode,
            answer=text,
            classified_kind=kind,
            answer_ids=exec_.answer_ids,
            count=count,
            seeds=plan.seed_ids,
            plan=exec_.resolved,
            cited_nodes=cited[:60],
            reasoning_path=[c.id for c in cited[:40]],
            subgraph=_subgraph_out(exec_),
            synthesizer=self.synthesizer.name,
            metrics=QueryMetrics(trace_id=metrics.trace_id, latency_ms=0),
        )

    def _answer_vector(self, question: str, k: int, metrics: RequestMetrics) -> GraphAnswer:
        plan = build_plan(question, [])  # only for answer type / aggregate detection
        retrieved = self.vector_index.query(question, k=k)
        ids = retrieved.entity_ids
        filtered = self._filter(ids, plan) if plan.answer_type else sorted(ids)
        count = len(filtered) if plan.aggregate else None
        nodes = [n for nid in filtered if (n := self.store.get_node(nid)) is not None]
        text = self.synthesizer.synthesize(
            question,
            nodes,
            count=count,
            property_value=None,
            path_sentences=[p.text[:200] for _, p in retrieved.passages],
            metrics=metrics,
        )
        cited = [CitedNode(id=n["id"], type=n["type"], name=n["name"]) for n in nodes[:60]]
        kind: QueryKind = "aggregation" if plan.aggregate else "hybrid"
        return GraphAnswer(
            question=question,
            mode="vector",
            answer=text,
            classified_kind=kind,
            answer_ids=filtered,
            count=count,
            seeds=[p.id for _, p in retrieved.passages],
            cited_nodes=cited,
            reasoning_path=[p.id for _, p in retrieved.passages],
            subgraph=SubgraphOut(nodes=[_node_out(n, None) for n in nodes[:40]], edges=[]),
            synthesizer=self.synthesizer.name,
            metrics=QueryMetrics(trace_id=metrics.trace_id, latency_ms=0),
        )
