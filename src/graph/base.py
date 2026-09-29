"""Graph store contract shared by the in-memory, NetworkX and Neo4j backends."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from src.embeddings.base import Embedder, cosine

Direction = Literal["out", "in", "both"]


class Node(TypedDict):
    """Graph node as exchanged between layers.

    ``updated_at`` is the ISO date the node was last merged; ``source_filing_date``
    is the filing date of the evidence (used by the staleness detector).
    """

    id: str
    type: str
    name: str
    properties: dict[str, Any]
    embedding: list[float]
    updated_at: str
    source_filing_date: str | None


class Edge(TypedDict):
    """Directed, typed edge."""

    from_id: str
    to_id: str
    type: str
    properties: dict[str, Any]


class ScoredNode(Node):
    """Node returned by vector search."""

    similarity: float


class HopNode(Node):
    """Node returned by traversal."""

    hops_from_seed: int


class Subgraph(TypedDict):
    """Result of a traversal."""

    nodes: list[HopNode]
    edges: list[Edge]


class UnknownNodeError(KeyError):
    """Raised when an edge references a node that does not exist."""


def today_iso() -> str:
    """UTC date as ``YYYY-MM-DD``."""
    return datetime.now(UTC).date().isoformat()


@runtime_checkable
class GraphStore(Protocol):
    """Knowledge-graph persistence contract.

    Every backend guarantees MERGE semantics: upserting an existing node or
    an existing ``(from, to, type)`` edge updates properties instead of
    duplicating.
    """

    embedder: Embedder

    def upsert_node(
        self,
        *,
        id: str,
        type: str,
        name: str,
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        source_filing_date: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        """Create or update a node."""
        ...

    def upsert_edge(
        self, *, from_id: str, to_id: str, type: str, properties: dict[str, Any] | None = None
    ) -> None:
        """Create or update an edge; raises :class:`UnknownNodeError`."""
        ...

    def get_node(self, node_id: str) -> Node | None:
        """Fetch one node or ``None``."""
        ...

    def nodes(self) -> list[Node]:
        """All nodes."""
        ...

    def edges(self) -> list[Edge]:
        """All edges."""
        ...

    def neighbors(
        self, node_id: str, *, edge_type: str | None = None, direction: Direction = "both"
    ) -> list[tuple[Node, Edge]]:
        """Adjacent nodes with the connecting edge."""
        ...

    def vector_search(
        self, *, query: str, k: int = 5, type_filter: str | None = None
    ) -> list[ScoredNode]:
        """Top-k nodes by cosine similarity of the query embedding."""
        ...

    def traverse(
        self, *, seed_ids: list[str], max_hops: int = 2, edge_types: list[str] | None = None
    ) -> Subgraph:
        """Breadth-first expansion from ``seed_ids``."""
        ...

    def count_by_type(self, *, type: str, where: dict[str, Any] | None = None) -> int:
        """Count nodes of a type matching property equalities."""
        ...

    def clear(self) -> None:
        """Delete everything."""
        ...

    @property
    def n_nodes(self) -> int:
        """Node count."""
        ...

    @property
    def n_edges(self) -> int:
        """Edge count."""
        ...


class LocalGraphMixin:
    """Generic ``vector_search``/``traverse``/``count_by_type`` for local stores.

    Subclasses provide ``nodes``, ``get_node``, ``neighbors`` and ``embedder``.
    """

    embedder: Embedder

    def nodes(self) -> list[Node]:  # pragma: no cover - abstract
        raise NotImplementedError

    def get_node(self, node_id: str) -> Node | None:  # pragma: no cover - abstract
        raise NotImplementedError

    def neighbors(
        self, node_id: str, *, edge_type: str | None = None, direction: Direction = "both"
    ) -> list[tuple[Node, Edge]]:  # pragma: no cover - abstract
        raise NotImplementedError

    def vector_search(
        self, *, query: str, k: int = 5, type_filter: str | None = None
    ) -> list[ScoredNode]:
        """Exact cosine search over all node embeddings (fine below ~1e5 nodes)."""
        qvec = self.embedder.embed_one(query)
        scored: list[tuple[float, Node]] = []
        for n in self.nodes():
            if type_filter and n["type"] != type_filter:
                continue
            scored.append((cosine(qvec, n["embedding"]), n))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [ScoredNode(**n, similarity=round(s, 6)) for s, n in scored[:k]]

    def traverse(
        self, *, seed_ids: list[str], max_hops: int = 2, edge_types: list[str] | None = None
    ) -> Subgraph:
        """BFS from the seeds; unknown seeds are ignored."""
        allowed = set(edge_types) if edge_types else None
        visited: dict[str, int] = {s: 0 for s in seed_ids if self.get_node(s) is not None}
        edges_used: list[Edge] = []
        seen_edges: set[tuple[str, str, str]] = set()
        frontier = list(visited)
        for hop in range(max_hops):
            next_frontier: list[str] = []
            for nid in frontier:
                for neighbor, edge in self.neighbors(nid):
                    if allowed is not None and edge["type"] not in allowed:
                        continue
                    key = (edge["from_id"], edge["to_id"], edge["type"])
                    if key not in seen_edges:
                        seen_edges.add(key)
                        edges_used.append(edge)
                    if neighbor["id"] not in visited:
                        visited[neighbor["id"]] = hop + 1
                        next_frontier.append(neighbor["id"])
            frontier = next_frontier
            if not frontier:
                break
        out_nodes: list[HopNode] = []
        for nid, d in visited.items():
            node = self.get_node(nid)
            if node is not None:
                out_nodes.append(HopNode(**node, hops_from_seed=d))
        return Subgraph(nodes=out_nodes, edges=edges_used)

    def count_by_type(self, *, type: str, where: dict[str, Any] | None = None) -> int:
        """Count nodes of ``type`` whose properties equal every ``where`` pair."""
        n = 0
        for node in self.nodes():
            if node["type"] != type:
                continue
            if where and not all(node["properties"].get(k) == v for k, v in where.items()):
                continue
            n += 1
        return n
