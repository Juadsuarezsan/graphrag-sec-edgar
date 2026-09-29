"""In-memory graph store used by tests, the demo and the eval harness."""

from __future__ import annotations

from typing import Any

from src.embeddings.base import Embedder
from src.embeddings.hashing import HashingEmbedder
from src.graph.base import Direction, Edge, LocalGraphMixin, Node, UnknownNodeError, today_iso


class InMemoryGraph(LocalGraphMixin):
    """Dictionary-backed store implementing :class:`~src.graph.base.GraphStore`.

    Args:
        embedder: Embedder used when a node is upserted without an embedding
            and for ``vector_search``. Defaults to the hashing embedder.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self._nodes: dict[str, Node] = {}
        self._edges: dict[tuple[str, str, str], Edge] = {}
        self._out: dict[str, list[tuple[str, str, str]]] = {}
        self._in: dict[str, list[tuple[str, str, str]]] = {}

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
        """MERGE a node: properties are shallow-merged, embedding replaced if given."""
        existing = self._nodes.get(id)
        merged_props = {**(existing["properties"] if existing else {}), **(properties or {})}
        if embedding is not None:
            emb = embedding
        elif existing is not None:
            emb = existing["embedding"]
        else:
            emb = self.embedder.embed_one(f"{name} {type} {merged_props.get('description', '')}")
        sfd = source_filing_date or (existing["source_filing_date"] if existing else None)
        self._nodes[id] = Node(
            id=id,
            type=type,
            name=name,
            properties=merged_props,
            embedding=emb,
            updated_at=updated_at or today_iso(),
            source_filing_date=sfd,
        )
        self._out.setdefault(id, [])
        self._in.setdefault(id, [])

    def upsert_edge(
        self, *, from_id: str, to_id: str, type: str, properties: dict[str, Any] | None = None
    ) -> None:
        """MERGE an edge keyed by ``(from, to, type)``."""
        if from_id not in self._nodes or to_id not in self._nodes:
            raise UnknownNodeError(f"unknown node in edge {from_id}-{type}->{to_id}")
        key = (from_id, to_id, type)
        if key in self._edges:
            self._edges[key]["properties"].update(properties or {})
            return
        self._edges[key] = Edge(
            from_id=from_id, to_id=to_id, type=type, properties=properties or {}
        )
        self._out[from_id].append(key)
        self._in[to_id].append(key)

    def get_node(self, node_id: str) -> Node | None:
        """Fetch one node."""
        return self._nodes.get(node_id)

    def nodes(self) -> list[Node]:
        """All nodes in insertion order."""
        return list(self._nodes.values())

    def edges(self) -> list[Edge]:
        """All edges in insertion order."""
        return list(self._edges.values())

    def neighbors(
        self, node_id: str, *, edge_type: str | None = None, direction: Direction = "both"
    ) -> list[tuple[Node, Edge]]:
        """Adjacent nodes; ``direction`` is relative to ``node_id``."""
        keys: list[tuple[str, str, str]] = []
        if direction in ("out", "both"):
            keys.extend(self._out.get(node_id, []))
        if direction in ("in", "both"):
            keys.extend(self._in.get(node_id, []))
        out: list[tuple[Node, Edge]] = []
        for key in keys:
            edge = self._edges[key]
            if edge_type and edge["type"] != edge_type:
                continue
            other = edge["to_id"] if edge["from_id"] == node_id else edge["from_id"]
            out.append((self._nodes[other], edge))
        return out

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._out.clear()
        self._in.clear()

    @property
    def n_nodes(self) -> int:
        """Node count."""
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        """Edge count."""
        return len(self._edges)
