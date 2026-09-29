"""Persistent NetworkX-backed store (JSON on disk) for running without Docker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger

from src.embeddings.base import Embedder
from src.embeddings.hashing import HashingEmbedder
from src.graph.base import Direction, Edge, LocalGraphMixin, Node, UnknownNodeError, today_iso


class NetworkXGraphStore(LocalGraphMixin):
    """``nx.MultiDiGraph`` store with :meth:`save`/:meth:`load` to JSON.

    Handles tens of thousands of nodes comfortably; the whole graph is kept in
    memory and flushed atomically to ``path`` on :meth:`save`.

    Args:
        path: JSON file; loaded on construction when it exists.
        embedder: Embedder for new nodes and vector search.
    """

    def __init__(self, path: str | Path | None = None, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.path = Path(path) if path else None
        self._g: Any = nx.MultiDiGraph()
        if self.path and self.path.exists():
            self.load(self.path)

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
        """MERGE a node."""
        existing: dict[str, Any] | None = self._g.nodes[id] if self._g.has_node(id) else None
        merged_props = {**(existing["properties"] if existing else {}), **(properties or {})}
        if embedding is not None:
            emb = embedding
        elif existing is not None:
            emb = existing["embedding"]
        else:
            emb = self.embedder.embed_one(f"{name} {type} {merged_props.get('description', '')}")
        sfd = source_filing_date or (existing["source_filing_date"] if existing else None)
        self._g.add_node(
            id,
            type=type,
            name=name,
            properties=merged_props,
            embedding=emb,
            updated_at=updated_at or today_iso(),
            source_filing_date=sfd,
        )

    def upsert_edge(
        self, *, from_id: str, to_id: str, type: str, properties: dict[str, Any] | None = None
    ) -> None:
        """MERGE an edge keyed by ``(from, to, type)`` (the multigraph key)."""
        if not self._g.has_node(from_id) or not self._g.has_node(to_id):
            raise UnknownNodeError(f"unknown node in edge {from_id}-{type}->{to_id}")
        if self._g.has_edge(from_id, to_id, key=type):
            self._g[from_id][to_id][type]["properties"].update(properties or {})
            return
        self._g.add_edge(from_id, to_id, key=type, properties=properties or {})

    def _node(self, node_id: str) -> Node:
        d = self._g.nodes[node_id]
        return Node(
            id=node_id,
            type=d["type"],
            name=d["name"],
            properties=d["properties"],
            embedding=d["embedding"],
            updated_at=d["updated_at"],
            source_filing_date=d["source_filing_date"],
        )

    def get_node(self, node_id: str) -> Node | None:
        """Fetch one node."""
        return self._node(node_id) if self._g.has_node(node_id) else None

    def nodes(self) -> list[Node]:
        """All nodes."""
        return [self._node(n) for n in self._g.nodes]

    def edges(self) -> list[Edge]:
        """All edges."""
        return [
            Edge(from_id=u, to_id=v, type=k, properties=d["properties"])
            for u, v, k, d in self._g.edges(keys=True, data=True)
        ]

    def neighbors(
        self, node_id: str, *, edge_type: str | None = None, direction: Direction = "both"
    ) -> list[tuple[Node, Edge]]:
        """Adjacent nodes with edges."""
        if not self._g.has_node(node_id):
            return []
        out: list[tuple[Node, Edge]] = []
        if direction in ("out", "both"):
            for _, v, k, d in self._g.out_edges(node_id, keys=True, data=True):
                if edge_type and k != edge_type:
                    continue
                out.append(
                    (
                        self._node(v),
                        Edge(from_id=node_id, to_id=v, type=k, properties=d["properties"]),
                    )
                )
        if direction in ("in", "both"):
            for u, _, k, d in self._g.in_edges(node_id, keys=True, data=True):
                if edge_type and k != edge_type:
                    continue
                out.append(
                    (
                        self._node(u),
                        Edge(from_id=u, to_id=node_id, type=k, properties=d["properties"]),
                    )
                )
        return out

    def clear(self) -> None:
        """Remove everything."""
        self._g.clear()

    @property
    def n_nodes(self) -> int:
        """Node count."""
        return int(self._g.number_of_nodes())

    @property
    def n_edges(self) -> int:
        """Edge count."""
        return int(self._g.number_of_edges())

    def save(self, path: str | Path | None = None) -> Path:
        """Write the graph as JSON (atomic rename) and return the path."""
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path configured for NetworkXGraphStore.save")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"nodes": self.nodes(), "edges": self.edges()}
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
        logger.info("graph saved to {} ({} nodes, {} edges)", target, self.n_nodes, self.n_edges)
        return target

    def load(self, path: str | Path) -> None:
        """Replace the current graph with the JSON at ``path``."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.clear()
        for n in payload["nodes"]:
            self.upsert_node(
                id=n["id"],
                type=n["type"],
                name=n["name"],
                properties=n["properties"],
                embedding=n["embedding"],
                source_filing_date=n.get("source_filing_date"),
                updated_at=n.get("updated_at"),
            )
        for e in payload["edges"]:
            self.upsert_edge(
                from_id=e["from_id"], to_id=e["to_id"], type=e["type"], properties=e["properties"]
            )
        logger.info("graph loaded from {} ({} nodes, {} edges)", path, self.n_nodes, self.n_edges)
