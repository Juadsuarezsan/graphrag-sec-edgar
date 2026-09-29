"""Neo4j 5 backend: Cypher MERGE, native vector index, variable-length traversal.

The driver is injected so contract tests can run against a recorded fake;
integration against a live Neo4j is pending a Docker daemon (see README).
"""

from __future__ import annotations

from typing import Any, Protocol

from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.embeddings.base import Embedder
from src.embeddings.hashing import HashingEmbedder
from src.graph.base import (
    Direction,
    Edge,
    HopNode,
    Node,
    ScoredNode,
    Subgraph,
    UnknownNodeError,
    today_iso,
)

VECTOR_INDEX = "entity_embedding"
NODE_LABEL = "Entity"


class _Driver(Protocol):
    """Subset of ``neo4j.Driver`` used by the store (lets tests inject a fake)."""

    def session(self, *, database: str | None = None) -> Any: ...

    def close(self) -> None: ...


class Neo4jTransientError(RuntimeError):
    """Wraps driver transient failures so retries stay driver-agnostic."""


class Neo4jGraphStore:
    """Graph store on Neo4j 5.13+.

    Nodes carry a generic ``Entity`` label plus their type as a second label,
    so ``MATCH (n:Company)`` and the shared vector index both work.

    Args:
        driver: ``neo4j.Driver`` (or a compatible fake).
        embedder: Embedder for new nodes and query vectors.
        database: Neo4j database name.
    """

    def __init__(
        self, driver: _Driver, embedder: Embedder | None = None, database: str = "neo4j"
    ) -> None:
        self._driver = driver
        self._db = database
        self.embedder = embedder or HashingEmbedder()

    @classmethod
    def from_settings(
        cls, uri: str, user: str, password: str, embedder: Embedder | None = None
    ) -> Neo4jGraphStore:
        """Build a store from connection parameters (imports the driver lazily)."""
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10.0)
        return cls(driver, embedder=embedder)

    @retry(
        retry=retry_if_exception_type(Neo4jTransientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, max=2),
        reraise=True,
    )
    def _run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            with self._driver.session(database=self._db) as session:
                result = session.run(query, **params)
                return [dict(r) for r in result]
        except (OSError, TimeoutError) as exc:
            raise Neo4jTransientError(str(exc)) from exc

    def ensure_schema(self) -> None:
        """Create the uniqueness constraint and the vector index (idempotent)."""
        self._run(
            f"CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (n:{NODE_LABEL}) REQUIRE n.id IS UNIQUE"
        )
        self._run(
            f"CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS FOR (n:{NODE_LABEL}) "
            "ON (n.embedding) OPTIONS {indexConfig: {`vector.dimensions`: $dims, "
            "`vector.similarity_function`: 'cosine'}}",
            dims=self.embedder.dims,
        )
        logger.info("neo4j schema ensured (dims={})", self.embedder.dims)

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
        """``MERGE`` on ``id``; properties are flattened under ``p_`` keys."""
        emb = embedding or self.embedder.embed_one(f"{name} {type}")
        props = {f"p_{k}": v for k, v in (properties or {}).items()}
        self._run(
            f"MERGE (n:{NODE_LABEL} {{id: $id}}) "
            f"SET n:{type}, n.type = $type, n.name = $name, n += $props, "
            "n.updated_at = $updated_at, "
            "n.source_filing_date = coalesce($sfd, n.source_filing_date), "
            "n.embedding = CASE WHEN $has_emb THEN $embedding ELSE coalesce(n.embedding, $embedding) END",
            id=id,
            type=type,
            name=name,
            props=props,
            updated_at=updated_at or today_iso(),
            sfd=source_filing_date,
            has_emb=embedding is not None,
            embedding=emb,
        )

    def upsert_edge(
        self, *, from_id: str, to_id: str, type: str, properties: dict[str, Any] | None = None
    ) -> None:
        """``MERGE`` a typed relationship between two existing nodes."""
        rows = self._run(
            f"MATCH (a:{NODE_LABEL} {{id: $from_id}}), (b:{NODE_LABEL} {{id: $to_id}}) "
            f"MERGE (a)-[r:{type}]->(b) SET r += $props RETURN count(r) AS n",
            from_id=from_id,
            to_id=to_id,
            props=properties or {},
        )
        if not rows or int(rows[0].get("n", 0)) == 0:
            raise UnknownNodeError(f"unknown node in edge {from_id}-{type}->{to_id}")

    @staticmethod
    def _to_node(raw: dict[str, Any]) -> Node:
        props = {k[2:]: v for k, v in raw.items() if k.startswith("p_")}
        return Node(
            id=str(raw["id"]),
            type=str(raw.get("type", "")),
            name=str(raw.get("name", "")),
            properties=props,
            embedding=list(raw.get("embedding") or []),
            updated_at=str(raw.get("updated_at", "")),
            source_filing_date=raw.get("source_filing_date"),
        )

    def get_node(self, node_id: str) -> Node | None:
        """Fetch one node."""
        rows = self._run(f"MATCH (n:{NODE_LABEL} {{id: $id}}) RETURN n", id=node_id)
        return self._to_node(dict(rows[0]["n"])) if rows else None

    def nodes(self) -> list[Node]:
        """All nodes."""
        return [self._to_node(dict(r["n"])) for r in self._run(f"MATCH (n:{NODE_LABEL}) RETURN n")]

    def edges(self) -> list[Edge]:
        """All edges."""
        rows = self._run(
            f"MATCH (a:{NODE_LABEL})-[r]->(b:{NODE_LABEL}) "
            "RETURN a.id AS from_id, b.id AS to_id, type(r) AS type, properties(r) AS props"
        )
        return [
            Edge(
                from_id=r["from_id"], to_id=r["to_id"], type=r["type"], properties=dict(r["props"])
            )
            for r in rows
        ]

    def neighbors(
        self, node_id: str, *, edge_type: str | None = None, direction: Direction = "both"
    ) -> list[tuple[Node, Edge]]:
        """Adjacent nodes via a direction-aware ``MATCH``."""
        rel = f"[r:{edge_type}]" if edge_type else "[r]"
        pattern = {
            "out": f"(a)-{rel}->(m)",
            "in": f"(a)<-{rel}-(m)",
            "both": f"(a)-{rel}-(m)",
        }[direction]
        rows = self._run(
            f"MATCH (a:{NODE_LABEL} {{id: $id}}) MATCH {pattern} "
            "RETURN m, startNode(r).id AS from_id, endNode(r).id AS to_id, type(r) AS type, "
            "properties(r) AS props",
            id=node_id,
        )
        return [
            (
                self._to_node(dict(r["m"])),
                Edge(
                    from_id=r["from_id"],
                    to_id=r["to_id"],
                    type=r["type"],
                    properties=dict(r["props"]),
                ),
            )
            for r in rows
        ]

    def vector_search(
        self, *, query: str, k: int = 5, type_filter: str | None = None
    ) -> list[ScoredNode]:
        """``db.index.vector.queryNodes`` on the native index."""
        qvec = self.embedder.embed_one(query)
        where = "WHERE node.type = $type_filter " if type_filter else ""
        rows = self._run(
            f"CALL db.index.vector.queryNodes($index, $k_over, $vec) YIELD node, score "
            f"{where}RETURN node, score ORDER BY score DESC LIMIT $k",
            index=VECTOR_INDEX,
            k_over=k * 4 if type_filter else k,
            k=k,
            vec=qvec,
            type_filter=type_filter,
        )
        return [
            ScoredNode(**self._to_node(dict(r["node"])), similarity=float(r["score"])) for r in rows
        ]

    def traverse(
        self, *, seed_ids: list[str], max_hops: int = 2, edge_types: list[str] | None = None
    ) -> Subgraph:
        """Variable-length expansion ``(seed)-[*1..max_hops]-(m)`` with shortest hop count."""
        rel = f"[:{'|'.join(edge_types)}*1..{max_hops}]" if edge_types else f"[*1..{max_hops}]"
        rows = self._run(
            f"MATCH (s:{NODE_LABEL}) WHERE s.id IN $seeds "
            f"MATCH p = (s)-{rel}-(m:{NODE_LABEL}) "
            "WITH m, min(length(p)) AS hops, collect(p) AS paths "
            "RETURN m, hops, [p IN paths | [r IN relationships(p) | "
            "{from_id: startNode(r).id, to_id: endNode(r).id, type: type(r), props: properties(r)}]] AS rels",
            seeds=seed_ids,
            max_hops=max_hops,
        )
        nodes: dict[str, HopNode] = {}
        for sid in seed_ids:
            seed = self.get_node(sid)
            if seed is not None:
                nodes[sid] = HopNode(**seed, hops_from_seed=0)
        edges: dict[tuple[str, str, str], Edge] = {}
        for r in rows:
            node = self._to_node(dict(r["m"]))
            if node["id"] not in nodes:
                nodes[node["id"]] = HopNode(**node, hops_from_seed=int(r["hops"]))
            for path in r["rels"]:
                for e in path:
                    key = (e["from_id"], e["to_id"], e["type"])
                    edges.setdefault(
                        key,
                        Edge(
                            from_id=e["from_id"],
                            to_id=e["to_id"],
                            type=e["type"],
                            properties=dict(e["props"]),
                        ),
                    )
        return Subgraph(nodes=list(nodes.values()), edges=list(edges.values()))

    def count_by_type(self, *, type: str, where: dict[str, Any] | None = None) -> int:
        """``MATCH (n:Type) WHERE ... RETURN count(n)``."""
        conds = " AND ".join(f"n.p_{k} = $w_{k}" for k in (where or {}))
        params = {f"w_{k}": v for k, v in (where or {}).items()}
        clause = f"WHERE {conds} " if conds else ""
        rows = self._run(f"MATCH (n:{type}) {clause}RETURN count(n) AS n", **params)
        return int(rows[0]["n"]) if rows else 0

    def clear(self) -> None:
        """``MATCH (n) DETACH DELETE n``."""
        self._run(f"MATCH (n:{NODE_LABEL}) DETACH DELETE n")

    @property
    def n_nodes(self) -> int:
        """Node count."""
        rows = self._run(f"MATCH (n:{NODE_LABEL}) RETURN count(n) AS n")
        return int(rows[0]["n"]) if rows else 0

    @property
    def n_edges(self) -> int:
        """Edge count."""
        rows = self._run(f"MATCH (:{NODE_LABEL})-[r]->(:{NODE_LABEL}) RETURN count(r) AS n")
        return int(rows[0]["n"]) if rows else 0

    def close(self) -> None:
        """Close the underlying driver."""
        self._driver.close()
