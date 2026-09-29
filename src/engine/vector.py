"""Vector RAG baseline over the same corpus as the graph.

Every node is rendered as a passage ("Apple Inc. (Company) ... is supplied by
TSMC; competes with Samsung ...") and filing chunks are added as passages
too. Retrieval is cosine over TF-IDF hashed vectors; the "answer" is the set
of entities named in the top-k passages, so it can be scored with the same
hit metric as GraphRAG. No LLM is involved (declared in the results).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.embeddings.base import Embedder, cosine
from src.embeddings.tfidf import TfidfHashingEmbedder
from src.graph.base import GraphStore

_EDGE_TEMPLATES: dict[str, tuple[str, str]] = {
    "CEO_OF": ("{a} is the CEO of {b}.", "{b} is led by CEO {a}."),
    "DIRECTOR_OF": ("{a} sits on the board of {b} as a director.", "{b} has {a} on its board."),
    "SUBSIDIARY_OF": ("{a} is a subsidiary of {b}.", "{b} owns the subsidiary {a}."),
    "SUPPLIED_BY": ("{a} is supplied by {b}; {a} depends on {b}.", "{b} is a supplier of {a}."),
    "COMPETES_WITH": ("{a} competes with {b}.", "{b} competes with {a}."),
    "MENTIONS_RISK": ("{a} mentions the risk {b} in its 10-K.", "{b} is a risk mentioned by {a}."),
    "OFFERS_PRODUCT": ("{a} offers the product {b}.", "{b} is a product of {a}."),
    "OPERATES_IN": ("{a} operates in the {b} market.", "{b} market participant: {a}."),
}


@dataclass
class Passage:
    """A retrievable text unit with the entity ids it explicitly names."""

    id: str
    text: str
    entity_ids: set[str] = field(default_factory=set)
    embedding: list[float] = field(default_factory=list)


@dataclass
class Retrieved:
    """Result of a vector query."""

    passages: list[tuple[float, Passage]]

    @property
    def entity_ids(self) -> set[str]:
        """Union of entities named in the retrieved passages."""
        out: set[str] = set()
        for _, p in self.passages:
            out |= p.entity_ids
        return out


def node_passages(store: GraphStore) -> list[Passage]:
    """Render every node (with its edges) as one passage."""
    passages: list[Passage] = []
    for node in store.nodes():
        props = node["properties"]
        head = f"{node['name']} ({node['type']})."
        details = [
            f"{k}: {v}"
            for k, v in props.items()
            if k not in ("aliases", "description", "embedding") and v not in (None, "")
        ]
        sentences: list[str] = []
        ids = {node["id"]}
        for other, edge in store.neighbors(node["id"]):
            tpl_out, tpl_in = _EDGE_TEMPLATES.get(
                edge["type"], ("{a} {rel} {b}.", "{b} {rel} {a}.")
            )
            if edge["from_id"] == node["id"]:
                sentences.append(tpl_out.format(a=node["name"], b=other["name"], rel=edge["type"]))
            else:
                sentences.append(tpl_in.format(a=other["name"], b=node["name"], rel=edge["type"]))
            ids.add(other["id"])
        text = " ".join(
            [head, "; ".join(details) + "." if details else "", " ".join(sentences)]
        ).strip()
        passages.append(Passage(id=f"node:{node['id']}", text=text, entity_ids=ids))
    return passages


class VectorIndex:
    """Fit-once, query-many index of passages."""

    def __init__(self, passages: list[Passage], embedder: Embedder | None = None) -> None:
        self.embedder: Embedder = embedder or TfidfHashingEmbedder()
        if isinstance(self.embedder, TfidfHashingEmbedder) and not self.embedder.is_fitted:
            self.embedder.fit([p.text for p in passages])
        vectors = self.embedder.embed([p.text for p in passages]) if passages else []
        for p, v in zip(passages, vectors, strict=True):
            p.embedding = v
        self.passages = passages

    @classmethod
    def from_store(
        cls,
        store: GraphStore,
        extra_passages: list[Passage] | None = None,
        embedder: Embedder | None = None,
    ) -> VectorIndex:
        """Index all node passages plus optional filing chunks."""
        return cls(node_passages(store) + list(extra_passages or []), embedder=embedder)

    def query(self, question: str, k: int = 5) -> Retrieved:
        """Top-k passages by cosine similarity (ties broken by passage id)."""
        qvec = self.embedder.embed_one(question)
        scored = sorted(
            ((cosine(qvec, p.embedding), p) for p in self.passages),
            key=lambda sp: (-sp[0], sp[1].id),
        )
        return Retrieved(passages=[(round(s, 6), p) for s, p in scored[:k]])

    def __len__(self) -> int:
        return len(self.passages)
