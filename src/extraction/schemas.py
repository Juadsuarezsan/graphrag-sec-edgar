"""Closed Pydantic schemas for extraction output.

Node and edge types are literals: an LLM cannot invent ``ACQUIRED_BY`` or
``Organization``; the schema rejects it and the caller retries or drops the
record. ``docs/graph_schema.md`` mirrors these definitions.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, field_validator, model_validator

NodeType = Literal["Company", "Person", "Subsidiary", "Product", "Risk", "Market"]
EdgeType = Literal[
    "CEO_OF",
    "DIRECTOR_OF",
    "SUBSIDIARY_OF",
    "SUPPLIED_BY",
    "COMPETES_WITH",
    "MENTIONS_RISK",
    "OFFERS_PRODUCT",
    "OPERATES_IN",
]
NODE_TYPES: tuple[str, ...] = get_args(NodeType)
EDGE_TYPES: tuple[str, ...] = get_args(EdgeType)

#: Allowed (source type, target type) per edge type. Enforced at validation.
EDGE_SIGNATURES: dict[str, tuple[NodeType, NodeType]] = {
    "CEO_OF": ("Person", "Company"),
    "DIRECTOR_OF": ("Person", "Company"),
    "SUBSIDIARY_OF": ("Subsidiary", "Company"),
    "SUPPLIED_BY": ("Company", "Company"),
    "COMPETES_WITH": ("Company", "Company"),
    "MENTIONS_RISK": ("Company", "Risk"),
    "OFFERS_PRODUCT": ("Company", "Product"),
    "OPERATES_IN": ("Company", "Market"),
}

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str, prefix: str = "") -> str:
    """Deterministic ascii id for an entity name (``"Apple Inc."`` → ``apple_inc``)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = _SLUG_RE.sub("_", ascii_name.lower()).strip("_")
    return f"{prefix}{slug}" if slug else f"{prefix}unnamed"


class ExtractedEntity(BaseModel):
    """An entity found in a chunk."""

    id: str = Field(min_length=1, max_length=120, pattern=r"^[a-z0-9_]+$")
    type: NodeType
    name: str = Field(min_length=1, max_length=200)
    properties: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name is blank")
        return v


class ExtractedRelationship(BaseModel):
    """A typed relationship between two extracted (or pre-existing) entities."""

    from_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    to_id: str = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    type: EdgeType
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _no_self_loop(self) -> ExtractedRelationship:
        if self.from_id == self.to_id:
            raise ValueError(f"self-loop {self.from_id}-{self.type}->{self.to_id}")
        return self


class ExtractionResult(BaseModel):
    """Validated extraction output for one chunk."""

    chunk_id: str
    extractor: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @model_validator(mode="after")
    def _dedupe_and_check_signatures(self) -> ExtractionResult:
        seen: dict[str, ExtractedEntity] = {}
        for ent in self.entities:
            if ent.id in seen and seen[ent.id].type != ent.type:
                raise ValueError(f"entity {ent.id} declared with two types")
            seen.setdefault(ent.id, ent)
        self.entities = list(seen.values())
        kept: list[ExtractedRelationship] = []
        keys: set[tuple[str, str, str]] = set()
        for rel in self.relationships:
            sig = EDGE_SIGNATURES[rel.type]
            src, dst = seen.get(rel.from_id), seen.get(rel.to_id)
            if src is not None and src.type != sig[0]:
                raise ValueError(
                    f"{rel.type} source must be {sig[0]}, got {src.type} ({rel.from_id})"
                )
            if dst is not None and dst.type != sig[1]:
                raise ValueError(
                    f"{rel.type} target must be {sig[1]}, got {dst.type} ({rel.to_id})"
                )
            key = (rel.from_id, rel.to_id, rel.type)
            if key not in keys:
                keys.add(key)
                kept.append(rel)
        self.relationships = kept
        return self

    @property
    def entity_ids(self) -> set[str]:
        """Ids of all extracted entities."""
        return {e.id for e in self.entities}

    def merge(self, other: ExtractionResult) -> ExtractionResult:
        """Union of two results (used to fold several chunks of one filing)."""
        return ExtractionResult(
            chunk_id=f"{self.chunk_id}+{other.chunk_id}",
            extractor=self.extractor if self.extractor == other.extractor else "mixed",
            entities=self.entities + other.entities,
            relationships=self.relationships + other.relationships,
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            cost_usd=round(self.cost_usd + other.cost_usd, 6),
        )


def extraction_tool_schema() -> dict[str, Any]:
    """JSON schema of the ``record_extraction`` tool given to Claude."""
    entity_schema = ExtractedEntity.model_json_schema()
    rel_schema = ExtractedRelationship.model_json_schema()
    for schema in (entity_schema, rel_schema):
        schema.pop("title", None)
    return {
        "type": "object",
        "properties": {
            "entities": {"type": "array", "items": entity_schema},
            "relationships": {"type": "array", "items": rel_schema},
        },
        "required": ["entities", "relationships"],
    }
