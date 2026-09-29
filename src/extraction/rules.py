"""Deterministic extractor: structured exhibits plus conservative prose patterns.

Exhibit 21 and DEF 14A are tabular, so rules recover them with high
precision. For Item 1 / 1A prose the rules only fire on explicit cue phrases
(``compete with``, ``supplied by``, ``products include`` ...) and a risk
lexicon; recall is deliberately low and is what the Claude extractor improves.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from loguru import logger

from src.extraction.schemas import (
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    slugify,
)
from src.ingestion.models import Chunk
from src.ingestion.parser import parse_def14a_directors, parse_exhibit21

_ORG_TOKEN = r"[A-Z][A-Za-z0-9&.'\-]*"
_ORG_NAME = rf"{_ORG_TOKEN}(?:\s+{_ORG_TOKEN}){{0,4}}"
_LIST_SEP = re.compile(r",\s*|\s+and\s+|\s*;\s*")

_COMPETE_RE = re.compile(
    rf"compet(?:e|es|ing|ition|itors)\s+(?:directly\s+)?(?:with|from|against|include|including)\s+"
    rf"(?:companies\s+such\s+as\s+|firms\s+like\s+)?(?P<list>{_ORG_NAME}(?:(?:,\s*|\s+and\s+){_ORG_NAME}){{0,8}})",
)
_SUPPLIER_RE = re.compile(
    rf"(?:supplied\s+by|sourced\s+from|rel(?:y|ies)\s+on|single[- ]source(?:d)?\s+(?:from|supplier[s]?\s*(?:,|is|are)))\s+"
    rf"(?P<list>{_ORG_NAME}(?:(?:,\s*|\s+and\s+){_ORG_NAME}){{0,6}})\s+(?:for|to|as)\b",
)
_PRODUCTS_RE = re.compile(
    r"(?:products?|offerings?|platforms?|brands?)\s+(?:include|including|such\s+as|comprise|are)\s+"
    r"(?P<list>[A-Z][\w+.'\- ]{1,40}(?:,\s*[A-Z][\w+.'\- ]{1,40}){1,12}(?:,?\s+and\s+[A-Z][\w+.'\- ]{1,40})?)",
)
_CEO_RE = re.compile(
    r"(?P<name>[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){1,3}),?\s+(?:our|the Company's|the)\s+"
    r"(?:Chairman\s+and\s+)?Chief\s+Executive\s+Officer",
)
_GENERIC = {
    "the",
    "our",
    "company",
    "companies",
    "other",
    "others",
    "various",
    "certain",
    "these",
    "those",
    "many",
    "some",
    "we",
}

RISK_LEXICON: dict[str, tuple[str, list[str]]] = {
    "risk_china_exposure": ("China exposure", ["china", "taiwan", "hong kong"]),
    "risk_supply_concentration": (
        "Supplier concentration",
        ["single source", "sole source", "limited number of suppliers", "supplier concentration"],
    ),
    "risk_climate_regulation": (
        "Climate regulation",
        ["climate change", "greenhouse gas", "emissions regulation"],
    ),
    "risk_interest_rate": ("Interest rate sensitivity", ["interest rate"]),
    "risk_cybersecurity": (
        "Cybersecurity incidents",
        ["cybersecurity", "cyber-attack", "cyberattack", "data breach", "security breach"],
    ),
    "risk_ai_regulation": (
        "AI regulation",
        [
            "regulation of ai",
            "ai regulation",
            "artificial intelligence regulation",
            "regulate artificial intelligence",
            "ai act",
        ],
    ),
    "risk_antitrust": (
        "Antitrust enforcement",
        ["antitrust", "competition law", "competition authorities"],
    ),
    "risk_drug_pricing": (
        "Drug pricing pressure",
        ["drug pricing", "pricing pressure", "inflation reduction act"],
    ),
    "risk_patent_expiry": (
        "Patent expiration",
        ["patent expir", "loss of exclusivity", "generic competition"],
    ),
    "risk_fx": (
        "Foreign exchange volatility",
        ["foreign exchange", "currency fluctuation", "exchange rate"],
    ),
    "risk_tariffs": (
        "Tariffs and trade restrictions",
        ["tariff", "trade restriction", "export control"],
    ),
    "risk_talent": (
        "Talent retention",
        ["key personnel", "retain talent", "attract and retain", "highly skilled"],
    ),
    "risk_commodity_prices": (
        "Commodity price volatility",
        ["commodity price", "crude oil price", "raw material cost"],
    ),
}


def _split_names(blob: str) -> list[str]:
    names: list[str] = []
    for part in _LIST_SEP.split(blob):
        part = part.strip(" .;:")
        if not part or part.lower() in _GENERIC or len(part) < 2:
            continue
        if part.split()[0].lower() in _GENERIC:
            continue
        names.append(part)
    return names


def _sentence_around(text: str, start: int, end: int, width: int = 160) -> str:
    return text[max(0, start - width) : min(len(text), end + width)].strip()


class RuleBasedExtractor:
    """Regex/table extractor. Deterministic, no network, no tokens."""

    name = "rules-v1"

    def __init__(self, risk_lexicon: dict[str, tuple[str, list[str]]] | None = None) -> None:
        self._risks = risk_lexicon or RISK_LEXICON

    def extract(self, chunk: Chunk, *, focal: ExtractedEntity) -> ExtractionResult:
        """Dispatch on the chunk's section."""
        entities: list[ExtractedEntity] = [focal]
        rels: list[ExtractedRelationship] = []
        if chunk.item == "EX-21":
            self._exhibit21(chunk, focal, entities, rels)
        elif chunk.item == "DEF14A":
            self._def14a(chunk, focal, entities, rels)
        else:
            self._prose(chunk, focal, entities, rels)
        result = ExtractionResult(
            chunk_id=chunk.id, extractor=self.name, entities=entities, relationships=rels
        )
        logger.debug(
            "{}: {} entities, {} relationships",
            chunk.id,
            len(result.entities),
            len(result.relationships),
        )
        return result

    # -- structured -----------------------------------------------------------
    def _exhibit21(
        self,
        chunk: Chunk,
        focal: ExtractedEntity,
        ents: list[ExtractedEntity],
        rels: list[ExtractedRelationship],
    ) -> None:
        for row in parse_exhibit21(chunk.text):
            sid = slugify(row.name, prefix="sub_")
            props: dict[str, Any] = {"jurisdiction": row.jurisdiction, "parent": focal.id}
            if row.ownership_pct is not None:
                props["ownership_pct"] = row.ownership_pct
            ents.append(ExtractedEntity(id=sid, type="Subsidiary", name=row.name, properties=props))
            rels.append(
                ExtractedRelationship(
                    from_id=sid,
                    to_id=focal.id,
                    type="SUBSIDIARY_OF",
                    properties={"source": "Exhibit 21"},
                )
            )

    def _def14a(
        self,
        chunk: Chunk,
        focal: ExtractedEntity,
        ents: list[ExtractedEntity],
        rels: list[ExtractedRelationship],
    ) -> None:
        for row in parse_def14a_directors(chunk.text):
            pid = slugify(row.name)
            props = {"role": "Director", "age": row.age, "director_since": row.director_since}
            ents.append(ExtractedEntity(id=pid, type="Person", name=row.name, properties=props))
            rels.append(
                ExtractedRelationship(
                    from_id=pid,
                    to_id=focal.id,
                    type="DIRECTOR_OF",
                    properties={"since": row.director_since, "source": "DEF 14A"},
                )
            )
            if row.role and "chief executive" in row.role.lower() or row.role == "CEO":
                rels.append(ExtractedRelationship(from_id=pid, to_id=focal.id, type="CEO_OF"))

    # -- prose ----------------------------------------------------------------
    def _prose(
        self,
        chunk: Chunk,
        focal: ExtractedEntity,
        ents: list[ExtractedEntity],
        rels: list[ExtractedRelationship],
    ) -> None:
        text = chunk.text
        for m in _COMPETE_RE.finditer(text):
            for name in _split_names(m.group("list")):
                cid = slugify(name)
                if cid == focal.id:
                    continue
                ents.append(ExtractedEntity(id=cid, type="Company", name=name))
                rels.append(
                    ExtractedRelationship(
                        from_id=focal.id,
                        to_id=cid,
                        type="COMPETES_WITH",
                        evidence=_sentence_around(text, m.start(), m.end()),
                    )
                )
        for m in _SUPPLIER_RE.finditer(text):
            for name in _split_names(m.group("list")):
                cid = slugify(name)
                if cid == focal.id:
                    continue
                ents.append(ExtractedEntity(id=cid, type="Company", name=name))
                rels.append(
                    ExtractedRelationship(
                        from_id=focal.id,
                        to_id=cid,
                        type="SUPPLIED_BY",
                        evidence=_sentence_around(text, m.start(), m.end()),
                    )
                )
        for m in _PRODUCTS_RE.finditer(text):
            for name in _split_names(m.group("list")):
                if len(name.split()) > 4:
                    continue
                pid = slugify(name, prefix="prod_")
                ents.append(
                    ExtractedEntity(
                        id=pid, type="Product", name=name, properties={"company": focal.id}
                    )
                )
                rels.append(
                    ExtractedRelationship(
                        from_id=focal.id,
                        to_id=pid,
                        type="OFFERS_PRODUCT",
                        evidence=_sentence_around(text, m.start(), m.end()),
                    )
                )
        for m in _CEO_RE.finditer(text):
            name = m.group("name")
            pid = slugify(name)
            ents.append(
                ExtractedEntity(id=pid, type="Person", name=name, properties={"role": "CEO"})
            )
            rels.append(
                ExtractedRelationship(
                    from_id=pid,
                    to_id=focal.id,
                    type="CEO_OF",
                    evidence=_sentence_around(text, m.start(), m.end()),
                )
            )
        lowered = text.lower()
        for rid, (label, cues) in self._risks.items():
            hit = next((c for c in cues if c in lowered), None)
            if hit is None:
                continue
            ents.append(ExtractedEntity(id=rid, type="Risk", name=label))
            pos = lowered.find(hit)
            rels.append(
                ExtractedRelationship(
                    from_id=focal.id,
                    to_id=rid,
                    type="MENTIONS_RISK",
                    properties={"section": f"Item {chunk.item}", "cue": hit},
                    evidence=_sentence_around(text, pos, pos + len(hit), width=120),
                )
            )


def extract_many(
    extractor: RuleBasedExtractor, chunks: Iterable[Chunk], focal: ExtractedEntity
) -> ExtractionResult:
    """Run an extractor over chunks and merge the results."""
    merged: ExtractionResult | None = None
    for ch in chunks:
        res = extractor.extract(ch, focal=focal)
        merged = res if merged is None else merged.merge(res)
    return merged or ExtractionResult(chunk_id="empty", extractor=extractor.name, entities=[focal])
