"""Ingestion pipeline: filing → sections → chunks → extraction → graph MERGE.

Embeddings are computed in one batch per filing (not per node) and every
merged node receives ``source_filing_date`` so the staleness detector works.
"""

from __future__ import annotations

import html as html_lib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.extraction.base import EntityExtractor
from src.extraction.schemas import ExtractedEntity, ExtractionResult
from src.graph.base import GraphStore
from src.ingestion.chunker import chunk_sections, split_sections
from src.ingestion.models import Chunk, Filing, Section
from src.ingestion.parser import html_to_text
from src.observability import RequestMetrics

SAMPLE_DIR = Path("data/sec_edgar_sample")


@dataclass
class IngestReport:
    """What one filing contributed to the graph."""

    ticker: str
    accession: str
    n_sections: int
    n_chunks: int
    n_entities: int
    n_relationships: int
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serializable view."""
        return vars(self)


def focal_entity(filing: Filing, store: GraphStore | None = None) -> ExtractedEntity:
    """The filer as an extracted entity (always merged first).

    When ``store`` already holds a Company with the same ticker or CIK, its id
    is reused so ingestion enriches the existing node instead of creating a
    duplicate (``GOOGL`` → fixture id ``google``).
    """
    node_id = filing.company_id
    name = filing.company_name
    if store is not None:
        for n in store.nodes():
            props = n["properties"]
            if n["type"] == "Company" and (
                props.get("ticker") == filing.ticker
                or (filing.cik and props.get("cik") == filing.cik)
            ):
                node_id, name = n["id"], n["name"]
                break
    return ExtractedEntity(
        id=node_id,
        type="Company",
        name=name,
        properties={"ticker": filing.ticker, "cik": filing.cik},
    )


def merge_result(store: GraphStore, result: ExtractionResult, filing: Filing) -> tuple[int, int]:
    """MERGE an extraction result into ``store``. Returns ``(nodes, edges)`` touched."""
    texts = [f"{e.name} {e.type} {e.properties.get('description', '')}" for e in result.entities]
    vectors = store.embedder.embed(texts) if texts else []
    for ent, vec in zip(result.entities, vectors, strict=True):
        store.upsert_node(
            id=ent.id,
            type=ent.type,
            name=ent.name,
            properties={**ent.properties, "source_accession": filing.accession},
            embedding=vec,
            source_filing_date=filing.filing_date,
        )
    known = {e.id for e in result.entities}
    n_edges = 0
    for rel in result.relationships:
        if rel.from_id not in known and store.get_node(rel.from_id) is None:
            logger.warning("skip edge {}: unknown source {}", rel.type, rel.from_id)
            continue
        if rel.to_id not in known and store.get_node(rel.to_id) is None:
            logger.warning("skip edge {}: unknown target {}", rel.type, rel.to_id)
            continue
        props = {**rel.properties, "source_accession": filing.accession}
        if rel.evidence:
            props["evidence"] = rel.evidence
        store.upsert_edge(from_id=rel.from_id, to_id=rel.to_id, type=rel.type, properties=props)
        n_edges += 1
    return len(result.entities), n_edges


def ingest_sections(
    filing: Filing,
    sections: list[Section],
    extractor: EntityExtractor,
    store: GraphStore,
    *,
    max_chars: int = 2000,
) -> IngestReport:
    """Chunk the sections, extract and merge. Used by both HTML and sample paths."""
    metrics = RequestMetrics(name=f"ingest:{filing.ticker}")
    focal = focal_entity(filing, store)
    chunks: list[Chunk] = chunk_sections(filing, sections, max_chars=max_chars)
    merged: ExtractionResult | None = None
    warnings: list[str] = []
    for ch in chunks:
        res = extractor.extract(ch, focal=focal)
        merged = res if merged is None else merged.merge(res)
    if merged is None:
        warnings.append("no chunks produced")
        merged = ExtractionResult(chunk_id="empty", extractor=extractor.name, entities=[focal])
    n_nodes, n_edges = merge_result(store, merged, filing)
    metrics.tokens_in, metrics.tokens_out, metrics.cost_usd = (
        merged.tokens_in,
        merged.tokens_out,
        merged.cost_usd,
    )
    metrics.extra = {"nodes": n_nodes, "edges": n_edges, "chunks": len(chunks)}
    metrics.finish()
    return IngestReport(
        ticker=filing.ticker,
        accession=filing.accession,
        n_sections=len(sections),
        n_chunks=len(chunks),
        n_entities=n_nodes,
        n_relationships=n_edges,
        tokens_in=merged.tokens_in,
        tokens_out=merged.tokens_out,
        cost_usd=merged.cost_usd,
        latency_ms=metrics.latency_ms,
        warnings=warnings,
    )


def ingest_html(
    filing: Filing, html: str, extractor: EntityExtractor, store: GraphStore
) -> IngestReport:
    """Full path for a 10-K HTML document."""
    text = html_to_text(html)
    return ingest_sections(filing, split_sections(text), extractor, store)


def ingest_exhibit21(
    filing: Filing, html_or_text: str, extractor: EntityExtractor, store: GraphStore
) -> IngestReport:
    """Path for an Exhibit 21 document (single structured section)."""
    text = html_to_text(html_or_text) if "<" in html_or_text else html_or_text
    section = Section(
        item="EX-21",
        title="Subsidiaries of the Registrant",
        text=text,
        char_start=0,
        char_end=len(text),
    )
    return ingest_sections(filing, [section], extractor, store, max_chars=max(len(text), 1))


def ingest_def14a(
    filing: Filing, html_or_text: str, extractor: EntityExtractor, store: GraphStore
) -> IngestReport:
    """Path for a DEF 14A proxy statement (director nominees)."""
    text = html_to_text(html_or_text) if "<" in html_or_text else html_or_text
    section = Section(
        item="DEF14A",
        title="Proxy Statement - Directors",
        text=text,
        char_start=0,
        char_end=len(text),
    )
    return ingest_sections(filing, [section], extractor, store, max_chars=max(len(text), 1))


_WORD_RE = re.compile(r"^[A-Za-z][a-z]+[,.;]?$")


def prose_ratio(text: str) -> float:
    """Share of whitespace tokens that look like English words.

    Inline-XBRL headers stripped by regex look like ``us-gaap:DebtSecuritiesMember
    2024-06-30 0000789019``; their ratio is near 0, real prose is above 0.7.
    """
    tokens = text.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if _WORD_RE.match(t)) / len(tokens)


def load_sample_filings(
    sample_dir: Path = SAMPLE_DIR, min_chars: int = 500, min_prose_ratio: float = 0.5
) -> list[tuple[Filing, Section]]:
    """Load ``data/sec_edgar_sample/*_10K.json`` excerpts as Item 1 sections.

    Excerpts shorter than ``min_chars`` (the four filers whose regex extraction
    failed upstream) or without prose (MSFT's excerpt is an XBRL header) are
    skipped with a warning; the JSON files are kept as evidence.
    """
    out: list[tuple[Filing, Section]] = []
    for path in sorted(sample_dir.glob("*_10K.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        text = html_lib.unescape(str(raw.get("business_section_excerpt", "")))
        if len(text) < min_chars:
            logger.warning("{}: excerpt has {} chars, skipped", path.name, len(text))
            continue
        ratio = prose_ratio(text)
        if ratio < min_prose_ratio:
            logger.warning(
                "{}: prose ratio {:.2f} < {} (XBRL header, no prose), skipped",
                path.name,
                ratio,
                min_prose_ratio,
            )
            continue
        filing = Filing(
            ticker=raw["ticker"],
            cik=raw["cik"],
            company_name=raw["company_name"],
            accession=raw["accession"],
            form_type="10-K",
            filing_date=raw.get("filing_date"),
            source_url=raw.get("source_url"),
        )
        out.append(
            (
                filing,
                Section(item="1", title="Business", text=text, char_start=0, char_end=len(text)),
            )
        )
    return out
