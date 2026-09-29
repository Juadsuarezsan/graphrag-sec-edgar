"""Ingestion pipeline end-to-end: synthetic HTML and the real Alphabet excerpt."""

from __future__ import annotations

from pathlib import Path

from src.engine.graphrag import GraphRAGEngine
from src.extraction.rules import RuleBasedExtractor
from src.graph.fixture import build_demo_graph
from src.graph.store import InMemoryGraph
from src.ingestion.models import Filing
from src.ingestion.pipeline import (
    SAMPLE_DIR,
    ingest_def14a,
    ingest_exhibit21,
    ingest_html,
    ingest_sections,
    load_sample_filings,
    prose_ratio,
)
from tests.conftest import DEF14A_HTML, EX21_HTML, ROOT, make_10k_html


def test_ingest_synthetic_10k_exhibit_and_proxy(filing: Filing) -> None:
    store = InMemoryGraph()
    rules = RuleBasedExtractor()
    rep = ingest_html(filing, make_10k_html("nested"), rules, store)
    assert rep.n_sections == 3 and rep.n_chunks >= 3 and rep.n_relationships >= 5
    assert store.get_node("test") is not None
    assert {e["type"] for e in store.edges()} >= {
        "OFFERS_PRODUCT",
        "COMPETES_WITH",
        "MENTIONS_RISK",
    }
    rep21 = ingest_exhibit21(filing, EX21_HTML, rules, store)
    assert rep21.n_relationships == 5
    rep14 = ingest_def14a(filing, DEF14A_HTML, rules, store)
    assert rep14.n_relationships == 3
    node = store.get_node("sub_test_capital_inc")
    assert node is not None and node["source_filing_date"] == "2025-06-30"
    assert node["properties"]["source_accession"] == filing.accession
    assert rep.tokens_in == 0 and rep.cost_usd == 0.0 and rep.as_dict()["ticker"] == "TEST"


def test_load_sample_filings_skips_stubs_and_xbrl() -> None:
    loaded = load_sample_filings(ROOT / SAMPLE_DIR)
    assert [f.ticker for f, _ in loaded] == ["GOOGL"]
    assert prose_ratio("us-gaap:DebtSecuritiesMember 2024-06-30 0000789019") < 0.2
    assert prose_ratio("We design, manufacture and market smartphones.") > 0.7
    assert prose_ratio("") == 0.0
    assert load_sample_filings(Path("/nonexistent")) == []


def test_e2e_real_googl_excerpt_into_fixture_graph_and_query() -> None:
    store = build_demo_graph()
    before = store.n_nodes
    for f, section in load_sample_filings(ROOT / SAMPLE_DIR):
        rep = ingest_sections(f, [section], RuleBasedExtractor(), store)
        assert rep.n_relationships >= 8
    # merged into the existing Alphabet node, no duplicate company
    assert store.get_node("googl") is None
    google = store.get_node("google")
    assert google is not None and google["properties"]["source_accession"] == "0001652044-26-000018"
    assert store.n_nodes > before
    engine = GraphRAGEngine(store)
    ans = engine.answer_sync("Which products does Alphabet offer?", mode="graph")
    assert {"prod_gmail", "prod_google_maps", "prod_android", "prod_gemini"} <= set(ans.answer_ids)
    ans2 = engine.answer_sync("Which company offers Gmail?", mode="graph")
    assert ans2.answer_ids == ["google"]
    assert "[google]" in ans2.answer
