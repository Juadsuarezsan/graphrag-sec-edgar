"""Small SEC-EDGAR-like graph fixture. ~25 companies, ~15 people, supplier and
competitor edges. Real ingestion of 10-K filings is a separate step you run
once when SEC_API_KEY is configured."""
from __future__ import annotations

from src.graph.store import InMemoryGraph


def build_demo_graph() -> InMemoryGraph:
    g = InMemoryGraph()

    # ---- Companies (S&P 500-ish) ----
    companies = [
        ("apple", "Apple Inc.", {"ticker": "AAPL", "in_sp500": True, "sector": "Tech", "revenue_2024": 383_000_000_000}),
        ("microsoft", "Microsoft Corp.", {"ticker": "MSFT", "in_sp500": True, "sector": "Tech", "revenue_2024": 245_000_000_000}),
        ("nvidia", "NVIDIA Corp.", {"ticker": "NVDA", "in_sp500": True, "sector": "Semiconductors", "revenue_2024": 60_000_000_000}),
        ("amd", "AMD Inc.", {"ticker": "AMD", "in_sp500": True, "sector": "Semiconductors"}),
        ("tsmc", "Taiwan Semiconductor", {"ticker": "TSM", "in_sp500": False, "sector": "Foundry"}),
        ("samsung", "Samsung Electronics", {"ticker": "005930.KS", "in_sp500": False, "sector": "Conglomerate"}),
        ("google", "Alphabet Inc.", {"ticker": "GOOGL", "in_sp500": True, "sector": "Tech"}),
        ("meta", "Meta Platforms", {"ticker": "META", "in_sp500": True, "sector": "Tech"}),
        ("tesla", "Tesla Inc.", {"ticker": "TSLA", "in_sp500": True, "sector": "Auto"}),
        ("ford", "Ford Motor", {"ticker": "F", "in_sp500": True, "sector": "Auto"}),
        ("gm", "General Motors", {"ticker": "GM", "in_sp500": True, "sector": "Auto"}),
        ("jpm", "JPMorgan Chase", {"ticker": "JPM", "in_sp500": True, "sector": "Financials"}),
        ("bofa", "Bank of America", {"ticker": "BAC", "in_sp500": True, "sector": "Financials"}),
        ("xom", "ExxonMobil", {"ticker": "XOM", "in_sp500": True, "sector": "Energy"}),
        ("pfe", "Pfizer", {"ticker": "PFE", "in_sp500": True, "sector": "Pharma"}),
        ("jnj", "Johnson & Johnson", {"ticker": "JNJ", "in_sp500": True, "sector": "Pharma"}),
    ]
    for cid, name, props in companies:
        g.upsert_node(id=cid, type="Company", name=name, properties=props)

    # ---- People (CEOs / directors) ----
    people = [
        ("tim_cook", "Tim Cook", {"role": "CEO", "company": "apple"}),
        ("satya_nadella", "Satya Nadella", {"role": "CEO", "company": "microsoft"}),
        ("jensen_huang", "Jensen Huang", {"role": "CEO", "company": "nvidia"}),
        ("lisa_su", "Lisa Su", {"role": "CEO", "company": "amd"}),
        ("morris_chang", "Morris Chang", {"role": "Founder", "company": "tsmc"}),
        ("sundar_pichai", "Sundar Pichai", {"role": "CEO", "company": "google"}),
        ("zuck", "Mark Zuckerberg", {"role": "CEO", "company": "meta"}),
        ("elon", "Elon Musk", {"role": "CEO", "company": "tesla"}),
        ("jamie_dimon", "Jamie Dimon", {"role": "CEO", "company": "jpm"}),
    ]
    for pid, name, props in people:
        g.upsert_node(id=pid, type="Person", name=name, properties=props)

    # ---- CEO_OF edges ----
    for pid, _, props in people:
        if props.get("role") in ("CEO", "Founder"):
            g.upsert_edge(from_id=pid, to_id=props["company"], type="CEO_OF")

    # ---- SUPPLIED_BY (TSMC is the giant supplier) ----
    for cid in ("apple", "nvidia", "amd", "tesla"):
        g.upsert_edge(from_id=cid, to_id="tsmc", type="SUPPLIED_BY",
                       properties={"materiality": "high"})
    g.upsert_edge(from_id="microsoft", to_id="tsmc", type="SUPPLIED_BY",
                   properties={"materiality": "medium"})

    # ---- COMPETES_WITH ----
    g.upsert_edge(from_id="apple", to_id="samsung", type="COMPETES_WITH")
    g.upsert_edge(from_id="nvidia", to_id="amd", type="COMPETES_WITH")
    g.upsert_edge(from_id="ford", to_id="gm", type="COMPETES_WITH")
    g.upsert_edge(from_id="ford", to_id="tesla", type="COMPETES_WITH")
    g.upsert_edge(from_id="pfe", to_id="jnj", type="COMPETES_WITH")
    g.upsert_edge(from_id="jpm", to_id="bofa", type="COMPETES_WITH")
    g.upsert_edge(from_id="google", to_id="microsoft", type="COMPETES_WITH")

    # ---- Risks ----
    risks = [
        ("risk_china_exposure", "China supply-chain exposure"),
        ("risk_chip_shortage", "Semiconductor shortage"),
        ("risk_climate_regulation", "Climate regulation"),
        ("risk_interest_rate", "Interest rate sensitivity"),
    ]
    for rid, name in risks:
        g.upsert_node(id=rid, type="Risk", name=name)

    for cid in ("apple", "nvidia", "amd", "microsoft", "tesla"):
        g.upsert_edge(from_id=cid, to_id="risk_china_exposure", type="MENTIONS_RISK")
    for cid in ("apple", "nvidia", "amd"):
        g.upsert_edge(from_id=cid, to_id="risk_chip_shortage", type="MENTIONS_RISK")
    for cid in ("xom", "ford", "gm", "tesla"):
        g.upsert_edge(from_id=cid, to_id="risk_climate_regulation", type="MENTIONS_RISK")
    for cid in ("jpm", "bofa"):
        g.upsert_edge(from_id=cid, to_id="risk_interest_rate", type="MENTIONS_RISK")

    return g
