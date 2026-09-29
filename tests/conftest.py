"""Shared fixtures: deterministic seeds, graph, engine, API client, sample HTML."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.config import get_settings
from src.engine.graphrag import GraphRAGEngine
from src.graph.fixture import build_demo_graph
from src.graph.store import InMemoryGraph
from src.ingestion.models import Filing

SEED = 20260516
ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(autouse=True)
def _seed() -> None:
    random.seed(SEED)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Never let a developer's .env leak into tests; reset the settings cache."""
    for key in (
        "ANTHROPIC_API_KEY",
        "VOYAGE_API_KEY",
        "LANGSMITH_API_KEY",
        "GRAPH_BACKEND",
        "EMBEDDER",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8000,https://demo.example")
    monkeypatch.setenv("RATE_LIMIT", "1000/minute")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def demo_graph() -> InMemoryGraph:
    return build_demo_graph()


@pytest.fixture(scope="session")
def engine(demo_graph: InMemoryGraph) -> GraphRAGEngine:
    return GraphRAGEngine(demo_graph)


@pytest.fixture
def client() -> Iterator[TestClient]:
    from src.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def filing() -> Filing:
    return Filing(
        ticker="TEST",
        cik="0000000001",
        company_name="Test Corp",
        accession="0000000001-25-000001",
        form_type="10-K",
        filing_date="2025-06-30",
    )


def make_10k_html(style: str = "inline") -> str:
    """Synthetic 10-K in one of three heading styles: inline, nested, nbsp."""
    body = "We design and sell widgets. Our products include WidgetOne, WidgetPro and WidgetMax. "
    body += "We compete with Acme Corporation and Globex Corporation. " * 4
    risk = (
        "Our supply chain depends on China and Taiwan. Cybersecurity incidents could harm us. " * 6
    )
    mdna = "Revenue increased 5% year over year driven by WidgetPro. " * 12
    if style == "inline":
        return (
            "<html><body><ix:header><div>us-gaap:Member hidden</div></ix:header>"
            "<p>TABLE OF CONTENTS</p><p>Item 1. Business 4</p><p>Item 1A. Risk Factors 9</p>"
            "<p>Item 7. MD&amp;A 20</p>"
            f"<p><b>Item 1. Business</b></p><p>{body}</p>"
            f"<p><b>Item 1A. Risk Factors</b></p><p>{risk}</p>"
            f"<p><b>Item 7. Management's Discussion and Analysis</b></p><p>{mdna}</p></body></html>"
        )
    if style == "nested":
        return (
            '<html><body><div style="display:none">hidden text</div>'
            "<div><span><font>Item</font></span><span><font> 1.</font></span>"
            "<span><font> Business</font></span></div>"
            f"<div>{body}</div>"
            "<div><span><font>Item</font><font> 1A.</font></span>"
            "<span><font> Risk Factors</font></span></div>"
            f"<div>{risk}</div>"
            "<div><span><font>Item 7.</font></span><span> Management's Discussion</span></div>"
            f"<div>{mdna}</div></body></html>"
        )
    return (
        f"<html><body><p>ITEM&#160;1.&nbsp;&nbsp;BUSINESS</p><p>{body}</p>"
        f"<p>ITEM&#160;1A.&nbsp;RISK&nbsp;FACTORS</p><p>{risk}</p>"
        f"<p>ITEM&#160;7.&nbsp;MANAGEMENT'S DISCUSSION</p><p>{mdna}</p></body></html>"
    )


EX21_HTML = (
    "<html><body><h3>Exhibit 21.1</h3><table>"
    "<tr><td>Name</td><td>Jurisdiction</td></tr>"
    "<tr><td>Test Operations International Limited</td><td>Ireland</td></tr>"
    "<tr><td>Widget Electronics, LLC (1)</td><td>Delaware</td></tr>"
    "<tr><td>Test Capital, Inc.</td><td>Nevada</td><td>100%</td></tr>"
    "</table><p>Imperial Widgets Limited (Canada)</p><p>XTO Widgets Inc. — Delaware</p></body></html>"
)

DEF14A_HTML = (
    "<html><body><table><tr><th>Name</th><th>Age</th><th>Director Since</th></tr>"
    "<tr><td>Dana Whitfield</td><td>61</td><td>2015</td></tr>"
    "<tr><td>Marcus Oyelaran</td><td>58</td><td>2019</td></tr></table>"
    "<p>Irene Castellanos, 55, has served as a director since 2020.</p></body></html>"
)
