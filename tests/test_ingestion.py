"""HTML parser, section splitter, chunker and exhibit parsers."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_sections, chunk_text, split_sections
from src.ingestion.models import Chunk, Filing, Section
from src.ingestion.parser import html_to_text, parse_def14a_directors, parse_exhibit21
from tests.conftest import DEF14A_HTML, EX21_HTML, make_10k_html


@pytest.mark.parametrize("style", ["inline", "nested", "nbsp"])
def test_sections_found_in_all_three_heading_styles(style: str) -> None:
    text = html_to_text(make_10k_html(style))
    assert "hidden" not in text and "us-gaap" not in text
    sections = split_sections(text)
    assert [s.item for s in sections] == ["1", "1A", "7"]
    assert sections[0].text.startswith("We design and sell widgets")
    assert "Cybersecurity incidents" in sections[1].text
    assert sections[1].title == "Risk Factors"


def test_html_to_text_edge_cases() -> None:
    assert html_to_text("") == "" and html_to_text("   ") == ""
    assert html_to_text("<p>a&nbsp;&nbsp;b</p><p>c</p>") == "a b\nc"
    assert html_to_text("<table><tr><td>x</td><td>y</td></tr></table>") == "x | y"
    assert html_to_text("<!-- hidden comment --><script>var a=1</script><p>ok</p>") == "ok"


def test_split_sections_fallbacks() -> None:
    full = split_sections("no headings here at all " * 20)
    assert len(full) == 1 and full[0].item == "FULL"
    toc_only = split_sections("Item 1. Business 3\nItem 1A. Risk Factors 5\n")
    assert toc_only[0].item == "FULL"


def test_chunk_text_boundaries_and_errors() -> None:
    assert chunk_text("") == [] and chunk_text("short", 100) == [("short", 0, 5)]
    text = "First sentence here. Second sentence follows. Third one closes it. " * 20
    chunks = chunk_text(text, max_chars=200, overlap=40)
    assert len(chunks) > 5
    assert all(len(c[0]) <= 200 for c in chunks)
    assert all(c[0].endswith(".") for c in chunks[:-1])
    assert chunks[1][1] < chunks[0][2]  # overlap
    with pytest.raises(ValueError):
        chunk_text("x", max_chars=0)
    oversized = chunk_text("a" * 5000, max_chars=1000, overlap=0)  # no sentence boundaries
    assert len(oversized) == 5


def test_chunk_sections_ids_are_deterministic(filing: Filing) -> None:
    sec = Section(
        item="1", title="Business", text="Sentence one. " * 300, char_start=10, char_end=4210
    )
    a = chunk_sections(filing, [sec], max_chars=1000)
    b = chunk_sections(filing, [sec], max_chars=1000)
    assert [c.id for c in a] == [c.id for c in b] and a[0].id.startswith("test-1-000-")
    assert a[0].char_start == 10 and isinstance(a[0], Chunk)


def test_filing_validation() -> None:
    with pytest.raises(ValueError):
        Filing(ticker="X", cik="1", company_name="X", accession="bad")
    f = Filing(ticker="BRK.B", cik="1", company_name="X", accession="0000000001-25-000001")
    assert f.company_id == "brk_b"


def test_parse_exhibit21_tables_and_prose() -> None:
    rows = parse_exhibit21(EX21_HTML)
    names = {r.name: r for r in rows}
    assert set(names) == {
        "Test Operations International Limited",
        "Widget Electronics, LLC",
        "Test Capital, Inc.",
        "Imperial Widgets Limited",
        "XTO Widgets Inc.",
    }
    assert (
        names["Test Capital, Inc."].ownership_pct == 100.0
        and names["Test Capital, Inc."].jurisdiction == "Nevada"
    )
    assert names["Imperial Widgets Limited"].jurisdiction == "Canada"
    assert parse_exhibit21("") == [] and parse_exhibit21("Name | Jurisdiction") == []


def test_parse_def14a_table_and_sentence() -> None:
    rows = {r.name: r for r in parse_def14a_directors(DEF14A_HTML)}
    assert set(rows) == {"Dana Whitfield", "Marcus Oyelaran", "Irene Castellanos"}
    assert rows["Dana Whitfield"].age == 61 and rows["Dana Whitfield"].director_since == 2015
    assert rows["Irene Castellanos"].age == 55 and rows["Irene Castellanos"].director_since == 2020
    assert parse_def14a_directors("<p>nothing relevant</p>") == []
