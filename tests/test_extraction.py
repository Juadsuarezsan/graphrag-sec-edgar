"""Extraction schemas, rule-based extractor and the mocked Claude extractor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import ValidationError

from src.extraction.claude import TOOL_NAME, ClaudeExtractionError, ClaudeExtractor
from src.extraction.rules import RuleBasedExtractor, extract_many
from src.extraction.schemas import (
    EDGE_TYPES,
    NODE_TYPES,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    extraction_tool_schema,
    slugify,
)
from src.ingestion.models import Chunk, Filing
from src.ingestion.parser import html_to_text
from tests.conftest import DEF14A_HTML, EX21_HTML

FOCAL = ExtractedEntity(id="test", type="Company", name="Test Corp")


def _chunk(filing: Filing, item: str, text: str) -> Chunk:
    return Chunk.make(filing, item, 0, text, 0, len(text))  # type: ignore[arg-type]


def test_slugify_and_closed_types() -> None:
    assert slugify("Taiwan Semiconductor Mfg. Co., Ltd.") == "taiwan_semiconductor_mfg_co_ltd"
    assert slugify("  ", prefix="sub_") == "sub_unnamed" and slugify("Café Inc") == "cafe_inc"
    assert "Company" in NODE_TYPES and "SUPPLIED_BY" in EDGE_TYPES
    with pytest.raises(ValidationError):
        ExtractedEntity(id="x", type="Organization", name="X")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExtractedRelationship(from_id="a", to_id="b", type="ACQUIRED")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        ExtractedRelationship(from_id="a", to_id="a", type="COMPETES_WITH")
    with pytest.raises(ValidationError):
        ExtractedEntity(id="Bad Id", type="Company", name="x")


def test_result_dedupes_and_checks_signatures() -> None:
    ents = [
        FOCAL,
        ExtractedEntity(id="tsmc", type="Company", name="TSMC"),
        ExtractedEntity(id="tsmc", type="Company", name="TSMC dup"),
    ]
    rels = [ExtractedRelationship(from_id="test", to_id="tsmc", type="SUPPLIED_BY")] * 2
    res = ExtractionResult(chunk_id="c", extractor="t", entities=ents, relationships=rels)
    assert (
        len(res.entities) == 2
        and len(res.relationships) == 1
        and res.entity_ids == {"test", "tsmc"}
    )
    with pytest.raises(ValidationError):
        ExtractionResult(
            chunk_id="c",
            extractor="t",
            entities=[FOCAL, ExtractedEntity(id="p", type="Person", name="P")],
            relationships=[ExtractedRelationship(from_id="test", to_id="p", type="SUPPLIED_BY")],
        )
    with pytest.raises(ValidationError):
        ExtractionResult(
            chunk_id="c",
            extractor="t",
            entities=[
                ExtractedEntity(id="x", type="Company", name="X"),
                ExtractedEntity(id="x", type="Person", name="X"),
            ],
        )
    merged = res.merge(ExtractionResult(chunk_id="d", extractor="t", tokens_in=5, cost_usd=0.1))
    assert merged.chunk_id == "c+d" and merged.tokens_in == 5 and merged.cost_usd == 0.1
    schema = extraction_tool_schema()
    assert set(schema["required"]) == {"entities", "relationships"}
    assert schema["properties"]["entities"]["items"]["properties"]["type"]["enum"] == list(
        NODE_TYPES
    )


def test_rules_exhibit21_and_def14a(filing: Filing) -> None:
    rules = RuleBasedExtractor()
    res = rules.extract(_chunk(filing, "EX-21", html_to_text(EX21_HTML)), focal=FOCAL)
    subs = [e for e in res.entities if e.type == "Subsidiary"]
    assert len(subs) == 5 and all(
        r.type == "SUBSIDIARY_OF" and r.to_id == "test" for r in res.relationships
    )
    assert (
        next(e for e in subs if e.name == "Test Capital, Inc.").properties["ownership_pct"] == 100.0
    )
    res = rules.extract(_chunk(filing, "DEF14A", html_to_text(DEF14A_HTML)), focal=FOCAL)
    people = {e.id for e in res.entities if e.type == "Person"}
    assert people == {"dana_whitfield", "marcus_oyelaran", "irene_castellanos"}
    assert {r.type for r in res.relationships} == {"DIRECTOR_OF"}


def test_rules_prose_cues_and_risk_lexicon(filing: Filing) -> None:
    text = (
        "Our products include WidgetOne, WidgetPro and WidgetMax. We compete with Acme Corporation and Globex Corporation. "
        "Our chips are supplied by Taiwan Semiconductor for all product lines. Jane Roe, our Chief Executive Officer, joined in 2020. "
        "Operations in China and Taiwan and cybersecurity incidents are risks; new tariffs could raise costs."
    )
    res = RuleBasedExtractor().extract(_chunk(filing, "1", text), focal=FOCAL)
    by_type = {
        t: {e.id for e in res.entities if e.type == t}
        for t in ("Product", "Company", "Person", "Risk")
    }
    assert by_type["Product"] == {"prod_widgetone", "prod_widgetpro", "prod_widgetmax"}
    assert by_type["Company"] == {
        "test",
        "acme_corporation",
        "globex_corporation",
        "taiwan_semiconductor",
    }
    assert by_type["Person"] == {"jane_roe"}
    assert by_type["Risk"] == {"risk_china_exposure", "risk_cybersecurity", "risk_tariffs"}
    kinds = {r.type for r in res.relationships}
    assert kinds == {"OFFERS_PRODUCT", "COMPETES_WITH", "SUPPLIED_BY", "CEO_OF", "MENTIONS_RISK"}
    assert all(r.evidence for r in res.relationships if r.type != "MENTIONS_RISK" or r.evidence)
    empty = RuleBasedExtractor().extract(_chunk(filing, "1", "Nothing relevant here."), focal=FOCAL)
    assert empty.entities == [FOCAL] and empty.relationships == []
    merged = extract_many(
        RuleBasedExtractor(),
        [_chunk(filing, "1", text), _chunk(filing, "1A", "Interest rate changes hurt us.")],
        FOCAL,
    )
    assert "risk_interest_rate" in merged.entity_ids
    assert extract_many(RuleBasedExtractor(), [], FOCAL).entities == [FOCAL]


def _message(payload: dict[str, Any] | None, tokens: tuple[int, int] = (1200, 300)) -> Any:
    blocks = [SimpleNamespace(type="text", text="thinking...")]
    if payload is not None:
        blocks.append(SimpleNamespace(type="tool_use", name=TOOL_NAME, input=payload))
    return SimpleNamespace(
        content=blocks, usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1])
    )


def test_claude_extractor_parses_tool_call_and_costs(filing: Filing, mocker: Any) -> None:
    client = mocker.MagicMock()
    client.messages.create.return_value = _message(
        {
            "entities": [{"id": "tsmc", "type": "Company", "name": "TSMC"}],
            "relationships": [
                {
                    "from_id": "test",
                    "to_id": "tsmc",
                    "type": "SUPPLIED_BY",
                    "evidence": "supplied by TSMC",
                }
            ],
        }
    )
    ex = ClaudeExtractor(client=client)
    res = ex.extract(_chunk(filing, "1", "We are supplied by TSMC."), focal=FOCAL)
    assert res.entity_ids == {"test", "tsmc"} and res.relationships[0].type == "SUPPLIED_BY"
    assert res.tokens_in == 1200 and res.tokens_out == 300 and res.cost_usd == pytest.approx(0.0081)
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-4-5-20250929"
    assert kwargs["tool_choice"] == {"type": "tool", "name": TOOL_NAME}
    assert "Item 1 (Business)" in kwargs["messages"][0]["content"]


def test_claude_extractor_errors_and_retries(filing: Filing, mocker: Any) -> None:
    with pytest.raises(ValueError):
        ClaudeExtractor()
    client = mocker.MagicMock()
    client.messages.create.return_value = _message(None)
    ex = ClaudeExtractor(client=client)
    ex._create.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    with pytest.raises(ClaudeExtractionError):
        ex.extract(_chunk(filing, "1", "x"), focal=FOCAL)
    client.messages.create.return_value = _message(
        {
            "entities": [{"id": "p", "type": "Person", "name": "P"}],
            "relationships": [{"from_id": "test", "to_id": "p", "type": "SUPPLIED_BY"}],
        }
    )
    with pytest.raises(ClaudeExtractionError):
        ex.extract(_chunk(filing, "1", "x"), focal=FOCAL)
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = [
        anthropic.APIConnectionError(request=request),
        _message({"entities": [], "relationships": []}),
    ]
    res = ex.extract(_chunk(filing, "1", "x"), focal=FOCAL)
    assert res.entities == [FOCAL] and client.messages.create.call_count == 4
