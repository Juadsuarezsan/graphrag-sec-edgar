"""Gold set integrity, metrics and the eval runner end-to-end on a subset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.schemas import GraphAnswer, QueryMetrics
from src.evaluation.gold import EXPECTED_COUNTS, GoldItem, category_counts, load_gold
from src.evaluation.metrics import FAITHFULNESS_RUBRIC, extraction_prf, percentile, score_answer
from src.evaluation.runner import (
    PENDING_FAITHFULNESS,
    render_results_md,
    run_all,
    run_extraction_eval,
)
from src.extraction.schemas import ExtractedEntity, ExtractionResult
from src.graph.store import InMemoryGraph
from tests.conftest import ROOT

GOLD = ROOT / "eval" / "gold.jsonl"
EXTRACTION_GOLD = ROOT / "eval" / "extraction_gold.jsonl"


def test_gold_has_100_items_with_spec_distribution(demo_graph: InMemoryGraph) -> None:
    items = load_gold(GOLD)
    assert len(items) == 100 and category_counts(items) == EXPECTED_COUNTS
    for it in items:
        for nid in it.expected_ids:
            assert demo_graph.get_node(nid) is not None, (it.id, nid)
        if it.category == "aggregation":
            assert it.expected_count == len(it.expected_ids), it.id
        assert it.ground_truth.startswith("manual")


def test_gold_item_validation() -> None:
    with pytest.raises(ValueError):
        GoldItem(id="x", category="aggregation", question="How many?", hops=1)
    with pytest.raises(ValueError):
        GoldItem(id="x", category="lookup", question="Who is?", hops=1)


def _answer(
    ids: list[str], kind: str = "lookup", count: int | None = None, text: str = ""
) -> GraphAnswer:
    return GraphAnswer(question="q", mode="graph", answer=text, classified_kind=kind, answer_ids=ids, count=count, metrics=QueryMetrics(trace_id="t", latency_ms=1))  # type: ignore[arg-type]


def test_score_answer_rules() -> None:
    item = GoldItem(
        id="a",
        category="lookup",
        question="Who is x?",
        expected_ids=["tim_cook"],
        hops=1,
        expected_text="MSFT",
    )
    assert not score_answer(item, _answer(["tim_cook"], text="nope"), "lookup").hit
    res = score_answer(item, _answer(["tim_cook", "extra"], text="MSFT"), "lookup")
    assert res.hit and res.precision == 0.5 and res.recall == 1.0 and res.routing_ok
    agg = GoldItem(
        id="b",
        category="aggregation",
        question="How many?",
        expected_ids=["a", "b"],
        expected_count=2,
        hops=1,
    )
    assert score_answer(agg, _answer(["a"], "aggregation", count=2), "aggregation").hit
    assert not score_answer(agg, _answer(["a", "b"], "aggregation", count=3), "aggregation").hit
    assert (
        percentile([], 50) == 0.0
        and percentile([1, 2, 3, 4], 50) == 2
        and percentile([1, 2, 3, 4], 95) == 4
    )


def test_extraction_prf() -> None:
    focal = ExtractedEntity(id="f", type="Company", name="F")
    gold = ExtractionResult(chunk_id="c", extractor="g", entities=[focal, ExtractedEntity(id="a", type="Company", name="A"), ExtractedEntity(id="b", type="Company", name="B")], relationships=[{"from_id": "f", "to_id": "a", "type": "COMPETES_WITH"}])  # type: ignore[list-item]
    pred = ExtractionResult(chunk_id="c", extractor="p", entities=[focal, ExtractedEntity(id="a", type="Company", name="A")], relationships=[{"from_id": "f", "to_id": "a", "type": "COMPETES_WITH"}])  # type: ignore[list-item]
    e, r = extraction_prf(gold, pred, focal_id="f")
    assert e.precision == 1.0 and e.recall == 0.5 and e.f1 == pytest.approx(2 / 3) and r.f1 == 1.0
    assert e.as_dict()["n_gold"] == 2 and len(FAITHFULNESS_RUBRIC) == 5


def test_extraction_eval_on_annotated_chunks() -> None:
    res = run_extraction_eval(EXTRACTION_GOLD)
    assert res["n_chunks"] == 5 and res["entity"]["f1"] > 0.8 and res["relationship"]["f1"] > 0.8
    real = next(c for c in res["chunks"] if c["id"] == "item1-googl-real")
    assert real["entities"]["recall"] >= 0.8


def test_run_all_writes_json_and_renders_markdown(
    tmp_path: Path, demo_graph: InMemoryGraph
) -> None:
    items = load_gold(GOLD)
    subset = tmp_path / "gold.jsonl"
    picked = [it for it in items if it.id in {"lk-01", "2h-01", "3h-04", "ag-04"}]
    subset.write_text("\n".join(json.dumps(it.model_dump()) for it in picked) + "\n")
    run_path = run_all(demo_graph, subset, EXTRACTION_GOLD, tmp_path / "runs", "unit")
    payload = json.loads(run_path.read_text())
    assert (
        payload["synthesis"] == "template (deterministic, no LLM)"
        and payload["faithfulness"] == PENDING_FAITHFULNESS
    )
    assert len(payload["systems"]) == 4 and payload["model_pinned"] == "claude-sonnet-4-5-20250929"
    graph = next(s for s in payload["systems"] if s["mode"] == "graph" and s["max_hops"] == 3)
    assert graph["overall_hit_rate"] == 1.0 and graph["cost_usd"] == 0.0
    md = render_results_md(run_path)
    assert "| GraphRAG (este sistema) |" in md and PENDING_FAITHFULNESS in md and "Vector RAG" in md
    assert "sin LLM" in md


def test_committed_results_match_committed_run() -> None:
    runs = sorted((ROOT / "eval" / "runs").glob("*.json"))
    assert runs, "eval/runs must contain at least one stored run"
    results = (ROOT / "eval" / "RESULTS.md").read_text(encoding="utf-8")
    assert runs[-1].name in results
