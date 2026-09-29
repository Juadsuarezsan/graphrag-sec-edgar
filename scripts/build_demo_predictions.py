"""Produce the demo's pre-baked payloads with the repository's own engine.

Writes ``demo/predictions.json`` (eight questions, two per category, answered
in graph + vector mode with the deterministic template synthesizer) and
``demo/graph.json`` (the full graph snapshot the visualiser falls back to when
the API is offline). Nothing here is typed by hand; re-run after changing the
fixture or the engine.

Run: python scripts/build_demo_predictions.py
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from src.engine.graphrag import GraphRAGEngine
from src.evaluation.gold import load_gold
from src.extraction.rules import RuleBasedExtractor
from src.graph import staleness
from src.graph.fixture import build_demo_graph
from src.ingestion.pipeline import SAMPLE_DIR, ingest_sections, load_sample_filings

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
PICKS = ["lk-01", "lk-18", "2h-01", "2h-21", "3h-01", "3h-04", "ag-02", "ag-12"]


def main() -> None:
    """Build both JSON files."""
    store = build_demo_graph()
    for filing, section in load_sample_filings(ROOT / SAMPLE_DIR):
        ingest_sections(filing, [section], RuleBasedExtractor(), store)
    staleness.mark(store, datetime.now(UTC).date())
    engine = GraphRAGEngine(store)
    gold = {g.id: g for g in load_gold(ROOT / "eval" / "gold.jsonl")}
    predictions = []
    for pid in PICKS:
        item = gold[pid]
        graph = engine.answer_sync(item.question, mode="graph")
        vector = engine.answer_sync(item.question, mode="vector")
        predictions.append(
            {
                "id": item.id,
                "category": item.category,
                "question": item.question,
                "expected_ids": item.expected_ids,
                "expected_count": item.expected_count,
                "graph": graph.model_dump(),
                "vector": {
                    "answer": vector.answer,
                    "answer_ids": vector.answer_ids,
                    "count": vector.count,
                    "latency_ms": vector.metrics.latency_ms,
                },
            }
        )
    payload = {
        "project": "graphrag-sec-edgar",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "synthesis": "template (deterministic, no LLM)",
        "graph": {"nodes": store.n_nodes, "edges": store.n_edges},
        "predictions": predictions,
    }
    (DEMO / "predictions.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    snapshot = {
        "nodes": [
            {
                "id": n["id"],
                "type": n["type"],
                "name": n["name"],
                "stale": bool(n["properties"].get("stale", False)),
                "source_filing_date": n["source_filing_date"],
            }
            for n in store.nodes()
        ],
        "edges": [
            {"from": e["from_id"], "to": e["to_id"], "type": e["type"]} for e in store.edges()
        ],
        "total_nodes": store.n_nodes,
        "total_edges": store.n_edges,
        "truncated": False,
    }
    (DEMO / "graph.json").write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {len(predictions)} predictions and a {store.n_nodes}-node snapshot to {DEMO}")


if __name__ == "__main__":
    main()
