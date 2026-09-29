"""``python -m eval.run`` — regenerate ``eval/runs/*.json`` and ``eval/RESULTS.md``.

Options:
    --name NAME        Run name (default ``fixture-deterministic``).
    --gold PATH        Gold JSONL (default ``eval/gold.jsonl``).
    --backend memory|networkx   Graph backend (default memory, fixture + sample).
    --graph-path PATH  JSON graph for the networkx backend.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.evaluation.runner import render_results_md, run_all
from src.extraction.rules import RuleBasedExtractor
from src.graph.base import GraphStore
from src.graph.fixture import populate
from src.graph.networkx_store import NetworkXGraphStore
from src.graph.store import InMemoryGraph
from src.ingestion.pipeline import SAMPLE_DIR, ingest_sections, load_sample_filings

ROOT = Path(__file__).resolve().parent


def build_store(backend: str, graph_path: str | None) -> GraphStore:
    """Fixture + real sample excerpts in the requested backend."""
    store: GraphStore
    if backend == "networkx":
        store = NetworkXGraphStore(graph_path or "data/processed/graph.json")
        if store.n_nodes:
            return store
    else:
        store = InMemoryGraph()
    populate(store)
    extractor = RuleBasedExtractor()
    for filing, section in load_sample_filings(ROOT.parent / SAMPLE_DIR):
        ingest_sections(filing, [section], extractor, store)
    return store


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--name", default="fixture-deterministic")
    parser.add_argument("--gold", default=str(ROOT / "gold.jsonl"))
    parser.add_argument("--extraction-gold", default=str(ROOT / "extraction_gold.jsonl"))
    parser.add_argument("--backend", choices=["memory", "networkx"], default="memory")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--out-dir", default=str(ROOT / "runs"))
    parser.add_argument("--results", default=str(ROOT / "RESULTS.md"))
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    store = build_store(args.backend, args.graph_path)
    run_path = run_all(
        store, Path(args.gold), Path(args.extraction_gold), Path(args.out_dir), args.name
    )
    Path(args.results).write_text(render_results_md(run_path), encoding="utf-8")
    sys.stdout.write(f"run: {run_path}\nresults: {args.results}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
