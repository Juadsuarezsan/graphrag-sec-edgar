"""Generate ``notebooks/demo.ipynb`` and execute its cells so outputs are real.

Runs without API keys (deterministic mode). Re-run after changing the engine.

Run: python scripts/build_notebook.py
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "demo.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "# GraphRAG sobre SEC EDGAR — recorrido de extremo a extremo\n\n"
        "Construye el grafo (fixture curado + excerpt real de Alphabet), ingiere un 10-K sintético, "
        "consulta en los tres modos y reproduce la tabla de evaluación. **Sin llaves de API**: síntesis por "
        "plantilla determinista, declarada en cada respuesta (`synthesizer='template'`).",
    ),
    (
        "code",
        "import json, sys\n"
        "from pathlib import Path\n"
        "ROOT = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n"
        "sys.path.insert(0, str(ROOT))\n"
        "from loguru import logger\n"
        "logger.remove(); logger.add(sys.stderr, level='WARNING')\n"
        "from src.graph.fixture import build_demo_graph, fixture_summary\n"
        "g = build_demo_graph()\n"
        "print(json.dumps(fixture_summary(), indent=1))",
    ),
    ("markdown", "## 1. Ingesta: HTML → secciones → chunks → extracción por reglas → MERGE"),
    (
        "code",
        "from src.ingestion.models import Filing\n"
        "from src.ingestion.parser import html_to_text\n"
        "from src.ingestion.chunker import split_sections\n"
        "from src.ingestion.pipeline import ingest_html, load_sample_filings, ingest_sections, SAMPLE_DIR\n"
        "from src.extraction.rules import RuleBasedExtractor\n"
        "html = ('<html><body><ix:header><div>us-gaap:hidden</div></ix:header>'\n"
        "        '<div><span><font>Item</font></span><span><font> 1.</font></span><span><font> Business</font></span></div>'\n"
        "        '<div>' + 'We design widgets. Our products include WidgetOne and WidgetPro. We compete with Acme Corporation. ' * 8 + '</div>'\n"
        "        '<div><span><font>Item 1A.</font></span><span> Risk Factors</span></div>'\n"
        "        '<div>' + 'Our supply chain depends on China. Cybersecurity incidents could harm us. ' * 8 + '</div></body></html>')\n"
        "print([ (s.item, s.n_chars) for s in split_sections(html_to_text(html)) ])\n"
        "filing = Filing(ticker='WDGT', cik='0000000009', company_name='Widget Corp', accession='0000000009-25-000001', filing_date='2025-06-30')\n"
        "report = ingest_html(filing, html, RuleBasedExtractor(), g)\n"
        "print(report.as_dict())\n"
        "for f, section in load_sample_filings(ROOT / SAMPLE_DIR):\n"
        "    print(f.ticker, ingest_sections(f, [section], RuleBasedExtractor(), g).as_dict())",
    ),
    ("markdown", "## 2. Staleness: nodos con evidencia de más de 365 días"),
    (
        "code",
        "from datetime import date\n"
        "from src.graph import staleness\n"
        "report = staleness.mark(g, date(2026, 9, 29))\n"
        "print(f'{report.stale_count} / {report.total_nodes} nodos stale')\n"
        "print([ (s.id, s.age_days) for s in report.stale_nodes[:5] ])",
    ),
    ("markdown", "## 3. Consulta en los tres modos"),
    (
        "code",
        "from src.engine.graphrag import GraphRAGEngine\n"
        "engine = GraphRAGEngine(g)\n"
        "q = 'Which CEOs lead companies that share a supplier with Apple?'\n"
        "for mode in ('graph', 'vector', 'hybrid'):\n"
        "    a = engine.answer_sync(q, mode=mode)\n"
        "    print(f'[{mode:6}] kind={a.classified_kind} ids={a.answer_ids}')\n"
        "    print('        plan =', [(p.edge_types, p.direction) for p in a.plan], '| latency_ms =', a.metrics.latency_ms)\n"
        "print()\n"
        "print(engine.answer_sync('How many S&P 500 companies mention China exposure as a risk?').answer)\n"
        "print(engine.answer_sync('Which products does Alphabet offer?').answer_ids)",
    ),
    ("markdown", "## 4. Evaluación: Vector RAG vs GraphRAG vs Hybrid vs ablación"),
    (
        "code",
        "import tempfile\n"
        "from src.evaluation.runner import run_all, render_results_md\n"
        "with tempfile.TemporaryDirectory() as tmp:\n"
        "    run_path = run_all(build_demo_graph(), ROOT / 'eval' / 'gold.jsonl', ROOT / 'eval' / 'extraction_gold.jsonl', Path(tmp), 'notebook')\n"
        "    md = render_results_md(run_path)\n"
        "print('\\n'.join(l for l in md.splitlines() if l.startswith('|'))[:1600])",
    ),
    (
        "markdown",
        "## 5. Qué falta para el sistema completo\n\n"
        "* `ANTHROPIC_API_KEY` → `ClaudeExtractor` sobre 100 × 3 filings reales y `ClaudeSynthesizer` con citas; faithfulness con la rúbrica de `src/evaluation/metrics.py`.\n"
        "* Salida de red a `sec.gov` → `scripts/download_data.py`.\n"
        "* Docker → `GRAPH_BACKEND=neo4j` con `docker compose up`.",
    ),
]


def main() -> None:
    """Execute the code cells in order and write the notebook with outputs."""
    namespace: dict[str, object] = {}
    cells = []
    for kind, source in CELLS:
        if kind == "markdown":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
            continue
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(source, "<cell>", "exec"), namespace)  # noqa: S102 - trusted repo code
        text = buf.getvalue()
        outputs = [{"output_type": "stream", "name": "stdout", "text": text}] if text else []
        cells.append(
            {
                "cell_type": "code",
                "execution_count": len(cells) + 1,
                "metadata": {},
                "source": source,
                "outputs": outputs,
            }
        )
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT} with {len(cells)} cells")


if __name__ == "__main__":
    main()
