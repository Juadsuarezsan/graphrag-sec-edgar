"""Evaluation runner: Vector RAG vs GraphRAG vs Hybrid (+ ablation) on the gold set.

Every run is written to ``eval/runs/<date>-<name>.json`` and
``eval/RESULTS.md`` is regenerated from that file only. Runs made without an
LLM are labelled ``"synthesis": "template (deterministic, no LLM)"`` so the
numbers are never mistaken for the full system's output.
"""

from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from src.api.schemas import QueryMode
from src.config import get_settings
from src.engine.graphrag import GraphRAGEngine
from src.evaluation.gold import CATEGORY_TO_KIND, GoldItem, category_counts, load_gold
from src.evaluation.metrics import FAITHFULNESS_RUBRIC, extraction_prf, percentile, score_answer
from src.extraction.rules import RuleBasedExtractor
from src.extraction.schemas import ExtractedEntity, ExtractionResult
from src.graph.base import GraphStore
from src.ingestion.models import Chunk, Filing

CATEGORIES: tuple[str, ...] = ("lookup", "two_hop", "three_hop", "aggregation")
SYSTEMS: tuple[tuple[str, QueryMode, int], ...] = (
    ("Vector RAG (TF-IDF, mismo corpus)", "vector", 3),
    ("GraphRAG (este sistema)", "graph", 3),
    ("Hybrid: Graph + Vector", "hybrid", 3),
)
ABLATION: tuple[str, QueryMode, int] = (
    "Ablation: GraphRAG sin traversal multi-hop (max_hops=1)",
    "graph",
    1,
)
PENDING_FAITHFULNESS = "pendiente (requiere ANTHROPIC_API_KEY)"


def run_system(
    engine: GraphRAGEngine, gold: list[GoldItem], mode: QueryMode, max_hops: int, k: int = 5
) -> dict[str, Any]:
    """Run one system over the gold set and return per-category metrics + cases."""
    cases: list[dict[str, Any]] = []
    per_cat: dict[str, dict[str, float]] = {
        c: {"n": 0, "hits": 0, "precision": 0.0, "recall": 0.0, "routing": 0} for c in CATEGORIES
    }
    latencies: list[int | float] = []
    tokens_in = tokens_out = 0
    cost = 0.0
    for item in gold:
        ans = engine.answer_sync(item.question, k=k, max_hops=max_hops, mode=mode)
        res = score_answer(item, ans, CATEGORY_TO_KIND[item.category])
        bucket = per_cat[item.category]
        bucket["n"] += 1
        bucket["hits"] += int(res.hit)
        bucket["precision"] += res.precision
        bucket["recall"] += res.recall
        bucket["routing"] += int(res.routing_ok)
        latencies.append(ans.metrics.latency_ms)
        tokens_in += ans.metrics.tokens_in
        tokens_out += ans.metrics.tokens_out
        cost += ans.metrics.cost_usd
        cases.append(
            {
                "id": item.id,
                "category": item.category,
                "question": item.question,
                "hit": res.hit,
                "precision": round(res.precision, 3),
                "recall": round(res.recall, 3),
                "expected_ids": item.expected_ids,
                "expected_count": item.expected_count,
                "answer_ids": ans.answer_ids,
                "count": ans.count,
                "classified_kind": ans.classified_kind,
                "routing_ok": res.routing_ok,
                "latency_ms": ans.metrics.latency_ms,
                "answer": ans.answer[:300],
            }
        )
    summary: dict[str, Any] = {}
    for c, b in per_cat.items():
        n = int(b["n"]) or 1
        summary[c] = {
            "n": int(b["n"]),
            "hit_rate": round(b["hits"] / n, 4),
            "precision": round(b["precision"] / n, 4),
            "recall": round(b["recall"] / n, 4),
            "routing_accuracy": round(b["routing"] / n, 4),
        }
    total_hits = sum(int(b["hits"]) for b in per_cat.values())
    return {
        "mode": mode,
        "max_hops": max_hops,
        "k": k,
        "overall_hit_rate": round(total_hits / len(gold), 4) if gold else 0.0,
        "by_category": summary,
        "latency_ms": {
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "mean": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        },
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": round(cost, 6),
        "faithfulness": None,
        "cases": cases,
    }


def run_extraction_eval(gold_path: Path) -> dict[str, Any]:
    """Entity/Relationship F1 of the rule-based extractor against hand-annotated chunks."""
    extractor = RuleBasedExtractor()
    rows: list[dict[str, Any]] = []
    ent_p = ent_r = rel_p = rel_r = 0.0
    n = 0
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        filing = Filing.model_validate(raw["filing"])
        focal = ExtractedEntity.model_validate(raw["focal"])
        chunk = Chunk.make(filing, raw["item"], 0, raw["text"], 0, len(raw["text"]))
        gold = ExtractionResult(
            chunk_id=chunk.id,
            extractor="gold",
            entities=[focal, *raw["entities"]],
            relationships=raw["relationships"],
        )
        pred = extractor.extract(chunk, focal=focal)
        e_prf, r_prf = extraction_prf(gold, pred, focal_id=focal.id)
        ent_p += e_prf.precision
        ent_r += e_prf.recall
        rel_p += r_prf.precision
        rel_r += r_prf.recall
        n += 1
        rows.append(
            {
                "id": raw["id"],
                "source": raw.get("source", ""),
                "entities": e_prf.as_dict(),
                "relationships": r_prf.as_dict(),
            }
        )
    n = n or 1
    ep, er, rp, rr = ent_p / n, ent_r / n, rel_p / n, rel_r / n
    return {
        "extractor": extractor.name,
        "n_chunks": len(rows),
        "entity": {
            "precision": round(ep, 4),
            "recall": round(er, 4),
            "f1": round(2 * ep * er / (ep + er), 4) if ep + er else 0.0,
        },
        "relationship": {
            "precision": round(rp, 4),
            "recall": round(rr, 4),
            "f1": round(2 * rp * rr / (rp + rr), 4) if rp + rr else 0.0,
        },
        "claude_extractor": PENDING_FAITHFULNESS.replace(
            "pendiente", "pendiente Entity/Relationship F1 de ClaudeExtractor"
        ),
        "chunks": rows,
    }


def worst_cases(
    systems: list[dict[str, Any]], mode: QueryMode = "graph", n: int = 10
) -> list[dict[str, Any]]:
    """The ``n`` lowest-scoring GraphRAG cases (misses first, then low precision)."""
    for s in systems:
        if s["mode"] == mode and s["max_hops"] >= 3:
            ranked = sorted(s["cases"], key=lambda c: (c["hit"], c["recall"], c["precision"]))
            return ranked[:n]
    return []


def run_all(
    store: GraphStore, gold_path: Path, extraction_gold_path: Path | None, out_dir: Path, name: str
) -> Path:
    """Run every system, write the JSON artifact and return its path."""
    gold = load_gold(gold_path)
    counts = category_counts(gold)
    engine = GraphRAGEngine(store)
    systems: list[dict[str, Any]] = []
    for label, mode, hops in (*SYSTEMS, ABLATION):
        logger.info("running {} ...", label)
        result = run_system(engine, gold, mode, hops)
        result["label"] = label
        systems.append(result)
    extraction = (
        run_extraction_eval(extraction_gold_path)
        if extraction_gold_path and extraction_gold_path.exists()
        else None
    )
    settings = get_settings()
    payload: dict[str, Any] = {
        "name": name,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "synthesis": "template (deterministic, no LLM)",
        "llm_enabled": bool(settings.anthropic_api_key),
        "model_pinned": settings.anthropic_model,
        "embedder": store.embedder.name,
        "graph": {"nodes": store.n_nodes, "edges": store.n_edges},
        "gold": {"path": str(gold_path), "n": len(gold), "by_category": counts},
        "faithfulness": PENDING_FAITHFULNESS,
        "faithfulness_rubric": list(FAITHFULNESS_RUBRIC),
        "systems": systems,
        "extraction": extraction,
        "worst_cases": worst_cases(systems),
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(UTC).date().isoformat()}-{name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("run written to {}", path)
    return path


def _pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def render_results_md(run_path: Path) -> str:
    """Render ``eval/RESULTS.md`` from one run artifact."""
    run = json.loads(run_path.read_text(encoding="utf-8"))
    lines: list[str] = []
    lines.append("# Resultados de evaluación — GraphRAG sobre SEC EDGAR")
    lines.append("")
    lines.append(
        f"Generado por `python -m eval.run` a partir de `{run_path.as_posix()}` ({run['generated_at']})."
    )
    lines.append("")
    lines.append(
        "> **Alcance declarado.** Corrida **sin LLM**: síntesis por plantilla determinista, extracción por reglas. "
    )
    lines.append(
        f"> Grafo evaluado: {run['graph']['nodes']} nodos / {run['graph']['edges']} aristas construidos a partir del fixture curado a mano "
    )
    lines.append(
        "> (`src/graph/fixture.py`) más el excerpt real del 10-K de Alphabet. Los hit rates miden **recuperación estructural**, no la calidad "
    )
    lines.append(
        f"> de la redacción del modelo. Faithfulness: **{run['faithfulness']}**. Modelo pinneado para la corrida con LLM: `{run['model_pinned']}`."
    )
    lines.append("")
    g = run["gold"]["by_category"]
    lines.append(
        f"Eval set: `{run['gold']['path']}` — {run['gold']['n']} preguntas (lookup {g.get('lookup', 0)} / dos saltos {g.get('two_hop', 0)} / tres+ saltos {g.get('three_hop', 0)} / aggregation {g.get('aggregation', 0)}), ground truth escrita a mano."
    )
    lines.append("")
    lines.append("## Tabla comparativa (hit rate por categoría)")
    lines.append("")
    lines.append(
        "| Sistema | Lookup Hit | 2-hop Hit | 3-hop Hit | Aggregation Hit | Faithfulness |"
    )
    lines.append("|---|---|---|---|---|---|")
    for s in run["systems"]:
        bc = s["by_category"]
        lines.append(
            f"| {s['label']} | {_pct(bc['lookup']['hit_rate'])} | {_pct(bc['two_hop']['hit_rate'])} | "
            f"{_pct(bc['three_hop']['hit_rate'])} | {_pct(bc['aggregation']['hit_rate'])} | {run['faithfulness']} |"
        )
    lines.append("")
    lines.append(
        "Hit = todos los ids esperados están en la respuesta (lookup / multi-hop) o el conteo coincide exactamente (aggregation)."
    )
    lines.append("")
    lines.append("## Precisión, routing, latencia y costo")
    lines.append("")
    lines.append(
        "| Sistema | Hit global | Precisión media (ids) | Routing acc. | p50 ms | p95 ms | Tokens in/out | Costo USD |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in run["systems"]:
        bc = s["by_category"]
        prec = sum(bc[c]["precision"] * bc[c]["n"] for c in CATEGORIES) / max(
            1, sum(bc[c]["n"] for c in CATEGORIES)
        )
        rout = sum(bc[c]["routing_accuracy"] * bc[c]["n"] for c in CATEGORIES) / max(
            1, sum(bc[c]["n"] for c in CATEGORIES)
        )
        lat = s["latency_ms"]
        lines.append(
            f"| {s['label']} | {_pct(s['overall_hit_rate'])} | {_pct(prec)} | {_pct(rout)} | {lat['p50']:.0f} | {lat['p95']:.0f} | "
            f"{s['tokens_in']}/{s['tokens_out']} | {s['cost_usd']:.4f} |"
        )
    lines.append("")
    lines.append(
        "Tokens y costo son 0 porque la corrida no invoca ningún LLM (fallback determinista, sin LLM)."
    )
    lines.append("")
    ex = run.get("extraction")
    lines.append("## Extracción de entidades y relaciones (F1 contra gold anotado a mano)")
    lines.append("")
    if ex:
        lines.append(
            "| Extractor | Chunks | Entity P | Entity R | Entity F1 | Rel. P | Rel. R | Rel. F1 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        e, r = ex["entity"], ex["relationship"]
        lines.append(
            f"| `{ex['extractor']}` (determinista) | {ex['n_chunks']} | {_pct(e['precision'])} | {_pct(e['recall'])} | {_pct(e['f1'])} | {_pct(r['precision'])} | {_pct(r['recall'])} | {_pct(r['f1'])} |"
        )
        lines.append(
            f"| `claude-tool-use` ({run['model_pinned']}) | — | {PENDING_FAITHFULNESS} | | | | | |"
        )
        lines.append("")
        lines.append(
            "Gold de extracción: `eval/extraction_gold.jsonl` (Exhibit 21 y DEF 14A sintéticos + un chunk real del 10-K de Alphabet), anotado a mano."
        )
    else:
        lines.append("Sin gold de extracción en esta corrida.")
    lines.append("")
    lines.append("## Rúbrica de faithfulness (LLM-as-judge, pendiente)")
    lines.append("")
    for crit in run["faithfulness_rubric"]:
        lines.append(f"- {crit}")
    lines.append("")
    lines.append("## Diez peores casos de GraphRAG")
    lines.append("")
    lines.append("| id | categoría | pregunta | hit | recall | precisión | esperado | obtenido |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in run["worst_cases"]:
        exp = (
            c["expected_count"]
            if c["category"] == "aggregation"
            else ", ".join(c["expected_ids"][:6]) + (" …" if len(c["expected_ids"]) > 6 else "")
        )
        got = (
            c["count"]
            if c["category"] == "aggregation"
            else ", ".join(c["answer_ids"][:6]) + (" …" if len(c["answer_ids"]) > 6 else "")
        )
        lines.append(
            f"| {c['id']} | {c['category']} | {c['question']} | {'sí' if c['hit'] else 'no'} | {c['recall']:.2f} | {c['precision']:.2f} | {exp} | {got} |"
        )
    lines.append("")
    lines.append("Análisis detallado en `docs/error_analysis.md`.")
    lines.append("")
    return "\n".join(lines)
