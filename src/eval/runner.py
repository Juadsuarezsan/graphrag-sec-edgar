from __future__ import annotations

import argparse
import asyncio
import json

from src.engine.graphrag import answer
from src.eval.queries import QUERIES
from src.graph.fixture import build_demo_graph


async def run_eval() -> dict:
    g = build_demo_graph()
    routing_correct = 0
    citation_correct = 0
    count_correct = 0
    cases = []

    for q in QUERIES:
        ans = await answer(g, q["question"])
        # Routing accuracy
        if ans.classified_kind == q["kind"]:
            routing_correct += 1
        # For lookup/multi_hop: cited node coverage
        if q["kind"] in ("lookup", "multi_hop"):
            cited_ids = {n.id for n in ans.cited_nodes}
            required = set(q.get("must_cite", []))
            citation_correct += int(required.issubset(cited_ids))
        # For aggregation: check count >= expected_min
        if q["kind"] == "aggregation":
            try:
                count = int(ans.answer.split("=")[-1].strip())
                if count >= q.get("expected_count_min", 0):
                    count_correct += 1
            except Exception:
                pass
        cases.append({
            "id": q["id"], "expected_kind": q["kind"], "got_kind": ans.classified_kind,
            "n_cited": len(ans.cited_nodes), "answer_snippet": ans.answer[:120],
        })

    n_lookup_mh = sum(1 for q in QUERIES if q["kind"] in ("lookup", "multi_hop"))
    n_agg = sum(1 for q in QUERIES if q["kind"] == "aggregation")
    return {
        "n": len(QUERIES),
        "routing_accuracy":  routing_correct / len(QUERIES),
        "citation_accuracy": citation_correct / max(n_lookup_mh, 1),
        "aggregation_accuracy": count_correct / max(n_agg, 1),
        "graph_size": {"nodes": g.n_nodes, "edges": g.n_edges},
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    r = asyncio.run(run_eval())
    if args.json:
        print(json.dumps(r, indent=2, default=str))
    else:
        print(f"Queries: {r['n']}  graph: {r['graph_size']}")
        print(f"Routing:     {r['routing_accuracy']:.1%}")
        print(f"Citation:    {r['citation_accuracy']:.1%}")
        print(f"Aggregation: {r['aggregation_accuracy']:.1%}")


if __name__ == "__main__":
    main()
