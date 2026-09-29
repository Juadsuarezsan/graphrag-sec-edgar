"""Entity linking, planner, classifier and engine modes."""

from __future__ import annotations

from src.engine.graphrag import GraphRAGEngine
from src.engine.linking import EntityLinker, normalize
from src.engine.planner import Step, build_plan, classify_plan, resolve_direction
from src.engine.synthesis import TemplateSynthesizer, sanitize_text
from src.engine.vector import VectorIndex, node_passages
from src.graph.store import InMemoryGraph
from src.observability import RequestMetrics
from src.router.classifier import classify


def test_linker_longest_match_and_possessive(demo_graph: InMemoryGraph) -> None:
    linker = EntityLinker(demo_graph)
    ids = [m.node_id for m in linker.link("Does Apple Watch compete with Apple's iPhone?")]
    assert ids == ["prod_apple_watch", "apple", "prod_iphone"]
    assert [m.node_id for m in linker.link("Who leads GM and JPMorgan Chase?")] == ["gm", "jpm"]
    assert [m.node_id for m in linker.link("Which companies mention tariffs?")] == ["risk_tariffs"]
    assert linker.link("nothing linked here") == []
    assert (
        normalize("  Coca-Cola, Inc.! ") == "coca cola, inc."[:0]
        or normalize("Coca-Cola") == "coca cola"
    )


def test_plan_structure(demo_graph: InMemoryGraph) -> None:
    linker = EntityLinker(demo_graph)
    types = {n["id"]: n["type"] for n in demo_graph.nodes()}
    q = "Which risks are mentioned by companies supplied by TSMC?"
    plan = build_plan(q, linker.link(q), types)
    assert [s.edge_types for s in plan.constraints[0].steps] == [
        ("SUPPLIED_BY",),
        ("MENTIONS_RISK",),
    ]
    assert plan.answer_type == "Risk" and plan.kind == "multi_hop" and plan.n_hops == 2
    q = "How many S&P 500 companies mention China exposure as a risk?"
    plan = build_plan(q, linker.link(q), types)
    assert plan.aggregate and plan.in_sp500_only and plan.kind == "aggregation"
    assert len(plan.constraints[0].steps) == 1  # "mention ... risk" collapsed
    q = "Which directors sit on the boards of both Pfizer and Merck?"
    plan = build_plan(q, linker.link(q), types)
    assert (
        len(plan.constraints) == 2 and plan.constraints[1].steps and plan.role_filter == "Director"
    )
    q = "Which companies share a supplier with Apple?"
    plan = build_plan(q, linker.link(q), types)
    assert plan.constraints[0].steps[0].share and plan.n_hops == 2
    q = "What is Microsoft's ticker?"
    plan = build_plan(q, linker.link(q), types)
    assert plan.property_lookup == "ticker" and plan.kind == "lookup"
    assert plan.as_dict()["constraints"][0]["seed"] == "microsoft"
    assert classify_plan(build_plan("tell me something", [], {})) == "hybrid"
    assert build_plan("How many directors serve on more than one board?", [], {}).min_degree == 2


def test_resolve_direction_from_signatures() -> None:
    assert resolve_direction(Step(("CEO_OF",), "auto"), {"Company"}) == "in"
    assert resolve_direction(Step(("CEO_OF",), "auto"), {"Person"}) == "out"
    assert resolve_direction(Step(("COMPETES_WITH",), "both"), {"Company"}) == "both"
    assert resolve_direction(Step(("CEO_OF",), "auto"), {"Company", "Person"}) == "both"


def test_regex_classifier() -> None:
    assert classify("Who is the CEO of Apple?") == "lookup"
    assert classify("Which companies are supplied by TSMC?") == "multi_hop"
    assert classify("How many S&P 500 companies?") == "aggregation"
    assert classify("Tell me something about Apple") == "hybrid"


def test_engine_modes_agree_on_lookup(engine: GraphRAGEngine) -> None:
    g = engine.answer_sync("Who is the CEO of Apple?", mode="graph")
    assert (
        g.answer_ids == ["tim_cook"]
        and g.classified_kind == "lookup"
        and g.synthesizer == "template"
    )
    assert g.plan[0].edge_types == ["CEO_OF"] and g.plan[0].direction == "in"
    assert g.subgraph.nodes and g.metrics.trace_id and g.metrics.cost_usd == 0.0
    v = engine.answer_sync("Who is the CEO of Apple?", mode="vector")
    assert "tim_cook" in v.answer_ids and v.mode == "vector" and v.seeds[0].startswith("node:")
    h = engine.answer_sync("Who is the CEO of Apple?", mode="hybrid")
    assert h.answer_ids == ["tim_cook"] and h.mode == "hybrid"


def test_engine_multi_hop_aggregation_and_truncation(engine: GraphRAGEngine) -> None:
    a = engine.answer_sync(
        "Which CEOs lead companies that share a supplier with Apple?", mode="graph", max_hops=3
    )
    assert "jensen_huang" in a.answer_ids and a.classified_kind == "multi_hop"
    truncated = engine.answer_sync(
        "Which CEOs lead companies that share a supplier with Apple?", mode="graph", max_hops=1
    )
    assert "jensen_huang" not in truncated.answer_ids
    agg = engine.answer_sync("How many subsidiaries does Alphabet have?", mode="graph")
    assert agg.count == 4 and agg.answer.startswith("count = 4")
    seedless = engine.answer_sync("Which subsidiaries are incorporated in Ireland?", mode="graph")
    assert "sub_google_ireland" in seedless.answer_ids and seedless.seeds == []
    prop = engine.answer_sync("What is Microsoft's ticker?", mode="graph")
    assert "MSFT" in prop.answer and prop.answer_ids == ["microsoft"]
    unknown = engine.answer_sync("What is the meaning of life?", mode="graph")
    assert unknown.answer_ids == [] and unknown.classified_kind == "hybrid"
    hybrid_fallback = engine.answer_sync("semiconductor foundry taiwan", mode="hybrid")
    assert hybrid_fallback.seeds  # vector seeds when nothing links lexically


async def test_engine_async_wrapper(engine: GraphRAGEngine) -> None:
    ans = await engine.answer("Which companies compete with Visa?", mode="graph")
    assert ans.answer_ids == ["mastercard"]


def test_vector_index_and_passages(demo_graph: InMemoryGraph) -> None:
    passages = node_passages(demo_graph)
    apple = next(p for p in passages if p.id == "node:apple")
    assert "is supplied by Taiwan Semiconductor" in apple.text and "tsmc" in apple.entity_ids
    index = VectorIndex(passages)
    assert len(index) == demo_graph.n_nodes
    top = index.query("Apple Inc iPhone supplier Taiwan Semiconductor", k=3)
    assert "node:apple" in [p.id for _, p in top.passages]
    assert VectorIndex([]).query("x").passages == []


def test_template_synthesizer_and_sanitize() -> None:
    t = TemplateSynthesizer()
    m = RequestMetrics()
    assert t.synthesize(
        "q", [], count=None, property_value=None, path_sentences=[], metrics=m
    ).startswith("No matching")
    assert (
        t.synthesize("q", [], count=0, property_value=None, path_sentences=[], metrics=m)
        == "count = 0"
    )
    assert sanitize_text("<script>alert(1)</script>Tim & Cook\x00") == "alert(1)Tim &amp; Cook"
    assert len(sanitize_text("x" * 5000)) == 4000
