import pytest

from src.engine.graphrag import answer
from src.graph.fixture import build_demo_graph


@pytest.mark.asyncio
async def test_lookup_ceo_apple():
    g = build_demo_graph()
    ans = await answer(g, "Who is the CEO of Apple?")
    assert ans.classified_kind == "lookup"
    cited_ids = {n.id for n in ans.cited_nodes}
    assert "apple" in cited_ids or "tim_cook" in cited_ids


@pytest.mark.asyncio
async def test_multi_hop_tsmc_suppliers():
    g = build_demo_graph()
    ans = await answer(g, "Which companies are supplied by TSMC?")
    assert ans.classified_kind == "multi_hop"
    cited_ids = {n.id for n in ans.cited_nodes}
    # We expect at least TSMC and one of its customers in the subgraph
    assert "tsmc" in cited_ids
    assert any(x in cited_ids for x in ("apple", "nvidia", "amd", "tesla"))


@pytest.mark.asyncio
async def test_aggregation_company_count():
    g = build_demo_graph()
    ans = await answer(g, "How many S&P 500 companies are there?")
    assert ans.classified_kind == "aggregation"
    count = int(ans.answer.split("=")[-1].strip())
    assert count >= 10
