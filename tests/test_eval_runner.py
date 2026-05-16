import pytest

from src.eval.runner import run_eval


@pytest.mark.asyncio
async def test_routing_above_threshold():
    r = await run_eval()
    assert r["routing_accuracy"] >= 0.80
    assert r["aggregation_accuracy"] >= 0.80
