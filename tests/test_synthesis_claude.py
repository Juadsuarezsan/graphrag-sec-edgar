"""ClaudeSynthesizer with a mocked SDK client (never hits the API)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from src.engine.synthesis import ClaudeSynthesizer
from src.graph.fixture import build_demo_graph
from src.observability import RequestMetrics


def _msg(text: str) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=800, output_tokens=120),
    )


def test_claude_synthesizer_records_usage_and_sanitizes(mocker: Any) -> None:
    client = mocker.MagicMock()
    client.messages.create.return_value = _msg("Tim Cook [tim_cook] leads Apple <b>Inc.</b>")
    synth = ClaudeSynthesizer(client=client)
    g = build_demo_graph()
    metrics = RequestMetrics(name="query")
    node = g.get_node("tim_cook")
    assert node is not None
    text = synth.synthesize(
        "Who is the CEO of Apple?",
        [node],
        count=None,
        property_value=None,
        path_sentences=["Tim Cook -CEO_OF-> Apple"],
        metrics=metrics,
    )
    assert text == "Tim Cook [tim_cook] leads Apple Inc."
    assert (
        metrics.llm_calls == 1
        and metrics.tokens_in == 800
        and metrics.cost_usd == pytest.approx(0.0042)
    )
    kwargs = client.messages.create.call_args.kwargs
    assert (
        kwargs["model"] == "claude-sonnet-4-5-20250929"
        and "Facts:" in kwargs["messages"][0]["content"]
    )


def test_claude_synthesizer_retries_and_requires_key(mocker: Any) -> None:
    with pytest.raises(ValueError):
        ClaudeSynthesizer()
    client = mocker.MagicMock()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    client.messages.create.side_effect = [anthropic.APIConnectionError(request=request), _msg("")]
    synth = ClaudeSynthesizer(client=client)
    synth._create.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    text = synth.synthesize(
        "q", [], count=2, property_value="x", path_sentences=[], metrics=RequestMetrics()
    )
    assert text == "The model returned an empty answer." and client.messages.create.call_count == 2
