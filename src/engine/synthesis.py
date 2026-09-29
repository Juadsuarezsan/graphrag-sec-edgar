"""Answer synthesis: deterministic template (default) or Claude with citations.

The template path is what every stored eval run uses (no LLM); it is labelled
``synthesizer="template"`` in every response so it is never confused with
model output. :class:`ClaudeSynthesizer` is real client code, exercised in
tests with a mocked SDK client.
"""

from __future__ import annotations

import html
import re
from typing import Protocol

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import DEFAULT_ANTHROPIC_MODEL
from src.graph.base import Node
from src.observability import RequestMetrics

_TAG_RE = re.compile(r"<[^>]+>")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_text(text: str, max_chars: int = 4000) -> str:
    """Strip HTML tags and control characters from model or template output.

    The API returns plain text (JSON); the demo renders it with ``textContent``.
    Tags are removed rather than escaped so ``->`` arrows and ``&`` in company
    names survive intact.
    """
    cleaned = _CTRL_RE.sub("", _TAG_RE.sub("", text))
    return html.unescape(cleaned)[:max_chars]


class Synthesizer(Protocol):
    """Turns a plan result into prose."""

    name: str

    def synthesize(
        self,
        question: str,
        answer_nodes: list[Node],
        *,
        count: int | None,
        property_value: str | None,
        path_sentences: list[str],
        metrics: RequestMetrics,
    ) -> str:
        """Produce the final answer text."""
        ...


class TemplateSynthesizer:
    """Deterministic, citation-preserving template."""

    name = "template"

    def synthesize(
        self,
        question: str,
        answer_nodes: list[Node],
        *,
        count: int | None,
        property_value: str | None,
        path_sentences: list[str],
        metrics: RequestMetrics,
    ) -> str:
        """Render ``count``, a property value or the list of answer nodes."""
        if count is not None:
            names = ", ".join(n["name"] for n in answer_nodes[:8])
            more = f" (+{len(answer_nodes) - 8} more)" if len(answer_nodes) > 8 else ""
            return sanitize_text(f"count = {count}" + (f" — {names}{more}" if names else ""))
        if property_value is not None:
            return sanitize_text(property_value)
        if not answer_nodes:
            return "No matching nodes found in the knowledge graph."
        names = ", ".join(f"{n['name']} [{n['id']}]" for n in answer_nodes[:12])
        more = f" (+{len(answer_nodes) - 12} more)" if len(answer_nodes) > 12 else ""
        evidence = (" Evidence: " + " ".join(path_sentences[:4])) if path_sentences else ""
        return sanitize_text(f"{names}{more}.{evidence}")


class ClaudeSynthesizer:
    """Natural-language synthesis with node citations via Claude.

    Args:
        api_key: Anthropic key (or inject ``client``).
        model: Dated model id.
        timeout_s: SDK timeout.
        client: Pre-built client (tests).
    """

    name = "claude"
    SYSTEM = (
        "You answer questions about public companies strictly from the graph facts provided. "
        "Cite node ids in square brackets after each claim, e.g. 'Tim Cook [tim_cook]'. "
        "If the facts do not answer the question, say so. Never add outside knowledge."
    )

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        timeout_s: float = 30.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("ClaudeSynthesizer needs an api_key or an injected client")
        self.model = model
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        reraise=True,
    )
    def _create(self, prompt: str) -> anthropic.types.Message:
        return self._client.messages.create(
            model=self.model,
            max_tokens=600,
            system=self.SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

    def synthesize(
        self,
        question: str,
        answer_nodes: list[Node],
        *,
        count: int | None,
        property_value: str | None,
        path_sentences: list[str],
        metrics: RequestMetrics,
    ) -> str:
        """Call Claude with the facts and record token usage on ``metrics``."""
        facts = "\n".join(f"- {s}" for s in path_sentences[:40]) or "- (no relationships)"
        nodes = ", ".join(f"{n['name']} [{n['id']}]" for n in answer_nodes[:30]) or "(none)"
        extra = f"\nAggregate count: {count}" if count is not None else ""
        extra += f"\nProperty value: {property_value}" if property_value else ""
        prompt = f"Question: {question}\n\nAnswer nodes: {nodes}\n\nFacts:\n{facts}{extra}\n\nAnswer in 2-4 sentences with citations."
        message = self._create(prompt)
        metrics.add_usage(
            self.model,
            int(getattr(message.usage, "input_tokens", 0)),
            int(getattr(message.usage, "output_tokens", 0)),
        )
        text = "".join(
            getattr(block, "text", "") for block in message.content if block.type == "text"
        )
        return sanitize_text(text.strip() or "The model returned an empty answer.")
