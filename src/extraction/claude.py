"""Claude-based extractor (tool use forced to the closed extraction schema).

Real client code with explicit timeout and tenacity retries. The test suite
mocks ``client.messages.create``; nothing here runs against the API without
``ANTHROPIC_API_KEY``.
"""

from __future__ import annotations

from typing import Any

import anthropic
from loguru import logger
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config import DEFAULT_ANTHROPIC_MODEL
from src.extraction.schemas import (
    EDGE_TYPES,
    NODE_TYPES,
    ExtractedEntity,
    ExtractionResult,
    extraction_tool_schema,
)
from src.ingestion.models import SECTION_TITLES, Chunk
from src.observability import estimate_cost_usd

TOOL_NAME = "record_extraction"

SYSTEM_PROMPT = f"""You extract a knowledge graph from SEC filing text.
Return entities and relationships ONLY through the `{TOOL_NAME}` tool.
Rules:
1. Entity types are exactly: {", ".join(NODE_TYPES)}.
2. Relationship types are exactly: {", ".join(EDGE_TYPES)}.
3. Ids are lowercase snake_case slugs of the canonical name (e.g. "taiwan_semiconductor").
   The filer is already known; reuse its id as given.
4. Only assert relationships stated or clearly implied in the text. Put the supporting
   sentence in `evidence` (max 500 chars). Never infer from outside knowledge.
5. Skip boilerplate, page numbers, table-of-contents lines and financial statement rows.
"""


class ClaudeExtractionError(RuntimeError):
    """Raised when the model returns no valid tool call after retries."""


class ClaudeExtractor:
    """Entity extractor backed by ``claude-sonnet-4-5-20250929``.

    Args:
        api_key: Anthropic key (required unless ``client`` is injected).
        model: Dated model id.
        timeout_s: Per-call timeout passed to the SDK.
        max_tokens: Output cap; extraction of a 2k-char chunk fits in ~1.5k tokens.
        client: Pre-built ``anthropic.Anthropic`` (tests inject a mock).
    """

    name = "claude-tool-use"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        timeout_s: float = 60.0,
        max_tokens: int = 2048,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        if client is None and not api_key:
            raise ValueError("ClaudeExtractor needs an api_key or an injected client")
        self.model = model
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout_s, max_retries=0
        )

    @retry(
        retry=retry_if_exception_type(
            (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.InternalServerError)
        ),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, max=20),
        reraise=True,
    )
    def _create(self, user_prompt: str) -> anthropic.types.Message:
        return self._client.messages.create(
            model=self.model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            tools=[
                {
                    "name": TOOL_NAME,
                    "description": "Record the entities and relationships found in the text.",
                    "input_schema": extraction_tool_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": user_prompt}],
        )

    @staticmethod
    def _prompt(chunk: Chunk, focal: ExtractedEntity) -> str:
        section = SECTION_TITLES.get(chunk.item, chunk.item)
        return (
            f"Filer: {focal.name} (id: {focal.id}, type: Company, ticker {chunk.filing.ticker}).\n"
            f"Filing: {chunk.filing.form_type} {chunk.filing.accession}, section Item {chunk.item} ({section}).\n\n"
            f"TEXT:\n{chunk.text}"
        )

    def extract(self, chunk: Chunk, *, focal: ExtractedEntity) -> ExtractionResult:
        """Extract from one chunk; the filer is always included in the result."""
        message = self._create(self._prompt(chunk, focal))
        payload: dict[str, Any] | None = None
        for block in message.content:
            if block.type == "tool_use" and block.name == TOOL_NAME:
                payload = dict(block.input) if isinstance(block.input, dict) else None
                break
        if payload is None:
            raise ClaudeExtractionError(f"no {TOOL_NAME} tool call in response for {chunk.id}")
        usage_in = int(getattr(message.usage, "input_tokens", 0))
        usage_out = int(getattr(message.usage, "output_tokens", 0))
        try:
            result = ExtractionResult(
                chunk_id=chunk.id,
                extractor=self.name,
                entities=[focal, *payload.get("entities", [])],
                relationships=payload.get("relationships", []),
                tokens_in=usage_in,
                tokens_out=usage_out,
                cost_usd=estimate_cost_usd(self.model, usage_in, usage_out),
            )
        except ValidationError as exc:
            logger.warning(
                "{}: dropping invalid extraction ({} errors)", chunk.id, exc.error_count()
            )
            raise ClaudeExtractionError(str(exc)) from exc
        logger.info(
            "{}: {} entities, {} relationships, tokens {}/{}, cost_usd={}",
            chunk.id,
            len(result.entities),
            len(result.relationships),
            usage_in,
            usage_out,
            result.cost_usd,
        )
        return result
