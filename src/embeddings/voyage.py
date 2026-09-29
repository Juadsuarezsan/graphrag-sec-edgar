"""Voyage AI embeddings over plain HTTP with timeout and retries.

Never executed in tests against the real API: the transport is mocked with
``respx``.
"""

from __future__ import annotations

from typing import Any

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_DIMS: dict[str, int] = {"voyage-3": 1024, "voyage-3-lite": 512}


class VoyageError(RuntimeError):
    """Raised when Voyage returns a non-retryable error or a malformed body."""


class _RetryableHTTP(RuntimeError):
    """Internal marker for 429/5xx responses."""


class VoyageEmbedder:
    """Client for ``POST /v1/embeddings``.

    Args:
        api_key: Voyage API key.
        model: Pinned model id (``voyage-3``).
        timeout_s: Per-request timeout.
        batch_size: Max texts per HTTP call (Voyage accepts up to 128).
        client: Optional pre-built ``httpx.Client`` (for tests).
    """

    name = "voyage"

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3",
        timeout_s: float = 30.0,
        batch_size: int = 64,
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self.model = model
        self.dims = VOYAGE_DIMS.get(model, 1024)
        self._timeout = timeout_s
        self._batch = batch_size
        self._client = client or httpx.Client(timeout=timeout_s)

    @retry(
        retry=retry_if_exception_type((httpx.TransportError, _RetryableHTTP)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.5, max=8),
        reraise=True,
    )
    def _call(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.post(
            VOYAGE_URL,
            json={"input": texts, "model": self.model, "input_type": "document"},
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._timeout,
        )
        if resp.status_code == 429 or resp.status_code >= 500:
            logger.warning("voyage transient status {}", resp.status_code)
            raise _RetryableHTTP(f"status {resp.status_code}")
        if resp.status_code >= 400:
            raise VoyageError(f"voyage error {resp.status_code}: {resp.text[:200]}")
        body: dict[str, Any] = resp.json()
        try:
            rows = sorted(body["data"], key=lambda r: int(r["index"]))
            return [[float(x) for x in r["embedding"]] for r in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise VoyageError("malformed voyage response") from exc

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches; empty input makes no HTTP call."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._call(texts[i : i + self._batch]))
        return out

    def embed_one(self, text: str) -> list[float]:
        """Embed one text."""
        return self.embed([text])[0]
