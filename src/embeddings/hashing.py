"""Hashing embedder: no training data, no downloads, fully deterministic."""

from __future__ import annotations

import hashlib

from src.embeddings.base import l2_normalize, tokenize


class HashingEmbedder:
    """Bag-of-hashed-tokens embedder (SHA-256 feature hashing).

    Each token contributes a pseudo-random signed vector derived from its
    hash, so identical token sets map to identical vectors and overlapping
    token sets are close in cosine space. This is the default backend for
    tests and for the demo, and the fallback when no API key is present.
    """

    name = "hashing-sha256"

    def __init__(self, dims: int = 256) -> None:
        self.dims = dims

    def _token_vector(self, token: str) -> list[float]:
        digest = hashlib.sha256(token.encode()).digest()
        raw = (digest * (self.dims // len(digest) + 1))[: self.dims]
        return [(b - 128) / 128.0 for b in raw]

    def embed_one(self, text: str) -> list[float]:
        """Embed one text; empty text maps to the ``<empty>`` token vector."""
        tokens = tokenize(text) or ["<empty>"]
        acc = [0.0] * self.dims
        for tok in tokens:
            for i, v in enumerate(self._token_vector(tok)):
                acc[i] += v
        return l2_normalize([x / len(tokens) for x in acc])

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed_one(t) for t in texts]
