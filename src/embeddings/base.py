"""Embedder protocol and shared vector helpers."""

from __future__ import annotations

import math
import re
from typing import Protocol, runtime_checkable

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer shared by the local embedders."""
    return _TOKEN_RE.findall(text.lower())


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity; 0.0 for empty or mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


def l2_normalize(vec: list[float]) -> list[float]:
    """Return the unit-norm copy of ``vec`` (zero vector stays zero)."""
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


@runtime_checkable
class Embedder(Protocol):
    """Batch text embedder.

    Implementations must be deterministic for a fixed configuration and must
    return vectors of exactly ``dims`` floats, one per input text.
    """

    name: str
    dims: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (empty input returns an empty list)."""
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        ...
