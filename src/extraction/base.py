"""Extractor contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.extraction.schemas import ExtractedEntity, ExtractionResult
from src.ingestion.models import Chunk


@runtime_checkable
class EntityExtractor(Protocol):
    """Turns one chunk into validated entities and relationships.

    ``focal`` is the filer itself, so extractors can attach ``OFFERS_PRODUCT``,
    ``MENTIONS_RISK`` etc. to the right company without guessing.
    """

    name: str

    def extract(self, chunk: Chunk, *, focal: ExtractedEntity) -> ExtractionResult:
        """Extract from ``chunk``."""
        ...
