"""Build the configured embedder."""

from __future__ import annotations

from loguru import logger

from src.config import Settings, get_settings
from src.embeddings.base import Embedder
from src.embeddings.hashing import HashingEmbedder
from src.embeddings.tfidf import TfidfHashingEmbedder


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Return the embedder selected by ``EMBEDDER``.

    ``voyage`` without ``VOYAGE_API_KEY`` and ``local`` without the ``[ml]``
    extra fall back to :class:`HashingEmbedder` with a warning, so the API
    always boots.
    """
    s = settings or get_settings()
    if s.embedder == "tfidf":
        return TfidfHashingEmbedder()
    if s.embedder == "voyage":
        if s.voyage_api_key:
            from src.embeddings.voyage import VoyageEmbedder

            return VoyageEmbedder(
                api_key=s.voyage_api_key, model=s.voyage_model, timeout_s=s.request_timeout_s
            )
        logger.warning("EMBEDDER=voyage but VOYAGE_API_KEY missing; using hashing embedder")
    if s.embedder == "local":
        try:
            from src.embeddings.local import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder()
        except ImportError as exc:
            logger.warning("{}; using hashing embedder", exc)
    return HashingEmbedder()
