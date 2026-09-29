"""Embedding backends behind one :class:`Embedder` protocol."""

from src.embeddings.base import Embedder, cosine
from src.embeddings.factory import build_embedder
from src.embeddings.hashing import HashingEmbedder
from src.embeddings.tfidf import TfidfHashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "TfidfHashingEmbedder", "build_embedder", "cosine"]
