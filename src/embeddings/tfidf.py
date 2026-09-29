"""TF-IDF embedder with the hashing trick: reproducible Vector RAG baseline.

Pure Python on purpose: it runs in CI without scikit-learn, torch or model
downloads and gives the same vectors on every machine.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

from src.embeddings.base import l2_normalize, tokenize


class TfidfHashingEmbedder:
    """Hashed TF-IDF vectors.

    Tokens are hashed into ``dims`` buckets (with a sign bit to reduce
    collision bias). IDF weights come from :meth:`fit`; before fitting every
    token has IDF 1.0 so the embedder still works as plain hashed TF.
    """

    name = "tfidf-hashing"

    def __init__(self, dims: int = 1024) -> None:
        self.dims = dims
        self._idf: dict[int, float] = {}
        self._n_docs = 0

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % self.dims
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign

    def fit(self, corpus: list[str]) -> TfidfHashingEmbedder:
        """Learn IDF weights from ``corpus`` (one string per document)."""
        df: Counter[int] = Counter()
        for doc in corpus:
            for idx in {self._bucket(t)[0] for t in tokenize(doc)}:
                df[idx] += 1
        self._n_docs = len(corpus)
        self._idf = {idx: math.log((1 + self._n_docs) / (1 + c)) + 1.0 for idx, c in df.items()}
        return self

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self._n_docs > 0

    def embed_one(self, text: str) -> list[float]:
        """Embed one text as an L2-normalised hashed TF-IDF vector."""
        vec = [0.0] * self.dims
        counts = Counter(tokenize(text))
        total = sum(counts.values()) or 1
        for tok, c in counts.items():
            idx, sign = self._bucket(tok)
            tf = c / total
            vec[idx] += sign * tf * self._idf.get(idx, 1.0)
        return l2_normalize(vec)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed_one(t) for t in texts]
