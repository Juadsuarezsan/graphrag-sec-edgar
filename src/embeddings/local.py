"""Local ``sentence-transformers`` embedder (optional ``[ml]`` extra).

The heavy import happens inside :meth:`SentenceTransformerEmbedder.__init__`
so that importing this module never pulls torch. Excluded from coverage;
exercised in tests through a fake ``sentence_transformers`` module.
"""

from __future__ import annotations

import importlib
from typing import Any

#: Pinned model id; ``bge-large-en-v1.5`` is the spec's local alternative.
DEFAULT_LOCAL_MODEL = "BAAI/bge-large-en-v1.5"


class SentenceTransformerEmbedder:
    """Wrapper around ``sentence_transformers.SentenceTransformer``.

    Raises:
        ImportError: with an actionable message when the ``[ml]`` extra is
            not installed.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL, batch_size: int = 32) -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:  # pragma: no cover - depends on extra
            raise ImportError(
                "sentence-transformers is not installed; run `pip install -e '.[ml]'`"
            ) from exc
        self._model: Any = module.SentenceTransformer(model_name)
        self.model_name = model_name
        self._batch = batch_size
        self.dims = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch (normalised, batched by ``batch_size``)."""
        if not texts:
            return []
        vectors = self._model.encode(texts, batch_size=self._batch, normalize_embeddings=True)
        return [[float(x) for x in row] for row in vectors]

    def embed_one(self, text: str) -> list[float]:
        """Embed one text."""
        return self.embed([text])[0]
