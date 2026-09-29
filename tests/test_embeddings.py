"""Embedder backends: determinism, dims, Voyage HTTP (respx), lazy local import, factory."""

from __future__ import annotations

import sys
import types

import httpx
import pytest
import respx

from src.config import get_settings
from src.embeddings import HashingEmbedder, TfidfHashingEmbedder, build_embedder, cosine
from src.embeddings.base import Embedder, l2_normalize
from src.embeddings.local import SentenceTransformerEmbedder
from src.embeddings.voyage import VOYAGE_URL, VoyageEmbedder, VoyageError


def test_hashing_is_deterministic_and_normalised() -> None:
    e = HashingEmbedder(dims=64)
    a, b = e.embed_one("Apple Inc"), e.embed_one("Apple Inc")
    assert a == b and len(a) == 64
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6
    assert isinstance(e, Embedder)


def test_hashing_similarity_orders_overlap() -> None:
    e = HashingEmbedder()
    q = e.embed_one("apple iphone")
    assert cosine(q, e.embed_one("apple iphone mac")) > cosine(q, e.embed_one("exxon crude oil"))
    assert e.embed([]) == []
    assert e.embed_one("") == e.embed_one("   ")


def test_tfidf_fit_changes_weights_and_batches() -> None:
    e = TfidfHashingEmbedder(dims=128)
    assert not e.is_fitted
    before = e.embed_one("tsmc supplies apple")
    e.fit(["tsmc supplies apple", "tsmc supplies nvidia", "coca cola beverages"])
    assert e.is_fitted
    after = e.embed_one("tsmc supplies apple")
    assert before != after
    assert len(e.embed(["a", "b"])) == 2
    assert cosine([], [1.0]) == 0.0 and cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


@respx.mock
def test_voyage_embeds_in_batches_and_sorts_by_index() -> None:
    route = respx.post(VOYAGE_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0.0, 1.0]},
                        {"index": 0, "embedding": [1.0, 0.0]},
                    ]
                },
            ),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.5, 0.5]}]}),
        ]
    )
    e = VoyageEmbedder(api_key="k", batch_size=2, client=httpx.Client())
    out = e.embed(["a", "b", "c"])
    assert out == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert route.call_count == 2
    assert route.calls[0].request.headers["authorization"] == "Bearer k"


@respx.mock
def test_voyage_retries_on_429_then_succeeds() -> None:
    route = respx.post(VOYAGE_URL).mock(
        side_effect=[
            httpx.Response(429),
            httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]}),
        ]
    )
    e = VoyageEmbedder(api_key="k", client=httpx.Client())
    e._call.retry.wait = lambda *_: 0  # type: ignore[attr-defined]
    assert e.embed_one("x") == [1.0]
    assert route.call_count == 2


@respx.mock
def test_voyage_400_and_malformed_raise() -> None:
    respx.post(VOYAGE_URL).mock(return_value=httpx.Response(400, text="bad"))
    e = VoyageEmbedder(api_key="k", client=httpx.Client())
    with pytest.raises(VoyageError):
        e.embed_one("x")
    respx.post(VOYAGE_URL).mock(return_value=httpx.Response(200, json={"nope": []}))
    with pytest.raises(VoyageError):
        e.embed_one("x")
    assert e.embed([]) == []


def test_local_embedder_uses_lazy_import(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def __init__(self, name: str) -> None:
            self.name = name

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(
            self, texts: list[str], batch_size: int, normalize_embeddings: bool
        ) -> list[list[float]]:
            return [[1.0, 0.0, 0.0] for _ in texts]

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    e = SentenceTransformerEmbedder("fake/model")
    assert e.dims == 3 and e.embed_one("hi") == [1.0, 0.0, 0.0] and e.embed([]) == []


def test_factory_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDER", "tfidf")
    get_settings.cache_clear()
    assert isinstance(build_embedder(), TfidfHashingEmbedder)
    monkeypatch.setenv("EMBEDDER", "voyage")
    get_settings.cache_clear()
    assert isinstance(build_embedder(), HashingEmbedder)  # no key -> fallback
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    get_settings.cache_clear()
    assert isinstance(build_embedder(), VoyageEmbedder)
    monkeypatch.setenv("EMBEDDER", "local")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # type: ignore[arg-type]
    get_settings.cache_clear()
    assert isinstance(build_embedder(), HashingEmbedder)
