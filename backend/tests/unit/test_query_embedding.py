"""The request-path embedder: in-memory LRU for prod, and what ``build_provider`` allows."""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from grounded.generation.providers.fake import StubLLMProvider
from grounded.generation.providers.gemini import GeminiProvider
from grounded.ingest.embed import (
    CachedEmbedder,
    Embedder,
    FakeEmbedder,
    GeminiEmbedder,
    LazyEmbedder,
)
from grounded.retrieval.query_embedding import LRUEmbedder, build_query_embedder
from grounded.runtime import ProviderConfigError, build_provider
from tests.support import make_settings


async def test_a_repeated_question_is_embedded_once() -> None:
    inner = FakeEmbedder()
    embedder = LRUEmbedder(inner, max_size=8)
    first = await embedder.embed(["how do I run a task?"], "RETRIEVAL_QUERY")
    again = await embedder.embed(["how do I run a task?"], "RETRIEVAL_QUERY")
    assert first == again
    assert len(inner.calls) == 1


async def test_only_the_missing_texts_reach_the_inner_embedder() -> None:
    inner = FakeEmbedder()
    embedder = LRUEmbedder(inner, max_size=8)
    await embedder.embed(["a"], "RETRIEVAL_QUERY")
    vectors = await embedder.embed(["a", "b", "a"], "RETRIEVAL_QUERY")
    assert inner.calls[1] == (["b"], "RETRIEVAL_QUERY")
    assert vectors[0] == vectors[2] != vectors[1]


async def test_task_type_is_part_of_the_key() -> None:
    inner = FakeEmbedder()
    embedder = LRUEmbedder(inner, max_size=8)
    await embedder.embed(["a"], "RETRIEVAL_QUERY")
    await embedder.embed(["a"], "RETRIEVAL_DOCUMENT")
    assert len(inner.calls) == 2


async def test_least_recently_used_entry_is_evicted() -> None:
    inner = FakeEmbedder()
    embedder = LRUEmbedder(inner, max_size=2)
    await embedder.embed(["a"], "RETRIEVAL_QUERY")
    await embedder.embed(["b"], "RETRIEVAL_QUERY")
    await embedder.embed(["a"], "RETRIEVAL_QUERY")  # a is now the most recent
    await embedder.embed(["c"], "RETRIEVAL_QUERY")  # evicts b
    await embedder.embed(["a"], "RETRIEVAL_QUERY")  # still cached
    await embedder.embed(["b"], "RETRIEVAL_QUERY")  # embedded again
    assert [texts for texts, _ in inner.calls] == [["a"], ["b"], ["c"], ["b"]]


_PROD = {
    "app_env": "prod",
    "database_url": "postgresql://app@db/grounded",
    "proxy_shared_secret": "x" * 32,
    "ip_hash_secret": "y" * 32,
}


def _gemini_behind(embedder: Embedder) -> GeminiEmbedder:
    """Unwrap the cache and the lazy wrapper, building the Gemini embedder (no network)."""
    assert isinstance(embedder, LRUEmbedder | CachedEmbedder)
    lazy = embedder._inner  # pyright: ignore[reportPrivateUsage]
    assert isinstance(lazy, LazyEmbedder)
    built = lazy._factory()  # pyright: ignore[reportPrivateUsage]
    assert isinstance(built, GeminiEmbedder)
    return built


@pytest.mark.parametrize("env", ["dev", "prod"])
async def test_the_request_path_embedder_never_retries_or_paces(env: str, tmp_path: Path) -> None:
    overrides = _PROD if env == "prod" else {"app_env": env}
    settings = make_settings(
        gemini_api_key="k", query_embedding_timeout_s=2.5, cache_dir=tmp_path, **overrides
    )
    async with AsyncExitStack() as stack:
        gemini = _gemini_behind(build_query_embedder(settings, stack))
    assert (
        gemini._max_retries,  # pyright: ignore[reportPrivateUsage]
        gemini._timeout_s,  # pyright: ignore[reportPrivateUsage]
        gemini._paced,  # pyright: ignore[reportPrivateUsage]
    ) == (0, 2.5, False)


def test_the_fake_provider_is_available_outside_prod() -> None:
    settings = make_settings(generator_providers=["fake", "gemini"])
    assert isinstance(build_provider(settings), StubLLMProvider)


def test_the_fake_provider_is_refused_in_prod() -> None:
    settings = make_settings(
        app_env="prod",
        generator_providers=["fake"],
        database_url="postgresql://app@db/grounded",
        proxy_shared_secret="x" * 32,
        ip_hash_secret="y" * 32,
    )
    with pytest.raises(ProviderConfigError, match="prod"):
        build_provider(settings)


def test_gemini_is_built_from_the_first_provider_entry() -> None:
    settings = make_settings(
        generator_providers=["gemini", "groq"],
        gemini_api_key="k-123",
        gemini_model="gemini-3.5-flash-lite",
        gemini_thinking_level="low",
    )
    provider = build_provider(settings)
    assert isinstance(provider, GeminiProvider)
    assert (provider.name, provider.model) == ("gemini", "gemini-3.5-flash-lite")


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"gemini_model": "gemini-3.5-flash-lite"}, "GEMINI_API_KEY"),
        ({"gemini_model": "gemini-3.5-flash-lite", "gemini_api_key": "  "}, "GEMINI_API_KEY"),
        ({"gemini_api_key": "k-123"}, "GEMINI_MODEL"),
        ({"gemini_api_key": "k-123", "gemini_model": " "}, "GEMINI_MODEL"),
    ],
)
def test_gemini_without_a_key_or_a_pinned_model_is_a_clear_error(
    overrides: dict[str, str], missing: str
) -> None:
    with pytest.raises(ProviderConfigError, match=missing):
        build_provider(make_settings(generator_providers=["gemini"], **overrides))


def test_an_unknown_provider_is_a_clear_error() -> None:
    with pytest.raises(ProviderConfigError, match="'groq'"):
        build_provider(make_settings(generator_providers=["groq", "gemini"]))
