"""The request-path embedder: in-memory LRU for prod, and what ``build_provider`` allows."""

from __future__ import annotations

import pytest

from grounded.generation.providers.fake import StubLLMProvider
from grounded.ingest.embed import FakeEmbedder
from grounded.retrieval.query_embedding import LRUEmbedder
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


def test_a_provider_without_an_adapter_is_a_clear_error() -> None:
    with pytest.raises(ProviderConfigError, match="gemini"):
        build_provider(make_settings(generator_providers=["gemini", "groq"]))
