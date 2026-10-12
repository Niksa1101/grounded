"""The parts of ``retrieval/rerank.py`` that need no Cohere at all: the ranking, the fake, the
cache and the error codes. ``test_cohere_reranker.py`` covers the adapter and the cache over it.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import pytest

from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.retrieval import rerank
from grounded.retrieval.rerank import (
    CachedReranker,
    CohereReranker,
    FakeReranker,
    Hang,
    RerankErrorCode,
    fake_score,
    rank_chunks,
    rerank_cache_key,
    rerank_document,
    rerank_error_code,
)
from grounded.runtime import ProviderConfigError, build_reranker
from tests.rerank_rig import QUERY, chunks
from tests.support import make_settings

# --- The document and the ranking ----------------------------------------------------------------


def test_the_document_is_the_breadcrumb_a_blank_line_and_the_content() -> None:
    (chunk,) = chunks(1)

    assert rerank_document(chunk) == "Tutorial > Page > Section 0\n\nContent of section 0."


def test_rank_chunks_orders_by_score_and_sets_it_on_copies() -> None:
    candidates = chunks(4)

    ranked = rank_chunks(candidates, [0.1, 0.9, 0.5, 0.7], top_n=3)

    assert [(c.chunk_id, c.rerank_score) for c in ranked] == [(1, 0.9), (3, 0.7), (2, 0.5)]
    assert all(c.rerank_score is None for c in candidates)


def test_rank_chunks_breaks_ties_by_input_order() -> None:
    ranked = rank_chunks(chunks(5), [0.5, 0.9, 0.5, 0.5, 0.9], top_n=5)

    assert [c.chunk_id for c in ranked] == [1, 4, 0, 2, 3]


def test_rank_chunks_overwrites_an_earlier_score() -> None:
    stale = [replace(c, rerank_score=0.99) for c in chunks(2)]

    ranked = rank_chunks(stale, [0.2, 0.3], top_n=2)

    assert [c.rerank_score for c in ranked] == [0.3, 0.2]


def test_rank_chunks_needs_one_score_per_chunk() -> None:
    with pytest.raises(ValueError, match="2 scores for 3 chunks"):
        rank_chunks(chunks(3), [0.1, 0.2], top_n=3)


# --- The fake ------------------------------------------------------------------------------------


async def test_a_bare_fake_scores_deterministically_and_ranks_the_same_every_time() -> None:
    first = await FakeReranker().rerank(QUERY, chunks(6), 6)
    second = await FakeReranker().rerank(QUERY, chunks(6), 6)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert all(c.rerank_score is not None and 0.0 <= c.rerank_score < 1.0 for c in first)
    assert fake_score("q", "d") == fake_score("q", "d") != fake_score("q", "e")


async def test_a_fake_follows_its_script_and_records_every_call() -> None:
    fake = FakeReranker([[0.1, 0.9], [0.8, 0.2]])

    ranked = await fake.rerank("first", chunks(2), 2)
    again = await fake.rerank("second", chunks(2), 1)

    assert [c.chunk_id for c in ranked] == [1, 0]
    assert [c.chunk_id for c in again] == [0]
    assert [call.query for call in fake.calls] == ["first", "second"]
    assert fake.calls[0].documents == tuple(rerank_document(c) for c in chunks(2))


@pytest.mark.parametrize(
    "error",
    [
        ProviderRateLimited("slow", retry_after_s=1.0, is_quota=False),
        ProviderUnavailable("down"),
        ProviderTimeout("late"),
        ProviderRequestRejected("no", status_code=404),
        ProviderBadOutput("garbled"),
    ],
    ids=lambda error: type(error).__name__,
)
async def test_a_fake_raises_a_scripted_error_and_records_the_call(error: ProviderError) -> None:
    fake = FakeReranker([error])

    with pytest.raises(type(error)):
        await fake.rerank(QUERY, chunks(3), 3)

    assert len(fake.calls) == 1


async def test_a_hanging_fake_ends_only_with_the_callers_timeout() -> None:
    fake = FakeReranker([Hang()])

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await fake.rerank(QUERY, chunks(3), 3)


async def test_a_fake_refuses_a_step_that_does_not_fit_the_chunks() -> None:
    with pytest.raises(ValueError, match="one score"):
        await FakeReranker([[0.5]]).rerank(QUERY, chunks(2), 2)
    with pytest.raises(ValueError, match="one score"):
        await FakeReranker([[0.5, 1.5]]).rerank(QUERY, chunks(2), 2)


async def test_a_fake_makes_no_call_for_no_chunks_and_refuses_a_bad_top_n() -> None:
    fake = FakeReranker([ProviderTimeout("late")])

    assert await fake.rerank(QUERY, [], 3) == []
    with pytest.raises(ValueError, match="top_n"):
        await fake.rerank(QUERY, chunks(2), 0)

    assert fake.calls == []


# --- The error codes -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProviderRateLimited("slow", retry_after_s=None, is_quota=False), "rate_limited"),
        (ProviderRateLimited("slow", retry_after_s=3.0, is_quota=True), "rate_limited"),
        (ProviderTimeout("late"), "timeout"),
        (ProviderUnavailable("down"), "unavailable"),
        (ProviderRequestRejected("no", status_code=401), "rejected"),
        (ProviderBadOutput("garbled"), "bad_output"),
        (ProviderError("something else"), "unavailable"),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, ProviderError) else value,
)
def test_each_provider_error_has_its_code(error: ProviderError, code: RerankErrorCode) -> None:
    assert rerank_error_code(error) == code


# --- The cache -----------------------------------------------------------------------------------


def key(
    *, model: str = "m", query: str = "q", count: int = 3, order: list[int] | None = None
) -> str:
    candidates = chunks(count)
    return rerank_cache_key(model, query, [candidates[i] for i in order or range(count)])


def test_the_key_is_stable_and_changes_with_each_part() -> None:
    assert key() == key()
    assert len({key(), key(model="n"), key(query="r"), key(count=4)}) == 4


def test_the_key_depends_on_the_order_of_the_candidates() -> None:
    # Scores are stored by position, so the same set in another order is another entry.
    assert key(order=[0, 1, 2]) != key(order=[2, 1, 0])


def test_the_key_depends_on_the_content_hash_not_the_other_fields() -> None:
    candidates = chunks(2)
    moved = [replace(candidates[0], url="https://elsewhere.example/"), candidates[1]]
    edited = [replace(candidates[0], content_hash="changed"), candidates[1]]

    assert rerank_cache_key("m", "q", moved) == rerank_cache_key("m", "q", candidates)
    assert rerank_cache_key("m", "q", edited) != rerank_cache_key("m", "q", candidates)


def test_the_key_depends_on_how_the_document_text_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    before = key()

    monkeypatch.setattr(rerank, "DOCUMENT_FORMAT_VERSION", rerank.DOCUMENT_FORMAT_VERSION + 1)

    assert key() != before


def test_a_separator_in_the_query_cannot_shift_into_a_neighbor() -> None:
    (chunk,) = chunks(1)

    assert rerank_cache_key("m", 'a","h0', [chunk]) != rerank_cache_key("m", "a", [chunk])
    assert rerank_cache_key("m|q", "", [chunk]) != rerank_cache_key("m", "q", [chunk])


async def test_a_cache_hit_asks_the_inner_reranker_nothing(tmp_path: Path) -> None:
    fake = FakeReranker()
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)

        first = await cached.rerank(QUERY, chunks(6), 6)
        second = await cached.rerank(QUERY, chunks(6), 6)

    assert len(fake.calls) == 1
    assert second == first
    assert (cached.hits, cached.misses) == (1, 1)


async def test_the_entry_has_no_top_n_so_it_serves_any(tmp_path: Path) -> None:
    fake = FakeReranker()
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)

        everything = await cached.rerank(QUERY, chunks(6), 6)
        top_two = await cached.rerank(QUERY, chunks(6), 2)
        beyond = await cached.rerank(QUERY, chunks(6), 99)

    assert len(fake.calls) == 1
    assert top_two == everything[:2]
    assert beyond == everything


async def test_a_changed_order_model_or_query_is_a_miss(tmp_path: Path) -> None:
    fake = FakeReranker()
    other_model = FakeReranker(model="another-model")
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)
        await cached.rerank(QUERY, chunks(4), 4)

        await cached.rerank(QUERY, list(reversed(chunks(4))), 4)
        await cached.rerank("another question", chunks(4), 4)
        await CachedReranker(other_model, cache).rerank(QUERY, chunks(4), 4)

    assert len(fake.calls) == 3
    assert len(other_model.calls) == 1


async def test_a_new_document_format_version_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeReranker()
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)
        await cached.rerank(QUERY, chunks(4), 4)

        monkeypatch.setattr(rerank, "DOCUMENT_FORMAT_VERSION", rerank.DOCUMENT_FORMAT_VERSION + 1)
        await cached.rerank(QUERY, chunks(4), 4)

    assert len(fake.calls) == 2


async def test_the_entry_survives_a_new_cache_object_on_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / "rerank.sqlite"
    fake = FakeReranker()
    with KVCache(path) as cache:
        first = await CachedReranker(fake, cache).rerank(QUERY, chunks(5), 5)
    with KVCache(path) as cache:
        second = await CachedReranker(fake, cache).rerank(QUERY, chunks(5), 5)

    assert len(fake.calls) == 1
    assert second == first


async def test_equal_scores_rank_the_same_from_the_cache_as_live(tmp_path: Path) -> None:
    fake = FakeReranker([[0.5, 0.5, 0.9, 0.5]])
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)

        live = await cached.rerank(QUERY, chunks(4), 4)
        replayed = await cached.rerank(QUERY, chunks(4), 4)

    assert [c.chunk_id for c in live] == [2, 0, 1, 3]
    assert replayed == live


async def test_an_error_from_the_inner_reranker_is_raised_and_not_stored(tmp_path: Path) -> None:
    fake = FakeReranker([ProviderTimeout("late")])
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)

        with pytest.raises(ProviderTimeout):
            await cached.rerank(QUERY, chunks(3), 3)
        assert len(cache) == 0
        await cached.rerank(QUERY, chunks(3), 3)  # the script is used up: the default scores

    assert len(fake.calls) == 2


@pytest.mark.parametrize(
    "garbage",
    [b"not json", b'{"scores": [0.1, 0.2]}', b'{"scores": [0.1, 0.2, 7.0]}', b'{"other": 1}'],
    ids=["not json", "wrong length", "score out of range", "wrong layout"],
)
async def test_an_unusable_entry_is_replaced_by_the_live_scores(
    tmp_path: Path, garbage: bytes
) -> None:
    fake = FakeReranker()
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)
        cache.put_many({rerank_cache_key(fake.model, QUERY, chunks(3)): garbage})

        live = await cached.rerank(QUERY, chunks(3), 3)
        replayed = await cached.rerank(QUERY, chunks(3), 3)

    assert len(fake.calls) == 1  # one live call replaced the entry; the next call was a hit
    assert replayed == live
    assert (cached.hits, cached.misses) == (1, 1)


async def test_no_chunks_touches_neither_the_cache_nor_the_inner_reranker(tmp_path: Path) -> None:
    fake = FakeReranker()
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(fake, cache)

        assert await cached.rerank(QUERY, [], 5) == []

        assert len(cache) == 0
    assert fake.calls == []
    assert (cached.hits, cached.misses) == (0, 0)


async def test_the_cache_file_holds_scores_and_not_the_question(tmp_path: Path) -> None:
    path = tmp_path / "rerank.sqlite"
    with KVCache(path) as cache:
        await CachedReranker(FakeReranker(), cache).rerank(QUERY, chunks(3), 3)

    raw = path.read_bytes()
    assert QUERY.encode() not in raw
    assert b"Content of section" not in raw


# --- The factory ---------------------------------------------------------------------------------


def is_empty(directory: Path) -> bool:
    return not any(directory.iterdir())


async def test_no_provider_means_no_reranker(tmp_path: Path) -> None:
    async with AsyncExitStack() as stack:
        assert build_reranker(make_settings(cache_dir=tmp_path), stack) is None

    assert is_empty(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "variable"),
    [
        ({"rerank_model": "rerank-v4.0-pro"}, "COHERE_API_KEY"),
        ({"rerank_model": "rerank-v4.0-pro", "cohere_api_key": "  "}, "COHERE_API_KEY"),
        ({"cohere_api_key": "test-key"}, "RERANK_MODEL"),
        ({"cohere_api_key": "test-key", "rerank_model": " "}, "RERANK_MODEL"),
    ],
)
async def test_cohere_needs_a_key_and_a_pinned_model(
    tmp_path: Path, overrides: dict[str, str], variable: str
) -> None:
    settings = make_settings(rerank_provider="cohere", cache_dir=tmp_path, **overrides)

    async with AsyncExitStack() as stack:
        with pytest.raises(ProviderConfigError, match=variable):
            build_reranker(settings, stack)


async def test_dev_and_eval_put_the_sqlite_cache_in_front_of_the_adapter(tmp_path: Path) -> None:
    for app_env, extra in (
        ("dev", {}),
        ("test", {}),
        ("eval", {"generator_providers": ["gemini"]}),
    ):
        cache_dir = tmp_path / app_env
        settings = make_settings(
            app_env=app_env,
            rerank_provider="cohere",
            rerank_model="rerank-v4.0-pro",
            cohere_api_key="test-key",
            cache_dir=cache_dir,
            **extra,
        )
        async with AsyncExitStack() as stack:
            reranker = build_reranker(settings, stack)

            assert isinstance(reranker, CachedReranker)
            assert reranker.model == "rerank-v4.0-pro"
        assert (cache_dir / "rerank.sqlite").is_file()


async def test_prod_has_no_rerank_cache_and_creates_no_file(tmp_path: Path) -> None:
    settings = make_settings(
        app_env="prod",
        database_url="postgresql://app@db.example/grounded",
        proxy_shared_secret="s" * 32,
        ip_hash_secret="i" * 32,
        daily_llm_budget=300,
        rerank_provider="cohere",
        rerank_model="rerank-v4.0-pro",
        cohere_api_key="test-key",
        cache_dir=tmp_path,
    )

    async with AsyncExitStack() as stack:
        reranker = build_reranker(settings, stack)

        assert isinstance(reranker, CohereReranker)

    assert is_empty(tmp_path)


async def test_the_settings_timeout_reaches_the_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    real_create = CohereReranker.create.__func__  # type: ignore[attr-defined]

    def spy(cls: type[CohereReranker], api_key: str, **kwargs: object) -> CohereReranker:
        seen.update(api_key=api_key, **kwargs)
        return real_create(cls, api_key, **kwargs)

    monkeypatch.setattr(CohereReranker, "create", classmethod(spy))
    settings = make_settings(
        rerank_provider="cohere",
        rerank_model="rerank-v4.0-pro",
        cohere_api_key="test-key",
        rerank_timeout_s=1.5,
        cache_dir=tmp_path,
    )

    async with AsyncExitStack() as stack:
        build_reranker(settings, stack)

    assert seen == {"api_key": "test-key", "model": "rerank-v4.0-pro", "timeout_s": 1.5}


def test_building_nothing_does_not_load_the_sdk() -> None:
    probe = (
        "import sys; from contextlib import AsyncExitStack; import grounded.runtime as r; "
        "from grounded.settings import Settings; "
        "settings = Settings(_env_file=None, rerank_provider='none'); "
        "assert r.build_reranker(settings, AsyncExitStack()) is None; "
        "print('cohere' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60
    )

    assert done.stdout.strip() == "False"
