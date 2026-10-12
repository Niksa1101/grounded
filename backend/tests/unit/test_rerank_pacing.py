"""The eval pacer (``PacedReranker``, 6.04): spacing of live calls, metering, and where it sits.

No test sleeps for real: the clock and the sleep are a ``FakeNow`` the test moves by hand
(AGENTS.md §8).
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AsyncExitStack
from pathlib import Path

import pytest

from grounded import runtime
from grounded.generation.providers.fake import StubLLMProvider
from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import ProviderRateLimited
from grounded.ingest.embed import FakeEmbedder
from grounded.retrieval.rerank import (
    CachedReranker,
    CohereReranker,
    FakeReranker,
    FakeStep,
    PacedReranker,
    RerankUsage,
    pacing_allowance_s,
    rerank_usage,
)
from grounded.retrieval.types import RetrievedChunk
from grounded.runtime import ProviderConfigError, build_reranker
from grounded.settings import Settings
from tests.rerank_rig import QUERY, chunks
from tests.support import make_settings


class FakeNow:
    """A clock and a sleep in one: ``sleep`` records the wait and moves the clock by it."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class SlowFake(FakeReranker):
    """A fake whose call takes ``took`` seconds of the shared clock."""

    def __init__(self, now: FakeNow, took: float, script: Sequence[FakeStep] = ()) -> None:
        super().__init__(script)
        self._now, self._took = now, took

    async def _scores(self, query: str, chunks_: Sequence[RetrievedChunk]) -> list[float]:
        self._now.now += self._took
        return await super()._scores(query, chunks_)


def paced(now: FakeNow, inner: FakeReranker, interval: float = 6.0) -> PacedReranker:
    return PacedReranker(inner, min_interval_s=interval, clock=now, sleep=now.sleep)


async def test_live_calls_start_at_least_the_interval_apart() -> None:
    now = FakeNow()
    pacer = paced(now, SlowFake(now, took=1.0))

    await pacer.rerank(QUERY, chunks(4), 4)  # first call: nothing to wait for
    await pacer.rerank(QUERY + "?", chunks(4), 4)  # 1 s after the first started: waits 5 s
    now.now += 10.0  # a long pause: the interval has passed on its own
    await pacer.rerank(QUERY + "??", chunks(4), 4)

    assert now.sleeps == [pytest.approx(5.0)]
    assert pacer.live_calls == 3


async def test_the_interval_runs_from_the_start_of_the_previous_call() -> None:
    now = FakeNow()
    pacer = paced(now, SlowFake(now, took=4.0))

    await pacer.rerank(QUERY, chunks(3), 3)  # starts at t0, ends at t0 + 4
    await pacer.rerank(QUERY + "?", chunks(3), 3)  # must start at t0 + 6: waits 2 s, not 6

    assert now.sleeps == [pytest.approx(2.0)]


async def test_an_interval_of_zero_never_waits_and_still_meters() -> None:
    now = FakeNow()
    pacer = paced(now, SlowFake(now, took=0.25), interval=0)

    for n in range(3):
        await pacer.rerank(QUERY + str(n), chunks(2), 2)

    assert now.sleeps == []
    assert pacer.live_calls == 3
    assert pacer.latencies_ms == [pytest.approx(250.0)] * 3


async def test_a_negative_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="min_interval_s"):
        PacedReranker(FakeReranker(), min_interval_s=-1)


async def test_a_failed_call_counts_for_the_spacing_but_not_for_the_latency() -> None:
    now = FakeNow()
    limited = ProviderRateLimited("slow down", retry_after_s=30.0, is_quota=False)
    pacer = paced(now, SlowFake(now, took=1.0, script=[limited]))

    with pytest.raises(ProviderRateLimited):
        await pacer.rerank(QUERY, chunks(3), 3)
    await pacer.rerank(QUERY + "?", chunks(3), 3)

    # The 429 was a request Cohere saw: the next one waits its interval (and not the Retry-After).
    assert now.sleeps == [pytest.approx(5.0)]
    assert pacer.live_calls == 2
    assert pacer.latencies_ms == [pytest.approx(1000.0)]  # only the call that returned


async def test_no_chunks_is_no_call_and_no_wait() -> None:
    now = FakeNow()
    pacer = paced(now, SlowFake(now, took=1.0))

    assert await pacer.rerank(QUERY, [], 5) == []

    assert (pacer.live_calls, now.sleeps) == (0, [])


# --- Under the cache: only misses wait and count -------------------------------------------------


async def test_a_cache_hit_is_neither_delayed_nor_counted(tmp_path: Path) -> None:
    now = FakeNow()
    fake = SlowFake(now, took=1.0)
    pacer = paced(now, fake)
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        cached = CachedReranker(pacer, cache)

        first = await cached.rerank(QUERY, chunks(5), 5)
        again = await cached.rerank(QUERY, chunks(5), 5)  # a hit, right after the first call
        other = await cached.rerank(QUERY + "?", chunks(5), 5)  # a miss: the interval applies

    assert again == first
    assert len(fake.calls) == 2  # the hit asked nobody
    assert now.sleeps == [pytest.approx(5.0)]  # the hit did not wait; only the second miss did
    assert pacer.live_calls == 2
    assert other != first
    assert rerank_usage(cached) == RerankUsage(
        live_calls=2, cache_hits=1, latencies_ms=(1000.0, 1000.0)
    )


def test_usage_is_read_off_the_layers_and_a_bare_reranker_claims_nothing(tmp_path: Path) -> None:
    assert rerank_usage(FakeReranker()) is None
    now = FakeNow()
    pacer = paced(now, FakeReranker())
    assert rerank_usage(pacer) == RerankUsage(live_calls=0, cache_hits=0, latencies_ms=())
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        # A cache over a reranker that is not a pacer still knows its misses (= live calls).
        assert rerank_usage(CachedReranker(FakeReranker(), cache)) == RerankUsage(0, 0, ())


def test_the_allowance_is_the_interval_through_the_cache_and_zero_elsewhere(
    tmp_path: Path,
) -> None:
    pacer = paced(FakeNow(), FakeReranker(), interval=6.0)
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        assert pacing_allowance_s(pacer) == 6.0
        assert pacing_allowance_s(CachedReranker(pacer, cache)) == 6.0
        assert pacing_allowance_s(CachedReranker(FakeReranker(), cache)) == 0.0
    assert pacing_allowance_s(FakeReranker()) == 0.0


# --- build_reranker ------------------------------------------------------------------------------


async def test_a_paced_reranker_is_a_cache_over_a_pacer_over_the_adapter(tmp_path: Path) -> None:
    settings = make_settings(
        rerank_provider="cohere",
        rerank_model="rerank-v4.0-pro",
        cohere_api_key="test-key",
        cache_dir=tmp_path,
        eval_rerank_min_interval_s=7.5,
    )
    async with AsyncExitStack() as stack:
        reranker = build_reranker(settings, stack, paced=True)

        assert isinstance(reranker, CachedReranker)
        assert isinstance(reranker.inner, PacedReranker)
        assert isinstance(reranker.inner._inner, CohereReranker)
        assert pacing_allowance_s(reranker) == 7.5


async def test_unpaced_is_the_cache_over_the_adapter_as_before(tmp_path: Path) -> None:
    settings = make_settings(
        rerank_provider="cohere",
        rerank_model="rerank-v4.0-pro",
        cohere_api_key="test-key",
        cache_dir=tmp_path,
    )
    async with AsyncExitStack() as stack:
        reranker = build_reranker(settings, stack)

        assert isinstance(reranker, CachedReranker)
        assert isinstance(reranker.inner, CohereReranker)


async def test_pacing_is_refused_in_prod_where_nothing_may_sleep(tmp_path: Path) -> None:
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
        with pytest.raises(ProviderConfigError, match="APP_ENV=prod"):
            build_reranker(settings, stack, paced=True)


async def test_no_provider_is_no_reranker_even_when_pacing_is_asked_for(tmp_path: Path) -> None:
    async with AsyncExitStack() as stack:
        assert build_reranker(make_settings(cache_dir=tmp_path), stack, paced=True) is None


def test_the_pacing_interval_is_a_setting_with_the_trial_limits_default() -> None:
    assert make_settings().eval_rerank_min_interval_s == 6.0
    assert make_settings(eval_rerank_min_interval_s=0).eval_rerank_min_interval_s == 0
    with pytest.raises(ValueError, match="eval_rerank_min_interval_s"):
        make_settings(eval_rerank_min_interval_s=-1)


@pytest.mark.parametrize(("app_env", "paced"), [("eval", True), ("dev", False), ("test", False)])
async def test_only_the_eval_runtime_builds_a_paced_reranker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, app_env: str, paced: bool
) -> None:
    seen: list[bool] = []

    def spy(settings: Settings, stack: AsyncExitStack, *, paced: bool = False) -> FakeReranker:
        seen.append(paced)
        return FakeReranker()

    monkeypatch.setattr(runtime, "build_reranker", spy)
    settings = make_settings(
        app_env=app_env,
        rerank_provider="cohere",
        rerank_model="rerank-v4.0-pro",
        cohere_api_key="test-key",
        cache_dir=tmp_path,
        **({"generator_providers": ["gemini"]} if app_env == "eval" else {}),
    )

    async with runtime.open_runtime(
        settings, embedder=FakeEmbedder(dim=settings.embedding_dim), provider=StubLLMProvider()
    ):
        pass

    assert seen == [paced]  # the request path (dev, test, prod) never sleeps
