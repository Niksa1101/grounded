"""The eval LLM cache and the 429 backoff (4.03, Tech §11 and §15.6).

No network, no database and no real sleeping: the providers are scripted fakes, the sleep is
recorded, and the clock is driven by hand."""

from __future__ import annotations

from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from grounded.generation.params import adapter_params
from grounded.generation.providers.base import GenerationResult, LLMProvider, Usage
from grounded.generation.providers.eval_wrappers import (
    LLM_EVAL_CACHE_FILE,
    BackoffExhaustedError,
    BackoffPolicy,
    BackoffProvider,
    CachingProvider,
    EvalLLM,
    EvalStats,
    eval_cache_key,
)
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.schemas.llm import LLMAnswer
from tests.support import EVAL_SETTINGS, make_settings

ANSWER = LLMAnswer(status="insufficient_context", answer_markdown="Not covered.", claims=[])
OTHER_ANSWER = LLMAnswer(status="insufficient_context", answer_markdown="Still not.", claims=[])
USAGE = Usage(input_tokens=1200, output_tokens=300, thinking_tokens=40)


class Verdict(BaseModel):
    """A judge-style schema (4.04): what the judge returns is not an ``LLMAnswer``."""

    verdict: Literal["SUPPORTED", "NOT_SUPPORTED"]
    reason: str


class OtherVerdict(BaseModel):
    verdict: Literal["SUPPORTED", "NOT_SUPPORTED"]
    reason: str


class Sleeps:
    """The injected sleep: records the waits instead of waiting."""

    def __init__(self) -> None:
        self.waited: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waited.append(seconds)


class TickClock:
    """Each reading is 5 ms after the last, so the lookup of a hit takes exactly 5 ms."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.005
        return self.now


def limited(retry_after_s: float | None, *, quota: bool = False) -> ProviderRateLimited:
    return ProviderRateLimited("429", retry_after_s=retry_after_s, is_quota=quota)


def fake(*script: Any, **kwargs: Any) -> FakeLLMProvider:
    kwargs.setdefault("name", "gemini")
    kwargs.setdefault("model", "gemini-test")
    kwargs.setdefault("usage", USAGE)
    return FakeLLMProvider(script, **kwargs)


async def call(
    provider: LLMProvider, schema: type[BaseModel] = LLMAnswer, **overrides: Any
) -> GenerationResult[Any]:
    args: dict[str, Any] = {
        "system": "sys",
        "user": "user",
        "temperature": 0.0,
        "max_output_tokens": 800,
        "timeout_s": 12.0,
    } | overrides
    return await provider.generate(schema=schema, **args)


@pytest.fixture
def kv(tmp_path: Path) -> KVCache:
    return KVCache(tmp_path / LLM_EVAL_CACHE_FILE)


def caching(
    inner: LLMProvider,
    kv: KVCache,
    stats: EvalStats | None = None,
    adapter_settings: dict[str, str] | None = None,
    clock: TickClock | None = None,
) -> CachingProvider:
    return CachingProvider(
        inner, kv, adapter_settings or {}, stats=stats or EvalStats(), clock=clock or TickClock()
    )


# --- The cache: a hit ---------------------------------------------------------------------------


async def test_an_identical_call_is_served_from_the_cache_without_calling_the_provider(
    kv: KVCache,
) -> None:
    inner = fake(ANSWER)  # a second live call would exhaust the script and fail
    stats = EvalStats()
    provider = caching(inner, kv, stats)

    first = await call(provider)
    second = await call(provider)

    assert len(inner.calls) == 1
    assert (stats.hits, stats.misses) == (1, 1)
    assert not first.cache_hit
    assert second.cache_hit
    assert second.parsed == first.parsed == ANSWER
    assert second.raw_text == first.raw_text


async def test_a_hit_gives_back_the_original_usage_and_the_lookup_time_as_latency(
    kv: KVCache,
) -> None:
    provider = caching(fake(ANSWER), kv)
    await call(provider)

    hit = await call(provider)

    # The cost of a cached run adds up as on a live one; the latency is the lookup, flagged.
    assert hit.usage == USAGE
    assert (hit.provider, hit.model) == ("gemini", "gemini-test")
    assert hit.latency_ms == 5
    assert hit.cache_hit


async def test_the_cache_survives_a_new_provider_over_the_same_file(tmp_path: Path) -> None:
    path = tmp_path / LLM_EVAL_CACHE_FILE
    with KVCache(path) as first_run:
        await call(caching(fake(ANSWER), first_run))
    with KVCache(path) as second_run:
        inner = fake()  # empty script: any live call fails the test
        result = await call(caching(inner, second_run))
    assert inner.calls == []
    assert result.parsed == ANSWER


async def test_a_judge_style_call_with_another_schema_is_cached_too(kv: KVCache) -> None:
    verdict = Verdict(verdict="SUPPORTED", reason="the source says so")
    inner = fake(verdict)
    stats = EvalStats()
    provider = caching(inner, kv, stats)

    first = await call(provider, Verdict, system="judge", user="claim + sources")
    second = await call(provider, Verdict, system="judge", user="claim + sources")

    assert len(inner.calls) == 1
    assert first.parsed == second.parsed == verdict
    assert second.cache_hit
    assert (stats.hits, stats.misses) == (1, 1)


async def test_the_same_prompt_under_another_schema_does_not_collide(kv: KVCache) -> None:
    inner = fake(
        Verdict(verdict="SUPPORTED", reason="r"), OtherVerdict(verdict="NOT_SUPPORTED", reason="r")
    )
    provider = caching(inner, kv)

    first = await call(provider, Verdict)
    second = await call(provider, OtherVerdict)

    assert len(inner.calls) == 2
    assert first.parsed.verdict == "SUPPORTED"
    assert second.parsed.verdict == "NOT_SUPPORTED"


# --- The cache: what changes the key ------------------------------------------------------------


@pytest.mark.parametrize(
    "changed",
    [
        {"system": "another system prompt"},
        {"user": "another question or context"},
        {"temperature": 0.7},
        {"max_output_tokens": 801},
    ],
    ids=["system", "user", "temperature", "max_output_tokens"],
)
async def test_a_changed_call_argument_is_a_miss(kv: KVCache, changed: dict[str, Any]) -> None:
    inner = fake(ANSWER, OTHER_ANSWER)
    provider = caching(inner, kv)

    await call(provider)
    other = await call(provider, **changed)

    assert len(inner.calls) == 2
    assert not other.cache_hit
    assert other.parsed == OTHER_ANSWER


async def test_a_changed_schema_is_a_miss(kv: KVCache) -> None:
    inner = fake(ANSWER, Verdict(verdict="SUPPORTED", reason="r"))
    provider = caching(inner, kv)

    await call(provider, LLMAnswer)
    await call(provider, Verdict)

    assert len(inner.calls) == 2


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"model": "gemini-test"}, {"model": "gemini-other"}),
        ({"name": "gemini"}, {"name": "groq"}),
    ],
    ids=["model", "provider"],
)
async def test_another_model_or_provider_is_a_miss(
    kv: KVCache, first: dict[str, str], second: dict[str, str]
) -> None:
    await call(caching(fake(ANSWER, **first), kv))
    inner = fake(OTHER_ANSWER, **second)

    result = await call(caching(inner, kv))

    assert len(inner.calls) == 1
    assert result.parsed == OTHER_ANSWER


@pytest.mark.parametrize(
    ("provider", "first", "second"),
    [
        ("gemini", {"gemini_thinking_level": "minimal"}, {"gemini_thinking_level": "high"}),
        ("groq", {"groq_reasoning_effort": "low"}, {"groq_reasoning_effort": "high"}),
    ],
)
async def test_a_changed_thinking_level_or_reasoning_effort_is_a_miss(
    kv: KVCache, provider: str, first: dict[str, str], second: dict[str, str]
) -> None:
    await call(
        caching(
            fake(ANSWER, name=provider),
            kv,
            adapter_settings=adapter_params(make_settings(**first), provider),
        )
    )
    inner = fake(OTHER_ANSWER, name=provider)

    result = await call(
        caching(inner, kv, adapter_settings=adapter_params(make_settings(**second), provider))
    )

    assert len(inner.calls) == 1
    assert result.parsed == OTHER_ANSWER


def test_a_setting_the_provider_does_not_read_does_not_change_its_key() -> None:
    # A moved Gemini setting must not throw away the Groq judge's verdicts: they cost scarce quota.
    low = make_settings(gemini_thinking_level="low")
    high = make_settings(gemini_thinking_level="high")
    assert adapter_params(low, "groq") == adapter_params(high, "groq")
    assert adapter_params(low, "gemini") != adapter_params(high, "gemini")
    assert adapter_params(low, "fake") == {}


def test_the_key_is_stable_and_a_pipe_in_a_prompt_cannot_shift_a_field() -> None:
    def key(**overrides: Any) -> str:
        args: dict[str, Any] = {
            "provider": "gemini",
            "model": "m",
            "temperature": 0.0,
            "max_output_tokens": 800,
            "system": "s",
            "user": "u",
            "schema": LLMAnswer,
            "adapter_settings": {"thinking_level": "minimal"},
        } | overrides
        return eval_cache_key(**args)

    assert key() == key()
    assert key(temperature=0) == key(temperature=0.0)
    assert key(system="a|b", user="c") != key(system="a", user="b|c")


# --- The cache: an entry that no longer validates, and what is never stored ---------------------


def key_of(provider: LLMProvider, **overrides: Any) -> str:
    args: dict[str, Any] = {
        "provider": provider.name,
        "model": provider.model,
        "temperature": 0.0,
        "max_output_tokens": 800,
        "system": "sys",
        "user": "user",
        "schema": LLMAnswer,
        "adapter_settings": {},
    } | overrides
    return eval_cache_key(**args)


@pytest.mark.parametrize(
    "entry",
    [
        b"\xff not even text",
        b'{"raw_text": "{", "usage": {"input_tokens": 1, "output_tokens": 1}}',  # invalid JSON
        b'{"raw_text": "{\\"status\\": \\"nonsense\\"}", "usage": '
        b'{"input_tokens": 1, "output_tokens": 1}}',  # no longer fits the schema
        b'{"raw_text": "", "usage": {"input_tokens": 1}, "extra": 1}',  # another layout
    ],
    ids=["corrupt", "invalid-json", "fails-the-schema", "other-layout"],
)
async def test_an_entry_that_no_longer_validates_is_replaced_by_one_live_call(
    kv: KVCache, entry: bytes
) -> None:
    inner = fake(ANSWER)  # exactly one live call is possible; a loop would exhaust the script
    stats = EvalStats()
    kv.put_many({key_of(inner): entry})
    provider = caching(inner, kv, stats)

    first = await call(provider)
    second = await call(provider)

    assert len(inner.calls) == 1
    assert not first.cache_hit  # the live reply, as for a miss
    assert second.cache_hit  # and it took the bad entry's place
    assert second.parsed == ANSWER
    assert (stats.hits, stats.misses, stats.invalid_entries) == (1, 1, 1)


async def test_nothing_is_stored_for_a_failed_call(kv: KVCache) -> None:
    errors: list[ProviderError] = [
        ProviderTimeout("late"),
        ProviderBadOutput("bad", raw="{", validation_error="x"),
        limited(5.0, quota=True),
    ]
    inner = fake(*errors, ANSWER)
    provider = caching(inner, kv)

    for _ in errors:
        with pytest.raises(ProviderError):
            await call(provider)
    assert len(kv) == 0

    result = await call(provider)  # the next identical call is live, and now stored
    assert not result.cache_hit
    assert len(kv) == 1


async def test_a_failed_call_still_counts_as_a_miss(kv: KVCache) -> None:
    stats = EvalStats()
    provider = caching(fake(ProviderTimeout("late")), kv, stats)
    with pytest.raises(ProviderTimeout):
        await call(provider)
    assert (stats.hits, stats.misses) == (0, 1)


async def test_the_entry_holds_the_raw_text_and_the_usage_only(kv: KVCache) -> None:
    inner = fake(ANSWER)
    await call(caching(inner, kv))

    stored = kv.get_many([key_of(inner)])[key_of(inner)]
    assert b"Not covered." in stored
    assert b"input_tokens" in stored
    assert b"gemini-test" not in stored  # identity is in the key, not repeated in the entry


# --- The backoff --------------------------------------------------------------------------------


def backoff(
    inner: LLMProvider,
    sleeps: Sleeps,
    *,
    stats: EvalStats | None = None,
    max_total_wait_s: float = 120.0,
    max_retries: int = 3,
) -> BackoffProvider:
    policy = BackoffPolicy(max_total_wait_s=max_total_wait_s, max_retries=max_retries)
    return BackoffProvider(inner, policy, stats=stats or EvalStats(), sleep=sleeps)


async def test_a_daily_quota_is_raised_at_once_and_is_recognisable() -> None:
    quota = limited(30.0, quota=True)
    inner, sleeps, stats = fake(quota, ANSWER), Sleeps(), EvalStats()

    with pytest.raises(ProviderRateLimited) as info:
        await call(backoff(inner, sleeps, stats=stats))

    assert info.value is quota  # the original error, untouched
    assert info.value.is_quota
    assert not isinstance(info.value, BackoffExhaustedError)
    assert sleeps.waited == []
    assert len(inner.calls) == 1  # never asked again
    assert stats.rate_limit_waits == 0


async def test_a_per_minute_429_waits_the_advertised_time_and_asks_again() -> None:
    inner, sleeps, stats = fake(limited(7.5), ANSWER), Sleeps(), EvalStats()

    result = await call(backoff(inner, sleeps, stats=stats))

    assert result.parsed == ANSWER
    assert sleeps.waited == [7.5]
    assert len(inner.calls) == 2
    assert (stats.rate_limit_waits, stats.waited_s) == (1, 7.5)


async def test_a_429_without_retry_after_waits_the_default() -> None:
    sleeps = Sleeps()
    await call(backoff(fake(limited(None), ANSWER), sleeps))
    assert sleeps.waited == [BackoffPolicy(max_total_wait_s=1.0).default_wait_s]


async def test_several_waits_add_up_and_the_call_goes_through_within_the_bound() -> None:
    sleeps = Sleeps()
    result = await call(backoff(fake(limited(40.0), limited(50.0), ANSWER), sleeps))
    assert result.parsed == ANSWER
    assert sleeps.waited == [40.0, 50.0]  # 90 s in all, under the 120 s bound


async def test_the_total_wait_is_bounded_and_then_the_error_is_raised() -> None:
    inner, sleeps = fake(limited(40.0), limited(40.0), limited(40.0), ANSWER), Sleeps()

    with pytest.raises(BackoffExhaustedError) as info:
        await call(backoff(inner, sleeps, max_total_wait_s=100.0))

    # 40 + 40 = 80 s waited; a third 40 s would make 120 s, over the 100 s bound.
    assert sleeps.waited == [40.0, 40.0]
    assert len(inner.calls) == 3
    assert info.value.waited_s == 80.0
    assert "would pass the 100s total wait" in str(info.value)


async def test_a_wait_over_the_whole_bound_is_raised_without_sleeping() -> None:
    inner, sleeps = fake(limited(500.0), ANSWER), Sleeps()

    with pytest.raises(BackoffExhaustedError) as info:
        await call(backoff(inner, sleeps))

    assert sleeps.waited == []
    assert len(inner.calls) == 1
    assert info.value.retry_after_s == 500.0


async def test_the_exhausted_error_is_a_rate_limit_error_that_is_not_a_quota() -> None:
    with pytest.raises(ProviderRateLimited) as info:
        await call(backoff(fake(limited(500.0)), Sleeps()))
    assert isinstance(info.value, BackoffExhaustedError)
    assert not info.value.is_quota  # per-minute: a later run may well succeed


async def test_zero_second_waits_cannot_loop_forever() -> None:
    inner, sleeps = fake(*[limited(0.0)] * 5, ANSWER), Sleeps()

    with pytest.raises(BackoffExhaustedError, match="after 2 waits"):
        await call(backoff(inner, sleeps, max_retries=2))

    assert sleeps.waited == [0.0, 0.0]
    assert len(inner.calls) == 3  # the call and two retries, then it gave up


async def test_the_bound_is_per_call() -> None:
    sleeps = Sleeps()
    provider = backoff(fake(limited(80.0), ANSWER, limited(80.0), OTHER_ANSWER), sleeps)

    await call(provider)
    await call(provider, user="a second call")  # starts with a fresh allowance

    assert sleeps.waited == [80.0, 80.0]


@pytest.mark.parametrize(
    "error",
    [
        ProviderUnavailable("503"),
        ProviderTimeout("late"),
        ProviderBadOutput("bad", raw="{", validation_error="x"),
        ProviderRequestRejected("bad key", status_code=401),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_every_other_error_passes_through_unchanged(error: ProviderError) -> None:
    inner, sleeps = fake(error, ANSWER), Sleeps()

    with pytest.raises(type(error)) as info:
        await call(backoff(inner, sleeps))

    assert info.value is error
    assert sleeps.waited == []
    assert len(inner.calls) == 1


# --- Both together, and the kit -----------------------------------------------------------------


async def test_a_waited_out_reply_is_cached_and_the_replay_calls_nothing(tmp_path: Path) -> None:
    sleeps = Sleeps()
    settings = make_settings(**EVAL_SETTINGS, cache_dir=tmp_path)
    async with AsyncExitStack() as stack:
        eval_llm = EvalLLM.open(settings, stack, sleep=sleeps)
        inner = fake(limited(3.0), ANSWER)
        provider = eval_llm.wrap(inner)

        first = await call(provider)
        second = await call(provider)

    assert (len(inner.calls), sleeps.waited) == (2, [3.0])
    assert (first.cache_hit, second.cache_hit) == (False, True)
    stats = eval_llm.stats
    assert (stats.hits, stats.misses, stats.rate_limit_waits, stats.waited_s) == (1, 1, 1, 3.0)
    assert (tmp_path / LLM_EVAL_CACHE_FILE).is_file()


async def test_the_generator_and_the_judge_share_one_cache_and_one_set_of_counters(
    tmp_path: Path,
) -> None:
    settings = make_settings(**EVAL_SETTINGS, cache_dir=tmp_path)
    async with AsyncExitStack() as stack:
        eval_llm = EvalLLM.open(settings, stack)
        generator = eval_llm.wrap(fake(ANSWER, name="gemini"))
        judge = eval_llm.wrap(fake(Verdict(verdict="SUPPORTED", reason="r"), name="groq"))

        await call(generator)
        await call(judge, Verdict)
        await call(generator)
        await call(judge, Verdict)

    assert (eval_llm.stats.hits, eval_llm.stats.misses) == (2, 2)


async def test_a_wrapped_provider_keeps_the_name_and_model_of_the_adapter(tmp_path: Path) -> None:
    settings = make_settings(**EVAL_SETTINGS, cache_dir=tmp_path)
    async with AsyncExitStack() as stack:
        wrapped = EvalLLM.open(settings, stack).wrap(fake(name="gemini", model="gemini-test"))
    assert (wrapped.name, wrapped.model) == ("gemini", "gemini-test")


def test_the_summary_reads_and_a_snapshot_is_a_copy() -> None:
    stats = EvalStats(hits=4, misses=6, invalid_entries=1, rate_limit_waits=2, waited_s=61.5)
    before = stats.snapshot()
    stats.hits += 1

    assert before.hits == 4
    assert stats.render() == (
        "Eval LLM cache: 5 hits, 6 misses (11 calls); rate-limit waits: 2 (61.5s); "
        "unusable entries replaced: 1"
    )
