"""The router inside ``AskPipeline`` where no database is needed (7.04, Tech §10, §15.6).

``no_rag`` mode touches neither the index nor the answer cache, so the pipeline can be driven with a
pool that is built and never opened. What is proved here is the part the eval harness depends on:
in eval mode (and for the CLI, ``primary_only``) the router is a pass-through. The error that comes
out is the very object the provider raised, so ``ask_batch``, the promptfoo provider and the gate
read the same ``is_quota``, ``retry_after_s``, ``waited_s`` and class names as before the router,
and no breaker ever turns a later question into a skip. Outside eval mode the same failure is the
router's ``AllProvidersUnavailable``, with the provider's error as its cause.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from grounded.evals.ask_batch import RetryPolicy, run_batch
from grounded.evals.gate import _category  # pyright: ignore[reportPrivateUsage]
from grounded.evals.promptfoo_provider import answer
from grounded.generation.pipeline import AskMode, AskPipeline
from grounded.generation.prompts import load_answer_prompt, load_no_rag_prompt
from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.eval_wrappers import (
    BackoffExhaustedError,
    EvalLLM,
)
from grounded.generation.providers.fake import FakeLLMProvider, ScriptStep
from grounded.generation.router import AllProvidersUnavailable
from grounded.infra.db import create_pool
from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.infra.timing import StageTimer
from grounded.ingest.embed import FakeEmbedder
from grounded.observability.cost import load_pricing
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.index import ActiveIndexCache
from grounded.schemas.api import AskResponse
from grounded.schemas.eval import GoldenItem, RelevantSection
from grounded.schemas.generation_eval import ErrorInfo
from grounded.schemas.llm import LLMAnswer, LLMClaim
from grounded.settings import Settings
from tests.support import EVAL_SETTINGS, FakeClock, make_settings

GEMINI_MODEL = "gemini-3.5-flash-lite"  # priced in pricing.toml
QUESTION = "Where does the quokka sleep?"


def answer_text() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=[], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep.", claims=[claim])


def gemini(*script: ScriptStep) -> FakeLLMProvider:
    return FakeLLMProvider(script, name="gemini", model=GEMINI_MODEL)


def pipeline(
    provider: LLMProvider,
    settings: Settings,
    *,
    primary_only: bool = False,
    fallbacks: Sequence[FakeLLMProvider] = (),
) -> AskPipeline:
    pool = create_pool(settings)  # built, never opened: no_rag does not touch the database
    return AskPipeline(
        settings=settings,
        pool=pool,
        index_cache=ActiveIndexCache(pool, ttl_s=settings.active_index_ttl_s),
        embedder=FakeEmbedder(),
        provider=provider,
        prompt=load_answer_prompt(),
        no_rag_prompt=load_no_rag_prompt(),
        pricing=load_pricing(),
        clock=FakeClock(),
        fallback_providers=fallbacks,
        primary_only=primary_only,
    )


def eval_settings(tmp_path: Path) -> Settings:
    return make_settings(cache_dir=tmp_path, **EVAL_SETTINGS)


def trace() -> RequestTrace:
    return RequestTrace(request_id=uuid4(), timer=StageTimer(FakeClock()))


async def ask(target: AskPipeline) -> AskResponse:
    return await target.ask(QUESTION, uuid4(), mode=AskMode.NO_RAG, trace=trace())


PROVIDER_ERRORS: list[Callable[[], ProviderError]] = [
    lambda: ProviderRateLimited("daily", retry_after_s=None, is_quota=True),
    lambda: ProviderRateLimited("daily", retry_after_s=4.0, is_quota=True),
    lambda: ProviderRateLimited("minute", retry_after_s=12.0, is_quota=False),
    lambda: ProviderRateLimited("minute", retry_after_s=None, is_quota=False),
    lambda: BackoffExhaustedError("gave up", retry_after_s=70.0, waited_s=30.0),
    lambda: ProviderUnavailable("503"),
    lambda: ProviderTimeout("late"),
    lambda: ProviderRequestRejected("bad key", status_code=401),
    lambda: ProviderBadOutput("bad", raw="{", validation_error="oops"),
]


# --- eval mode and the CLI: the provider's own error, no breaker ----------------------------------


@pytest.mark.parametrize("make_error", PROVIDER_ERRORS)
async def test_eval_mode_raises_the_providers_own_error_object(
    make_error: Callable[[], ProviderError], tmp_path: Path
) -> None:
    error = make_error()
    # A bad output is retried once by the router, so it needs a second failing step.
    steps: list[ScriptStep] = [error, error] if isinstance(error, ProviderBadOutput) else [error]
    provider = gemini(*steps)

    with pytest.raises(type(error)) as caught:
        await ask(pipeline(provider, eval_settings(tmp_path)))

    assert caught.value is error  # the same object: type, is_quota, retry_after_s, waited_s, all
    assert type(caught.value) is type(error)
    assert not isinstance(caught.value, AllProvidersUnavailable)


async def test_eval_mode_has_no_breaker_so_the_next_question_is_asked(tmp_path: Path) -> None:
    # With a breaker one 429 would skip every later golden question: the eval's own backoff would
    # never get to wait, and its quota stop would never see the quota.
    limited = ProviderRateLimited("slow", retry_after_s=30.0, is_quota=False)
    provider = gemini(limited, answer_text())
    target = pipeline(provider, eval_settings(tmp_path))

    with pytest.raises(ProviderRateLimited):
        await ask(target)
    response = await ask(target)

    assert response.status == "answered"
    assert len(provider.calls) == 2  # asked again, not skipped


async def test_a_primary_only_pipeline_outside_eval_has_no_breaker_either() -> None:
    limited = ProviderRateLimited("slow", retry_after_s=30.0, is_quota=False)
    provider = gemini(limited, answer_text())
    target = pipeline(provider, make_settings(), primary_only=True)

    with pytest.raises(ProviderRateLimited):
        await ask(target)
    assert (await ask(target)).status == "answered"
    assert len(provider.calls) == 2


async def test_a_primary_only_pipeline_takes_no_fallback(tmp_path: Path) -> None:
    other = FakeLLMProvider([], name="groq", model="openai/gpt-oss-120b")
    with pytest.raises(ValueError, match="takes no fallback"):
        pipeline(gemini(), make_settings(), primary_only=True, fallbacks=[other])
    # Eval mode is primary only whether or not the caller says so.
    with pytest.raises(ValueError, match="takes no fallback"):
        pipeline(gemini(), eval_settings(tmp_path), fallbacks=[other])


async def test_outside_eval_the_failure_is_the_routers_with_the_provider_error_as_cause() -> None:
    error = ProviderRateLimited("slow", retry_after_s=30.0, is_quota=False)
    provider = gemini(error)

    with pytest.raises(AllProvidersUnavailable) as caught:
        await ask(pipeline(provider, make_settings()))

    assert caught.value.causes == (error,)
    assert caught.value.__cause__ is error
    assert caught.value.retry_after_s == pytest.approx(30.0, abs=1.0)  # the breaker's open time


async def test_the_unwrapped_error_still_prints_a_traceback(tmp_path: Path) -> None:
    # ``raise causes[-1] from router_error`` makes a cycle in the cause chain (the router's own
    # error was raised from the very same object); printing it must not loop or fail.
    provider = gemini(ProviderUnavailable("503"))
    with pytest.raises(ProviderUnavailable) as caught:
        await ask(pipeline(provider, eval_settings(tmp_path)))
    assert "ProviderUnavailable" in "".join(traceback.format_exception(caught.value))


# --- the real eval path against a failing provider ------------------------------------------------
# The generator the harness uses is ``CachingProvider(BackoffProvider(adapter))`` (Tech §15.6). Each
# case below runs that real wrapper, the pipeline in eval mode, and then the code that reads the
# error: the promptfoo provider's payload, the golden batch's waits and stops, and the gate's
# classification. They replay what the harness' own tests assert for each error, through the router.


def golden(n: int = 1) -> GoldenItem:
    return GoldenItem(
        id=f"q{n:03d}",
        question=QUESTION,
        type="factual",
        answerable=True,
        reference_answer="x",
        relevant_sections=[RelevantSection(section="docs/en/docs/a.md", grade=2)],
    )


@dataclass
class EvalRig:
    """A scripted adapter behind the real eval wrappers (a fake sleep that records its waits) and
    the pipeline over it."""

    inner: FakeLLMProvider
    pipeline: AskPipeline
    sleeps: list[float]

    async def batch(self, count: int = 1) -> str | None:
        """The golden batch over this pipeline: the stop reason, or ``None``."""

        async def ask_one(item: GoldenItem, trace: RequestTrace) -> AskResponse:
            return await self.pipeline.ask(
                item.question, trace.request_id, mode=AskMode.NO_RAG, trace=trace
            )

        async def no_sleep(seconds: float) -> None:
            self.sleeps.append(seconds)

        items = [golden(n) for n in range(1, count + 1)]
        return await run_batch(items, ask_one, policy=RetryPolicy(), sleep=no_sleep)


@pytest.fixture
def eval_rig(tmp_path: Path) -> Iterator[Callable[..., EvalRig]]:
    caches: list[KVCache] = []

    def make(*script: ScriptStep) -> EvalRig:
        settings = eval_settings(tmp_path)
        sleeps: list[float] = []

        async def sleep(seconds: float) -> None:
            sleeps.append(seconds)

        cache = KVCache(tmp_path / f"llm_eval_{len(caches)}.sqlite")
        caches.append(cache)
        inner = gemini(*script)
        kit = EvalLLM(settings, cache, sleep=sleep)
        return EvalRig(inner, pipeline(kit.wrap(inner), settings), sleeps)

    yield make
    for cache in caches:
        cache.close()


async def test_a_backoff_that_gave_up_reaches_the_harness_unchanged(
    eval_rig: Callable[..., EvalRig],
) -> None:
    # A Retry-After over the bound: the wrapper raises BackoffExhaustedError without sleeping.
    slow = ProviderRateLimited("slow", retry_after_s=500.0, is_quota=False)
    rig = eval_rig(slow)

    result = await answer(rig.pipeline, QUESTION, AskMode.NO_RAG, pricing=load_pricing())

    metadata = result.payload["metadata"]
    assert metadata["error_kind"] == "BackoffExhaustedError"
    assert "ProviderRateLimited" in metadata["error_bases"]
    assert (metadata["is_quota"], metadata["retry_after_s"], metadata["waited_s"]) == (
        False,
        500.0,
        0.0,
    )
    assert result.payload["error"].startswith("[BackoffExhaustedError quota=false]")
    assert result.stop_run is False
    error = ErrorInfo(
        stage="generator", kind=metadata["error_kind"], bases=tuple(metadata["error_bases"])
    )
    assert _category(error) == "provider"  # counts toward ``inconclusive``, as before

    # The golden batch does not wait a second time for it: it stops the run, like a quota.
    batch = eval_rig(slow)
    stopped = await batch.batch()
    assert stopped is not None
    assert "rate limited after the eval backoff gave up" in stopped
    assert batch.sleeps == []


async def test_a_daily_quota_reaches_the_harness_unchanged(
    eval_rig: Callable[..., EvalRig],
) -> None:
    quota = ProviderRateLimited("daily", retry_after_s=None, is_quota=True)
    rig = eval_rig(quota)

    result = await answer(rig.pipeline, QUESTION, AskMode.NO_RAG, pricing=load_pricing())

    metadata = result.payload["metadata"]
    assert metadata["error_kind"] == "ProviderRateLimited"
    assert (metadata["is_quota"], metadata["retry_after_s"]) == (True, None)
    assert result.stop_run is True  # asking again in this run cannot work
    assert len(rig.inner.calls) == 1
    assert rig.sleeps == []

    batch = eval_rig(quota)
    stopped = await batch.batch(count=2)
    assert stopped is not None
    assert stopped.startswith("daily quota exhausted at q001")
    assert len(batch.inner.calls) == 1  # q002 was never asked


async def test_a_5xx_after_the_backoff_retries_is_the_original_error(
    eval_rig: Callable[..., EvalRig],
) -> None:
    # The wrapper asks again after 5, 10, 20 and 40 s (recorded, not slept), then raises the
    # original ProviderUnavailable, which the router would only wrap and the pipeline unwraps.
    rig = eval_rig(*[ProviderUnavailable("503") for _ in range(5)])

    result = await answer(rig.pipeline, QUESTION, AskMode.NO_RAG, pricing=load_pricing())

    assert result.payload["metadata"]["error_kind"] == "ProviderUnavailable"
    assert result.stop_run is False
    assert rig.sleeps == [5.0, 10.0, 20.0, 40.0]
    assert len(rig.inner.calls) == 5  # one router attempt: the retries are the wrapper's
    assert _category(ErrorInfo(stage="generator", kind="ProviderUnavailable")) == "provider"


async def test_bad_output_in_eval_mode_is_the_last_one_after_one_retry(
    eval_rig: Callable[..., EvalRig],
) -> None:
    rig = eval_rig("{", "{", answer_text())

    result = await answer(rig.pipeline, QUESTION, AskMode.NO_RAG, pricing=load_pricing())

    metadata = result.payload["metadata"]
    assert metadata["error_kind"] == "ProviderBadOutput"
    assert metadata["validation_retries"] == 1
    assert len(rig.inner.calls) == 2  # the call and its retry, not four
    assert _category(ErrorInfo(stage="generator", kind="ProviderBadOutput")) == (
        "generator_bad_output"
    )
