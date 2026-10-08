"""The behavior every LLM adapter must share, run against both (Tech.md §9.1, §16).

One scenario list, one parametrized fixture: ``GeminiProvider`` and ``GroqProvider`` get the same
success, usage, bad-output, error-mapping, timeout and retry scenarios from their recorded or
hand-made fixtures, so the two cannot drift apart unnoticed. What each adapter alone does (the
schema conversion, thinking levels, finish reasons, Groq's ``json_validate_failed``) is tested in
``test_gemini_provider.py`` and ``test_groq_provider.py``.

Which fixtures are recorded and which hand-made: ``tests/fixtures/{gemini,groq}`` (the READMEs and
the ``_synthetic`` markers). The expected token counts below are read off the recordings.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import anyio
import pytest

from grounded.generation.pipeline import AskMode, AskPipeline
from grounded.generation.prompts import load_answer_prompt, load_no_rag_prompt
from grounded.generation.providers.base import GenerationResult
from grounded.infra.db import create_pool
from grounded.infra.provider_errors import (
    TRUNCATED_FEEDBACK,
    ProviderBadOutput,
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
from grounded.schemas.llm import LLMAnswer
from tests.provider_rigs import GeminiRig, GroqRig, Reply, Rig, SeenRequest, Step
from tests.support import FakeClock, make_settings


@dataclass(frozen=True)
class Adapter:
    name: str
    rig: Callable[..., Rig]
    # (input, output, thinking) tokens of generate_success.json, as the adapter must report them.
    success_usage: tuple[int, int, int]
    # What the recorded cut-off reply billed: Gemini reports it, Groq's 400 carries no usage.
    cut_off_tokens: tuple[int, int]
    # The retry-after of generate_429_per_minute.json.
    per_minute_retry_after_s: float


ADAPTERS = [
    Adapter(
        "gemini",
        GeminiRig,
        success_usage=(91, 143, 0),
        cut_off_tokens=(91, 5),
        per_minute_retry_after_s=53.0,
    ),
    Adapter(
        "groq",
        GroqRig,
        success_usage=(376, 323, 27),
        cut_off_tokens=(0, 0),
        per_minute_retry_after_s=6.0,
    ),
]


@pytest.fixture(params=ADAPTERS, ids=lambda adapter: adapter.name)
def adapter(request: pytest.FixtureRequest) -> Adapter:
    return request.param


@pytest.fixture(autouse=True)
def nothing_may_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """No sleeping in the request path (Tech §9.5): the Groq SDK would sleep before a retry."""

    async def sleep(seconds: float) -> None:
        raise AssertionError(f"an adapter slept for {seconds}s")

    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(anyio, "sleep", sleep)


async def generate(rig: Rig, *, timeout_s: float = 12.0) -> GenerationResult[LLMAnswer]:
    return await rig.provider.generate(
        system="SYSTEM",
        user="USER",
        schema=LLMAnswer,
        temperature=0.0,
        max_output_tokens=800,
        timeout_s=timeout_s,
    )


# --- Success ------------------------------------------------------------------------------------


async def test_a_valid_reply_is_parsed_into_the_schema(adapter: Adapter) -> None:
    result = await generate(adapter.rig("success"))
    assert isinstance(result.parsed, LLMAnswer)
    assert result.parsed.status == "answered"
    assert result.parsed.claims[0].citation_ids == ["c1"]
    assert LLMAnswer.model_validate_json(result.raw_text) == result.parsed
    assert result.provider == adapter.name
    assert result.latency_ms == 250  # the rig's clock moves 250 ms per call


async def test_the_prompts_and_limits_reach_the_request(adapter: Adapter) -> None:
    rig = adapter.rig("success")
    await generate(rig)
    assert rig.requests == [SeenRequest("SYSTEM", "USER", temperature=0.0, max_output_tokens=800)]


async def test_usage_is_mapped_the_way_the_provider_bills_it(adapter: Adapter) -> None:
    usage = (await generate(adapter.rig("success"))).usage
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == adapter.success_usage


async def test_a_reply_without_usage_counts_zero_tokens(adapter: Adapter) -> None:
    usage = (await generate(adapter.rig("success_without_usage"))).usage
    assert (usage.input_tokens, usage.output_tokens, usage.thinking_tokens) == (0, 0, 0)


async def test_aclose_closes_the_sdk_client(adapter: Adapter) -> None:
    rig = adapter.rig()
    await rig.aclose()
    assert rig.closed


# --- Bad output ---------------------------------------------------------------------------------


async def test_a_reply_that_breaks_a_constraint_is_bad_output_with_compact_feedback(
    adapter: Adapter,
) -> None:
    # ``c12`` breaks the label pattern, which neither provider is held to while generating, so only
    # the Pydantic validation after the call catches it (AGENTS.md §6.2).
    bad = (
        '{"status": "answered", "answer_markdown": "Use add_task [c12].", "claims": '
        '[{"text": "Use add_task.", "citation_ids": ["c12"], "self_confidence": 1.0}], '
        '"follow_up_questions": []}'
    )
    with pytest.raises(ProviderBadOutput) as caught:
        await generate(adapter.rig(Reply(bad)))
    error = caught.value
    # The retry feedback names the field and the rule, with no echo of the rejected value.
    assert error.validation_error == (
        "claims.0.citation_ids.0: String should match pattern '^c[1-9]$'"
    )
    assert error.raw == bad
    assert error.retryable
    # The bad reply was billed: its usage travels on the error.
    assert (error.input_tokens, error.output_tokens) == adapter.success_usage[:2]


async def test_text_that_is_not_json_is_bad_output(adapter: Adapter) -> None:
    with pytest.raises(ProviderBadOutput) as caught:
        await generate(adapter.rig(Reply("Sorry, no JSON.")))
    assert caught.value.raw == "Sorry, no JSON."
    assert caught.value.validation_error.startswith("output: Invalid JSON")


async def test_a_recorded_cut_off_reply_is_bad_output_that_says_why(adapter: Adapter) -> None:
    # Real recordings: an output limit far too small, so there is no complete JSON.
    with pytest.raises(ProviderBadOutput) as caught:
        await generate(adapter.rig("invalid_output"))
    # The retry is told why, not only that the JSON broke (Phase 3 review #5).
    assert caught.value.validation_error == TRUNCATED_FEEDBACK
    assert caught.value.retryable  # a shorter answer can fit
    assert (caught.value.input_tokens, caught.value.output_tokens) == adapter.cut_off_tokens


# --- Error mapping ------------------------------------------------------------------------------


async def test_a_per_minute_429_is_rate_limited_with_the_server_delay(adapter: Adapter) -> None:
    with pytest.raises(ProviderRateLimited) as caught:
        await generate(adapter.rig("429_per_minute"))
    assert (caught.value.retry_after_s, caught.value.is_quota) == (
        adapter.per_minute_retry_after_s,
        False,
    )


async def test_a_daily_quota_429_is_flagged_as_quota(adapter: Adapter) -> None:
    with pytest.raises(ProviderRateLimited) as caught:
        await generate(adapter.rig("429_daily_quota"))
    assert caught.value.is_quota is True


async def test_a_500_is_unavailable(adapter: Adapter) -> None:
    with pytest.raises(ProviderUnavailable, match="500"):
        await generate(adapter.rig("500"))


async def test_a_bad_request_is_rejected_without_a_retry_hint(adapter: Adapter) -> None:
    with pytest.raises(ProviderRequestRejected) as caught:
        await generate(adapter.rig("400"))
    assert caught.value.status_code == 400


async def test_a_call_that_hangs_times_out(adapter: Adapter) -> None:
    with pytest.raises(ProviderTimeout):
        await generate(adapter.rig("hang"), timeout_s=0.01)


async def test_a_timeout_raised_by_the_http_layer_is_a_provider_timeout(adapter: Adapter) -> None:
    with pytest.raises(ProviderTimeout):
        await generate(adapter.rig("timeout"))


async def test_a_transport_failure_is_unavailable(adapter: Adapter) -> None:
    with pytest.raises(ProviderUnavailable):
        await generate(adapter.rig("connect_error"))


@pytest.mark.parametrize("failure", ["429_per_minute", "429_daily_quota", "500", "connect_error"])
async def test_a_failed_call_is_not_retried_or_waited_out_by_the_adapter(
    adapter: Adapter, failure: str
) -> None:
    # The SDK's own retries would sleep before a second request (``nothing_may_sleep`` fails any
    # sleep); retry and backoff belong to the router (Tech §10), which has to see the failure.
    rig = adapter.rig(failure, "success")
    with pytest.raises((ProviderRateLimited, ProviderUnavailable)):
        await generate(rig)
    assert len(rig.requests) == 1
    assert rig.remaining == 1  # the scripted success was never asked for


# --- The retry with feedback, through the real pipeline ------------------------------------------
# The loop lives in ``AskPipeline`` until the router (Phase 7); what each adapter owes it is a
# ``ProviderBadOutput`` that is retryable and carries the feedback. ``no_rag`` needs no database.

UNCITED_ANSWER = (
    '{"status": "answered", "answer_markdown": "Quokkas sleep.", "claims": '
    '[{"text": "Quokkas sleep.", "citation_ids": [], "self_confidence": 0.9}], '
    '"follow_up_questions": []}'
)


def no_rag_pipeline(rig: Rig) -> AskPipeline:
    settings = make_settings()
    pool = create_pool(settings)  # built, never opened: no_rag does not touch the database
    return AskPipeline(
        settings=settings,
        pool=pool,
        index_cache=ActiveIndexCache(pool, ttl_s=settings.active_index_ttl_s),
        embedder=FakeEmbedder(),
        provider=rig.provider,
        prompt=load_answer_prompt(),
        no_rag_prompt=load_no_rag_prompt(),
        pricing=load_pricing(),
        clock=FakeClock(),
    )


def trace_for(pipeline: AskPipeline) -> RequestTrace:
    return RequestTrace(request_id=uuid4(), timer=StageTimer(FakeClock()))


async def test_invalid_output_is_retried_once_and_the_second_request_carries_the_feedback(
    adapter: Adapter,
) -> None:
    steps: tuple[Step, ...] = ("invalid_output", Reply(UNCITED_ANSWER))
    rig = adapter.rig(*steps)
    pipeline = no_rag_pipeline(rig)
    trace = trace_for(pipeline)

    response = await pipeline.ask(
        "Where do quokkas sleep?", trace.request_id, mode=AskMode.NO_RAG, trace=trace
    )

    assert response.status == "answered"
    assert trace.validation_retries == 1
    first, second = rig.requests
    assert second.user == load_no_rag_prompt().render_retry(first.user, error=TRUNCATED_FEEDBACK)
    assert second.system == first.system


async def test_a_second_invalid_output_is_final_and_makes_no_third_request(
    adapter: Adapter,
) -> None:
    rig = adapter.rig("invalid_output", "invalid_output", Reply(UNCITED_ANSWER))
    pipeline = no_rag_pipeline(rig)
    trace = trace_for(pipeline)

    with pytest.raises(ProviderBadOutput):
        await pipeline.ask(
            "Where do quokkas sleep?", trace.request_id, mode=AskMode.NO_RAG, trace=trace
        )

    assert len(rig.requests) == 2  # exactly one retry (AGENTS.md §6.4)
    assert rig.remaining == 1
