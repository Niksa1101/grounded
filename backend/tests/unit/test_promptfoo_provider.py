"""The promptfoo Python provider (evals/promptfoo_provider.py) without promptfoo and without a
database: the result dicts it builds, the tagged errors, and the long-lived runner. The pipeline is
a scripted fake ``Asker``; the real pipeline is exercised in
tests/integration/test_promptfoo_provider.py."""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import psycopg
import pytest

from grounded.evals import promptfoo_provider as provider
from grounded.evals.promptfoo_provider import (
    CaseResult,
    PipelineRunner,
    answer,
    call_api,
    failure_payload,
    mode_of,
    skipped_payload,
    success_payload,
)
from grounded.generation.pipeline import AskMode, IndexMismatchError
from grounded.generation.providers.base import Usage
from grounded.generation.providers.eval_wrappers import BackoffExhaustedError
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.infra.timing import StageTimer
from grounded.ingest.embed import EmbedderUnavailableError
from grounded.observability.cost import load_pricing
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.index import NoActiveIndexError
from grounded.retrieval.types import RetrievedChunk
from grounded.runtime import ProviderConfigError
from grounded.schemas.api import AskResponse, Citation, Claim, Meta
from tests.support import FakeClock, make_settings

PRICING = load_pricing()
GEMINI = "gemini-3.5-flash-lite"

CHUNK = RetrievedChunk(
    chunk_id=11,
    section_id="docs/en/docs/a.md#top",
    anchor_path=("top",),
    breadcrumb_text="A > Top",
    url="https://fastapi.tiangolo.com/a/#top",
    content="Chunk text, with “curly quotes” and café.",
    token_count=9,
    content_hash="h",
)
OTHER = RetrievedChunk(
    chunk_id=12,
    section_id="docs/en/docs/b.md#",
    anchor_path=(),
    breadcrumb_text="B",
    url="https://fastapi.tiangolo.com/b/",
    content="Another chunk.",
    token_count=3,
    content_hash="h2",
)


def trace_after(*, hybrid: bool, retries: int = 0, invalid: int | None = 0) -> RequestTrace:
    """A trace as the pipeline leaves it after a request."""
    clock = FakeClock()
    trace = RequestTrace(request_id=uuid4(), timer=StageTimer(clock))
    trace.provider, trace.model = "gemini", GEMINI
    trace.add_usage(Usage(input_tokens=1_000, output_tokens=200, thinking_tokens=50))
    trace.validation_retries = retries
    trace.invalid_citation_count = invalid
    trace.dropped_claim_count = 0
    trace.removed_url_count = 1
    trace.llm_cache_hits = 1
    if hybrid:
        trace.retrieved_section_ids = [CHUNK.section_id, OTHER.section_id]
        trace.context_chunks = {"c1": CHUNK, "c2": OTHER}
    return trace


def response_for(
    trace: RequestTrace, *, status: str = "answered", cited: bool = True
) -> AskResponse:
    claims = [
        Claim(
            text="A claim.",
            citations=[1] if cited else [],
            confidence=0.8,
            confidence_components={"retrieval": 0.9},
        )
    ]
    citations = [
        Citation(
            n=1,
            chunk_id=CHUNK.chunk_id,
            url=CHUNK.url,  # pyright: ignore[reportArgumentType]
            title="A",
            breadcrumb=CHUNK.breadcrumb_text,
            snippet="Chunk text",
        )
    ]
    return AskResponse(
        status=status,  # pyright: ignore[reportArgumentType]
        answer_markdown="A claim. [1]" if cited else "A claim.",
        claims=claims,
        citations=citations if cited else [],
        follow_up_questions=[],
        min_confidence=0.8,
        meta=Meta(
            request_id=trace.request_id,
            provider="gemini",
            model=GEMINI,
            fallback_used=False,
            cache_hit=False,
            rerank_used=False,
            prompt_version="answer_v1@abc",
            index_version="0.141.1@4949e8a3",
            retrieval_config_hash="f" * 64,
            latency_ms={"total": 900, "embed": 40, "retrieval": 60, "llm": 800},
            tokens={"input": trace.input_tokens or 0, "output": trace.output_tokens or 0},
            shadow_cost_usd=0.0008,
        ),
    )


# --- the success dict --------------------------------------------------------------------------


def test_success_payload_is_the_ask_response_with_usage_cost_and_latency() -> None:
    trace = trace_after(hybrid=True)
    response = response_for(trace)

    payload = success_payload(response, trace, AskMode.HYBRID, cold_start=False)

    assert payload["output"] == response.model_dump(mode="json")
    assert payload["tokenUsage"] == {
        "total": 1_200,
        "prompt": 1_000,
        "completion": 200,
        "numRequests": 1,
    }
    assert payload["cost"] == 0.0008
    assert payload["latencyMs"] == 900
    assert "error" not in payload
    json.dumps(payload)  # promptfoo reads it as JSON


def test_a_retry_counts_as_a_request() -> None:
    trace = trace_after(hybrid=True, retries=1)

    payload = success_payload(response_for(trace), trace, AskMode.HYBRID, cold_start=False)

    assert payload["tokenUsage"]["numRequests"] == 2


def test_success_metadata_carries_what_the_assertions_and_the_gate_read() -> None:
    trace = trace_after(hybrid=True, retries=1, invalid=2)
    metadata = success_payload(response_for(trace), trace, AskMode.HYBRID, cold_start=True)[
        "metadata"
    ]

    assert metadata["mode"] == "hybrid"
    assert metadata["status"] == "answered"
    assert metadata["cold_start"] is True
    assert metadata["validation_retries"] == 1
    assert metadata["invalid_citation_count"] == 2  # the count before the labels were removed
    assert metadata["dropped_claim_count"] == 0
    assert metadata["removed_url_count"] == 1
    assert metadata["llm_cache_hits"] == 1
    assert (metadata["provider"], metadata["model"]) == ("gemini", GEMINI)
    assert metadata["prompt_version"] == "answer_v1@abc"
    assert metadata["index_version"] == "0.141.1@4949e8a3"
    assert metadata["retrieval_config_hash"] == "f" * 64
    assert metadata["latency_ms"] == {"total": 900, "embed": 40, "retrieval": 60, "llm": 800}
    assert metadata["shadow_cost_usd"] == 0.0008
    assert metadata["tokens"] == {"input": 1_000, "output": 200}


def test_context_has_the_chunk_text_and_ids_the_judge_and_precision_need() -> None:
    trace = trace_after(hybrid=True)

    metadata = success_payload(response_for(trace), trace, AskMode.HYBRID, cold_start=False)[
        "metadata"
    ]

    assert metadata["context"] == [
        {
            "label": "c1",
            "chunk_id": 11,
            "section_id": "docs/en/docs/a.md#top",
            "anchor_path": ["top"],
            "url": "https://fastapi.tiangolo.com/a/#top",
            "breadcrumb": "A > Top",
            "content": CHUNK.content,
        },
        {
            "label": "c2",
            "chunk_id": 12,
            "section_id": "docs/en/docs/b.md#",
            "anchor_path": [],
            "url": "https://fastapi.tiangolo.com/b/",
            "breadcrumb": "B",
            "content": "Another chunk.",
        },
    ]
    assert metadata["retrieved_section_ids"] == ["docs/en/docs/a.md#top", "docs/en/docs/b.md#"]


def test_each_claim_lists_the_chunks_it_cites() -> None:
    trace = trace_after(hybrid=True)

    metadata = success_payload(response_for(trace), trace, AskMode.HYBRID, cold_start=False)[
        "metadata"
    ]

    [claim] = metadata["claims"]
    assert claim["text"] == "A claim."
    assert claim["citations"] == [1]
    assert claim["chunk_ids"] == [CHUNK.chunk_id]
    assert claim["confidence"] == 0.8  # the server's, never the model's self-confidence
    assert claim["confidence_components"] == {"retrieval": 0.9}


def test_no_rag_has_no_context_and_no_retrieval() -> None:
    trace = trace_after(hybrid=False)

    metadata = success_payload(
        response_for(trace, cited=False), trace, AskMode.NO_RAG, cold_start=False
    )["metadata"]

    assert metadata["mode"] == "no_rag"
    assert metadata["context"] == []
    assert metadata["retrieved_section_ids"] == []
    assert metadata["claims"][0]["chunk_ids"] == []


def test_non_ascii_chunk_text_survives_json() -> None:
    trace = trace_after(hybrid=True)
    payload = success_payload(response_for(trace), trace, AskMode.HYBRID, cold_start=False)

    assert (
        json.loads(json.dumps(payload, ensure_ascii=False))["metadata"]["context"][0]["content"]
        == CHUNK.content
    )


# --- the tagged error --------------------------------------------------------------------------


def failure(exc: Exception, trace: RequestTrace | None = None) -> dict[str, Any]:
    return failure_payload(
        exc,
        trace or trace_after(hybrid=True, invalid=None),
        AskMode.HYBRID,
        pricing=PRICING,
        cold_start=False,
    )


def test_a_daily_quota_is_tagged_with_its_class_and_quota_true() -> None:
    payload = failure(
        ProviderRateLimited("daily quota exhausted", retry_after_s=None, is_quota=True)
    )

    assert payload["error"] == "[ProviderRateLimited quota=true] daily quota exhausted"
    assert payload["metadata"]["error_kind"] == "ProviderRateLimited"
    assert payload["metadata"]["is_quota"] is True
    assert payload["metadata"]["error_bases"] == ["ProviderError"]
    assert payload["metadata"]["retry_after_s"] is None


def test_a_per_minute_limit_is_quota_false_with_its_retry_after() -> None:
    payload = failure(ProviderRateLimited("slow down", retry_after_s=30.0, is_quota=False))

    assert payload["error"].startswith("[ProviderRateLimited quota=false] ")
    assert payload["metadata"]["is_quota"] is False
    assert payload["metadata"]["retry_after_s"] == 30.0


def test_the_backoff_subclass_keeps_its_own_name_and_names_its_bases() -> None:
    payload = failure(BackoffExhaustedError("gave up", retry_after_s=60.0, waited_s=120.0))

    assert payload["error"].startswith("[BackoffExhaustedError quota=false] ")
    assert payload["metadata"]["error_kind"] == "BackoffExhaustedError"
    assert payload["metadata"]["error_bases"] == ["ProviderRateLimited", "ProviderError"]
    assert payload["metadata"]["waited_s"] == 120.0


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ProviderTimeout("took too long"), "ProviderTimeout"),
        (ProviderUnavailable("503"), "ProviderUnavailable"),
        (ProviderRequestRejected("bad key", status_code=403), "ProviderRequestRejected"),
        (EmbedderUnavailableError("no key"), "EmbedderUnavailableError"),
        (NoActiveIndexError("none active"), "NoActiveIndexError"),
        (IndexMismatchError("other model"), "IndexMismatchError"),
        (psycopg.OperationalError("connection refused"), "OperationalError"),
    ],
)
def test_every_expected_failure_is_tagged_with_its_class_name(exc: Exception, kind: str) -> None:
    payload = failure(exc)

    assert payload["error"].startswith(f"[{kind} quota=false] ")
    assert payload["metadata"]["error_kind"] == kind
    assert payload["metadata"]["is_quota"] is False


def test_bad_output_keeps_the_compact_validation_error() -> None:
    exc = ProviderBadOutput(
        "output does not match LLMAnswer", validation_error="claims: Field required"
    )

    payload = failure(exc)

    assert payload["metadata"]["validation_error"] == "claims: Field required"
    assert payload["error"] == "[ProviderBadOutput quota=false] output does not match LLMAnswer"


def test_every_tagged_error_tells_promptfoo_not_to_retry_it() -> None:
    # promptfoo re-calls a provider whose error text has "429" or "rate limit", unless the result
    # says it is a quota. A message like this one would otherwise cost another API call.
    exc = BackoffExhaustedError("429: rate limit exceeded", retry_after_s=60.0, waited_s=120.0)

    metadata = failure(exc)["metadata"]

    assert metadata["rateLimitKind"] == "quota"
    assert "rate limit" in failure(exc)["error"]


def test_the_error_is_one_short_line_without_the_exception_noise() -> None:
    noisy = ProviderUnavailable("line one\n\n   line   two " + "x" * 1_000)

    error = failure(noisy)["error"]

    assert "\n" not in error
    assert "line one line two" in error
    assert len(error) < 300


def test_a_failure_keeps_what_the_request_had_spent() -> None:
    trace = trace_after(hybrid=True, retries=1, invalid=None)

    payload = failure(ProviderBadOutput("bad twice"), trace)

    assert payload["tokenUsage"] == {"total": 1_200, "prompt": 1_000, "completion": 200}
    assert payload["cost"] > 0
    assert payload["metadata"]["validation_retries"] == 1
    assert payload["metadata"]["invalid_citation_count"] is None
    assert payload["metadata"]["mode"] == "hybrid"
    assert "output" not in payload
    json.dumps(payload)


def test_a_failure_before_any_generation_has_no_token_usage() -> None:
    trace = RequestTrace(request_id=uuid4(), timer=StageTimer(FakeClock()))
    trace.provider, trace.model = "gemini", GEMINI

    payload = failure(ProviderTimeout("slow"), trace)

    assert "tokenUsage" not in payload
    assert payload["cost"] == 0.0


def test_skipped_payload_repeats_the_tag_and_marks_the_call_as_not_made() -> None:
    first = failure(ProviderRateLimited("daily quota exhausted", retry_after_s=None, is_quota=True))

    skipped = skipped_payload(first)

    assert skipped["error"].startswith("[ProviderRateLimited quota=true] skipped: ")
    assert skipped["metadata"]["skipped"] is True
    assert skipped["metadata"]["error_kind"] == "ProviderRateLimited"
    assert skipped["metadata"]["is_quota"] is True
    assert skipped["metadata"]["rateLimitKind"] == "quota"
    assert skipped["cost"] == 0.0
    assert "tokenUsage" not in skipped


# --- answer() ----------------------------------------------------------------------------------


class ScriptedAsker:
    """An ``Asker`` that replays a script; a step is a response builder or an exception."""

    def __init__(self, *steps: Callable[[RequestTrace], AskResponse] | Exception) -> None:
        self.steps = list(steps)
        self.questions: list[str] = []
        self.modes: list[AskMode] = []

    async def ask(
        self,
        question: str,
        request_id: UUID,
        *,
        mode: AskMode = AskMode.HYBRID,
        trace: RequestTrace | None = None,
    ) -> AskResponse:
        assert trace is not None
        self.questions.append(question)
        self.modes.append(mode)
        trace.provider, trace.model = "gemini", GEMINI
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        trace.add_usage(Usage(input_tokens=10, output_tokens=5))
        trace.invalid_citation_count = 0
        return step(trace)


def ok(trace: RequestTrace) -> AskResponse:
    return response_for(trace)


async def test_answer_returns_the_success_payload() -> None:
    asker = ScriptedAsker(ok)

    result = await answer(asker, "How?", AskMode.NO_RAG, pricing=PRICING, cold_start=True)

    assert result.stop_run is False
    assert result.payload["output"]["status"] == "answered"
    assert result.payload["metadata"]["cold_start"] is True
    assert (asker.questions, asker.modes) == (["How?"], [AskMode.NO_RAG])


async def test_answer_turns_an_expected_failure_into_a_tagged_error() -> None:
    asker = ScriptedAsker(ProviderTimeout("slow"))

    result = await answer(asker, "How?", AskMode.HYBRID, pricing=PRICING)

    assert result.payload["error"].startswith("[ProviderTimeout quota=false] ")
    assert result.stop_run is False


async def test_answer_does_not_hide_a_bug() -> None:
    asker = ScriptedAsker(ValueError("a bug"))

    with pytest.raises(ValueError, match="a bug"):
        await answer(asker, "How?", AskMode.HYBRID, pricing=PRICING)


@pytest.mark.parametrize(
    ("exc", "stops"),
    [
        (ProviderRateLimited("daily", retry_after_s=None, is_quota=True), True),
        (ProviderRequestRejected("bad key", status_code=401), True),
        (ProviderRateLimited("minute", retry_after_s=20.0, is_quota=False), False),
        (BackoffExhaustedError("gave up", retry_after_s=60.0, waited_s=120.0), False),
        (ProviderTimeout("slow"), False),
        (ProviderUnavailable("503"), False),
        (ProviderBadOutput("bad twice"), False),
        (EmbedderUnavailableError("quota"), False),
    ],
)
async def test_only_a_daily_quota_or_a_rejected_request_stops_the_run(
    exc: ProviderError, stops: bool
) -> None:
    result = await answer(ScriptedAsker(exc), "How?", AskMode.HYBRID, pricing=PRICING)

    assert result.stop_run is stops


# --- the runner --------------------------------------------------------------------------------


class Harness:
    """Counts how the runner opens and closes the pipeline it is given."""

    def __init__(self, asker: ScriptedAsker) -> None:
        self.asker = asker
        self.opened = 0
        self.closed = 0
        self.loops: set[int] = set()
        self.threads: set[str] = set()

    @asynccontextmanager
    async def open(self) -> AsyncGenerator[ScriptedAsker]:
        import asyncio

        self.opened += 1
        self.loops.add(id(asyncio.get_running_loop()))
        self.threads.add(threading.current_thread().name)
        try:
            yield self.asker
        finally:
            self.closed += 1

    def runner(self) -> PipelineRunner:
        return PipelineRunner(self.open, pricing=PRICING)


def test_the_runner_opens_the_pipeline_once_lazily_and_keeps_one_loop() -> None:
    harness = Harness(ScriptedAsker(ok, ok, ok))
    runner = harness.runner()
    try:
        assert harness.opened == 0  # nothing happens before the first call
        payloads = [runner.call(f"Question {n}?", AskMode.HYBRID) for n in range(3)]
    finally:
        runner.close()

    assert all("output" in p for p in payloads)
    assert harness.opened == 1
    assert len(harness.loops) == 1
    assert harness.threads == {"promptfoo-pipeline-loop"}  # not the caller's thread
    assert harness.asker.questions == ["Question 0?", "Question 1?", "Question 2?"]


def test_only_the_first_call_is_a_cold_start() -> None:
    harness = Harness(ScriptedAsker(ok, ok))
    runner = harness.runner()
    try:
        first = runner.call("One?", AskMode.HYBRID)
        second = runner.call("Two?", AskMode.HYBRID)
    finally:
        runner.close()

    assert (first["metadata"]["cold_start"], second["metadata"]["cold_start"]) == (True, False)


def test_close_closes_what_the_first_call_opened_and_is_safe_twice() -> None:
    harness = Harness(ScriptedAsker(ok))
    runner = harness.runner()
    runner.call("One?", AskMode.HYBRID)

    runner.close()
    runner.close()

    assert harness.closed == 1
    assert not any(t.name == "promptfoo-pipeline-loop" for t in threading.enumerate())


def test_close_without_a_call_does_nothing() -> None:
    harness = Harness(ScriptedAsker())

    harness.runner().close()

    assert (harness.opened, harness.closed) == (0, 0)


def test_after_a_daily_quota_no_further_call_reaches_the_pipeline() -> None:
    quota = ProviderRateLimited("daily quota exhausted", retry_after_s=None, is_quota=True)
    harness = Harness(ScriptedAsker(ok, quota, ok, ok))
    runner = harness.runner()
    try:
        results = [runner.call(f"Q{n}?", AskMode.HYBRID) for n in range(4)]
    finally:
        runner.close()

    assert "output" in results[0]
    assert results[1]["error"] == "[ProviderRateLimited quota=true] daily quota exhausted"
    for later in results[2:]:
        assert later["error"].startswith("[ProviderRateLimited quota=true] skipped: ")
        assert later["metadata"]["skipped"] is True
    assert len(harness.asker.questions) == 2  # the successful one and the one that hit the quota
    assert len(harness.asker.steps) == 2  # the last two steps were never asked


def test_a_per_minute_limit_does_not_stop_the_run() -> None:
    limited = BackoffExhaustedError("gave up", retry_after_s=60.0, waited_s=120.0)
    harness = Harness(ScriptedAsker(limited, ok))
    runner = harness.runner()
    try:
        first = runner.call("One?", AskMode.HYBRID)
        second = runner.call("Two?", AskMode.HYBRID)
    finally:
        runner.close()

    assert first["error"].startswith("[BackoffExhaustedError quota=false] ")
    assert "output" in second


def test_a_failed_open_is_raised_and_the_next_call_tries_again() -> None:
    attempts: list[int] = []

    @asynccontextmanager
    async def flaky() -> AsyncGenerator[ScriptedAsker]:
        attempts.append(1)
        if len(attempts) == 1:
            raise ProviderConfigError("GEMINI_API_KEY must be set")
        yield ScriptedAsker(ok)

    runner = PipelineRunner(flaky, pricing=PRICING)
    try:
        with pytest.raises(ProviderConfigError, match="GEMINI_API_KEY"):
            runner.call("One?", AskMode.HYBRID)
        assert "output" in runner.call("One?", AskMode.HYBRID)
    finally:
        runner.close()

    assert len(attempts) == 2


# --- entry point -------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "no_rag"])
def test_mode_of_reads_the_provider_config(mode: str) -> None:
    assert mode_of({"id": "file://provider.py", "config": {"mode": mode}}) == AskMode(mode)


@pytest.mark.parametrize(
    "options", [{}, {"config": {}}, {"config": {"mode": "hybrid_rerank"}}, {"config": 3}]
)
def test_mode_of_rejects_a_missing_or_unknown_mode(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=r"config\.mode must be one of"):
        mode_of(options)


def test_call_api_asks_the_question_in_the_configured_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    harness = Harness(ScriptedAsker(ok))
    runner = harness.runner()
    monkeypatch.setattr(provider, "default_runner", lambda: runner)
    try:
        payload = call_api("How do I?", {"config": {"mode": "no_rag"}}, {"vars": {}})
    finally:
        runner.close()

    assert payload["metadata"]["mode"] == "no_rag"
    assert harness.asker.questions == ["How do I?"]
    assert harness.asker.modes == [AskMode.NO_RAG]


def test_the_default_runner_refuses_to_run_outside_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider, "get_settings", lambda: make_settings())  # app_env="test"
    provider.default_runner.cache_clear()
    try:
        with pytest.raises(ProviderConfigError, match="eval mode"):
            provider.default_runner()
    finally:
        provider.default_runner.cache_clear()


def test_case_result_is_a_plain_pair() -> None:
    result = CaseResult({"a": 1})

    assert (result.payload, result.stop_run) == ({"a": 1}, False)
