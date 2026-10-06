"""``grounded ask --golden`` runner (evals/ask_batch.py): waits, stops and the summary. No database,
no network: ``ask_one`` is a scripted fake and the sleep is recorded instead of slept."""

from __future__ import annotations

import json
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from grounded.evals.ask_batch import (
    DEFAULT_POLICY,
    QuestionResult,
    RetryPolicy,
    is_complete,
    render_summary,
    run_batch,
    summarize,
    write_ask_batch,
)
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.observability.request_log import RequestTrace
from grounded.schemas.api import AskResponse, Meta
from grounded.schemas.eval import GoldenItem, RelevantSection

SECRET_QUESTION = "What is the secret phrase of the golden question?"


def item(n: int) -> GoldenItem:
    return GoldenItem(
        id=f"q{n:03d}",
        question=SECRET_QUESTION,
        type="factual",
        answerable=True,
        reference_answer="x",
        relevant_sections=[RelevantSection(section="docs/en/docs/a.md", grade=2)],
    )


def response(*, cost: float = 0.001, cache_hit: bool = False) -> AskResponse:
    return AskResponse(
        status="answered",
        answer_markdown="An answer.",
        claims=[],
        citations=[],
        follow_up_questions=[],
        min_confidence=None,
        meta=Meta(
            request_id=uuid4(),
            provider="fake",
            model="stub",
            fallback_used=False,
            cache_hit=cache_hit,
            rerank_used=False,
            prompt_version="answer_v1@00000000",
            index_version="v@00000000",
            retrieval_config_hash="0" * 64,
            latency_ms={"total": 5},
            tokens={"input": 1, "output": 1},
            shadow_cost_usd=cost,
        ),
    )


class Script:
    """``ask_one`` that consumes one scripted step per call: a response, or an error to raise."""

    def __init__(self, *steps: AskResponse | ProviderError) -> None:
        self.steps = deque(steps)
        self.asked: list[str] = []

    async def __call__(self, golden: GoldenItem, trace: RequestTrace) -> AskResponse:
        self.asked.append(golden.id)
        step = self.steps.popleft()
        if isinstance(step, ProviderError):
            raise step
        return step


class Sleeps:
    def __init__(self) -> None:
        self.waited: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waited.append(seconds)


async def run(
    items: list[GoldenItem], script: Script, *, policy: RetryPolicy = DEFAULT_POLICY
) -> tuple[list[QuestionResult], str | None, Sleeps]:
    results: list[QuestionResult] = []
    sleeps = Sleeps()
    stopped = await run_batch(items, script, policy=policy, sleep=sleeps, on_result=results.append)
    return results, stopped, sleeps


async def test_every_question_is_asked_in_order_and_summarized() -> None:
    script = Script(response(cost=0.002), response(cost=0.003, cache_hit=True))
    results, stopped, sleeps = await run([item(1), item(2)], script)
    assert stopped is None
    assert script.asked == ["q001", "q002"]
    assert sleeps.waited == []
    summary = summarize(results, total=2, aborted=stopped)
    assert (summary.attempted, summary.schema_valid, summary.cache_hits) == (2, 2, 1)
    assert summary.status_counts == {"answered": 2}
    assert summary.shadow_cost_usd == pytest.approx(0.005)
    assert is_complete(summary)


async def test_a_per_minute_429_waits_the_advertised_time_and_asks_again() -> None:
    limited = ProviderRateLimited("slow down", retry_after_s=7.0, is_quota=False)
    script = Script(limited, response())
    results, stopped, sleeps = await run([item(1)], script)
    assert stopped is None
    assert sleeps.waited == [7.0]
    assert script.asked == ["q001", "q001"]  # the same question, not the next one
    assert results[0].outcome == "ok"
    assert results[0].rate_limit_waits == 1


async def test_a_429_without_retry_after_waits_the_default() -> None:
    limited = ProviderRateLimited("slow down", retry_after_s=None, is_quota=False)
    _, _, sleeps = await run([item(1)], Script(limited, response()))
    assert sleeps.waited == [RetryPolicy().default_wait_s]


async def test_a_daily_quota_stops_the_run_without_waiting() -> None:
    quota = ProviderRateLimited("daily quota", retry_after_s=30.0, is_quota=True)
    script = Script(response(), quota)
    results, stopped, sleeps = await run([item(1), item(2), item(3)], script)
    assert stopped is not None
    assert "daily quota exhausted at q002" in stopped
    assert sleeps.waited == []
    assert script.asked == ["q001", "q002"]  # q003 is never asked
    summary = summarize(results, total=3, aborted=stopped)
    assert (summary.attempted, summary.not_run, summary.schema_valid) == (2, 1, 1)
    assert not is_complete(summary)
    assert "STOPPED EARLY" in render_summary(summary)


async def test_a_retry_after_over_the_limit_stops_instead_of_waiting() -> None:
    limited = ProviderRateLimited("later", retry_after_s=500.0, is_quota=False)
    results, stopped, sleeps = await run([item(1)], Script(limited))
    assert stopped is not None
    assert "Retry-After 500s" in stopped
    assert sleeps.waited == []
    assert results[0].outcome == "failed"


async def test_waiting_is_bounded_per_question() -> None:
    limited = [ProviderRateLimited("slow", retry_after_s=1.0, is_quota=False)] * 4
    policy = RetryPolicy(max_rate_limit_retries=3)
    script = Script(*limited)
    _, stopped, sleeps = await run([item(1)], script, policy=policy)
    assert stopped is not None
    assert "still rate limited after 3 waits" in stopped
    assert sleeps.waited == [1.0, 1.0, 1.0]
    assert len(script.asked) == 4  # the first call plus three retries, then it gave up


async def test_the_wait_budget_is_per_question() -> None:
    def limited() -> ProviderRateLimited:
        return ProviderRateLimited("slow", retry_after_s=1.0, is_quota=False)

    policy = RetryPolicy(max_rate_limit_retries=1)
    script = Script(limited(), response(), limited(), response())
    results, stopped, _ = await run([item(1), item(2)], script, policy=policy)
    assert stopped is None
    assert [r.rate_limit_waits for r in results] == [1, 1]


async def test_bad_output_after_the_pipeline_retry_is_a_result_not_a_stop() -> None:
    bad = ProviderBadOutput("the answer failed the citation checks")
    script = Script(bad, response())
    results, stopped, _ = await run([item(1), item(2)], script)
    assert stopped is None
    assert [r.outcome for r in results] == ["failed", "ok"]
    assert results[0].error is not None
    assert "ProviderBadOutput" in results[0].error
    assert not results[0].schema_valid
    summary = summarize(results, total=2, aborted=stopped)
    assert summary.status_counts == {"answered": 1, "failed": 1}
    assert [f["id"] for f in summary.failures] == ["q001"]
    assert not is_complete(summary)


async def test_provider_failures_in_a_row_stop_the_run_but_a_success_resets_the_count() -> None:
    down = [ProviderUnavailable("503"), ProviderTimeout("slow")]
    script = Script(down[0], down[1], response(), down[0], down[1], down[0])
    results, stopped, _ = await run([item(n) for n in range(1, 8)], script)
    assert stopped is not None
    assert "3 provider-side failures in a row" in stopped
    assert len(results) == 6  # the success at q003 reset the count, so q007 was never asked
    assert script.asked[-1] == "q006"


async def test_a_rejected_request_stops_the_run() -> None:
    rejected = ProviderRequestRejected("API key not valid", status_code=400)
    script = Script(rejected)
    results, stopped, _ = await run([item(1), item(2)], script)
    assert stopped is not None
    assert "ProviderRequestRejected at q001" in stopped
    assert len(results) == 1


async def test_a_failed_question_keeps_what_the_trace_learned() -> None:
    async def ask_one(golden: GoldenItem, trace: RequestTrace) -> AskResponse:
        trace.validation_retries = 1
        trace.invalid_citation_count = 2
        raise ProviderBadOutput("still bad")

    results: list[QuestionResult] = []
    await run_batch([item(1)], ask_one, on_result=results.append)
    assert (results[0].validation_retries, results[0].invalid_citation_count) == (1, 2)


async def test_results_and_summary_never_contain_question_text(tmp_path: Path) -> None:
    script = Script(response(), ProviderBadOutput(f"bad: {'x' * 500}"))
    results, stopped, _ = await run([item(1), item(2)], script)
    summary = summarize(results, total=2, aborted=stopped)
    path = tmp_path / "out" / "ask.json"
    write_ask_batch(
        path,
        summary,
        results,
        started_at=datetime(2026, 10, 6, tzinfo=UTC),
        golden_set_version="v1",
        golden_set_sha256="a" * 64,
        mode="hybrid",
        fake=True,
    )
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["fake_provider"] is True
    assert [r["id"] for r in written["results"]] == ["q001", "q002"]
    assert SECRET_QUESTION not in path.read_text(encoding="utf-8")
    assert SECRET_QUESTION not in render_summary(summary)
    assert len(written["results"][1]["error"]) < 250  # a long message is cut
