"""Run the golden questions through the ``/ask`` pipeline, one at a time (3.14, Phase 3 closeout).

This is CLI tooling for ``grounded ask --golden``, not part of the request path and not a metric
run: it answers "does every golden question get a schema-valid ``AskResponse``, and what did it
take" and its summary is informational, never a baseline (AGENTS.md §7).

Rules that follow from the free tiers (AGENTS.md §6.15):

- **Concurrency 1.** Questions run strictly in order.
- **``Retry-After`` is honored here, in the CLI layer.** The pipeline never sleeps (Tech §9.5);
  the sleeping lives in ``_ask_with_waits``. A per-minute 429 is waited out and the same question is
  asked again, at most ``max_rate_limit_retries`` times. Anything that cannot be fixed by waiting
  stops the run instead of looping: a daily quota, a ``Retry-After`` longer than ``max_wait_s``, a
  rejected request (bad key), a missing index, a dead database, and ``max_consecutive_failures``
  provider-side failures in a row. The results so far are kept and the stop reason is reported.
- **No question text** in the output or the log: results carry the golden ``id`` only, and a failure
  reason is the exception type and its short message (never the raw model output).

A question whose answer fails validation twice (``ProviderBadOutput``) is a *result* about the
model, so it is recorded and the run goes on. Wait and retry counts are per question, so one slow
question cannot use up the budget of the next.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import psycopg
from pydantic import BaseModel, ValidationError

from grounded.evals.retrieval_runner import RESULTS_DIR
from grounded.generation.pipeline import IndexMismatchError
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
)
from grounded.infra.timing import Clock, StageTimer
from grounded.ingest.embed import EmbedderUnavailableError
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.index import NoActiveIndexError
from grounded.schemas.api import AskResponse
from grounded.schemas.eval import GoldenItem

type AskOne = Callable[[GoldenItem, RequestTrace], Awaitable[AskResponse]]
type Sleep = Callable[[float], Awaitable[None]]

MAX_REASON_CHARS = 200


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Bounds on waiting and on giving up. The CLI exposes each one as an option."""

    max_rate_limit_retries: int = 3  # per question
    default_wait_s: float = 60.0  # when a 429 carries no Retry-After (the router's default, §10)
    max_wait_s: float = 120.0  # a longer Retry-After stops the run instead of being waited out
    max_consecutive_failures: int = 3  # provider-side failures in a row


DEFAULT_POLICY = RetryPolicy()


class QuestionResult(BaseModel):
    """One golden question's outcome. ``response`` is the full ``AskResponse`` when there is one."""

    id: str
    type: str
    answerable: bool
    outcome: Literal["ok", "failed"]
    schema_valid: bool
    status: str | None = None  # answered, partial, insufficient_context
    error: str | None = None
    cache_hit: bool = False
    validation_retries: int = 0
    invalid_citation_count: int = 0
    dropped_claim_count: int = 0
    removed_url_count: int = 0
    citation_count: int = 0
    min_confidence: float | None = None
    latency_total_ms: int | None = None
    # From the response meta: every attempt of an answered question, a failed first one included.
    # A question that failed has no response, so its cost is not in the batch total.
    shadow_cost_usd: float = 0.0
    rate_limit_waits: int = 0
    response: dict[str, Any] | None = None


class BatchSummary(BaseModel):
    total: int  # questions in the file (after --limit)
    attempted: int
    not_run: int  # left over when the run stopped early
    status_counts: dict[str, int]  # answered / partial / insufficient_context / failed
    schema_valid: int
    validation_retries: int
    invalid_citations: int
    dropped_claims: int
    removed_urls: int
    cache_hits: int
    shadow_cost_usd: float
    rate_limit_waits: int
    failures: list[dict[str, str]]  # {"id", "reason"}
    aborted: str | None  # why the run stopped early, if it did


def summarize(
    results: Sequence[QuestionResult], *, total: int, aborted: str | None
) -> BatchSummary:
    statuses = Counter(r.status if r.outcome == "ok" and r.status else "failed" for r in results)
    return BatchSummary(
        total=total,
        attempted=len(results),
        not_run=total - len(results),
        status_counts=dict(sorted(statuses.items())),
        schema_valid=sum(r.schema_valid for r in results),
        validation_retries=sum(r.validation_retries for r in results),
        invalid_citations=sum(r.invalid_citation_count for r in results),
        dropped_claims=sum(r.dropped_claim_count for r in results),
        removed_urls=sum(r.removed_url_count for r in results),
        cache_hits=sum(r.cache_hit for r in results),
        shadow_cost_usd=round(sum(r.shadow_cost_usd for r in results), 6),
        rate_limit_waits=sum(r.rate_limit_waits for r in results),
        failures=[{"id": r.id, "reason": r.error or "unknown"} for r in results if r.error],
        aborted=aborted,
    )


def is_complete(summary: BatchSummary) -> bool:
    """Every question ran and got a schema-valid response: the ticket's 30/30."""
    return summary.aborted is None and summary.schema_valid == summary.total


def render_summary(summary: BatchSummary) -> str:
    counts = ", ".join(f"{name} {n}" for name, n in summary.status_counts.items()) or "none"
    lines = [
        f"Questions: {summary.attempted}/{summary.total} run"
        + (f" ({summary.not_run} not run)" if summary.not_run else ""),
        f"Status: {counts}",
        f"Schema-valid AskResponse: {summary.schema_valid}/{summary.total}",
        f"Validation retries: {summary.validation_retries}; "
        f"invalid citations removed: {summary.invalid_citations}; "
        f"claims dropped: {summary.dropped_claims}; URLs removed: {summary.removed_urls}",
        f"Cache hits: {summary.cache_hits}; rate-limit waits: {summary.rate_limit_waits}",
        f"Shadow cost: ${summary.shadow_cost_usd:.6f} (answered questions only)",
    ]
    lines += [f"Failed {f['id']}: {f['reason']}" for f in summary.failures]
    if summary.aborted:
        lines.append(f"STOPPED EARLY: {summary.aborted}")
    return "\n".join(lines)


class _StopRunError(Exception):
    """The run cannot go on; ``args[0]`` says why. Raised inside a question, caught by the loop."""


async def run_batch(
    items: Sequence[GoldenItem],
    ask_one: AskOne,
    *,
    policy: RetryPolicy = DEFAULT_POLICY,
    sleep: Sleep = asyncio.sleep,
    clock: Clock = time.perf_counter,
    on_result: Callable[[QuestionResult], None] = lambda _result: None,
) -> str | None:
    """Ask every item in order. Each result goes to ``on_result`` as it is known, so the caller
    keeps what was done if the run stops or is interrupted. Returns the stop reason, or ``None``."""
    consecutive = 0
    for item in items:
        try:
            result, provider_side_failure = await _ask_with_waits(
                item, ask_one, policy=policy, sleep=sleep, clock=clock, on_result=on_result
            )
        except _StopRunError as stop:
            return str(stop)
        consecutive = consecutive + 1 if provider_side_failure else 0
        on_result(result)
        if consecutive >= policy.max_consecutive_failures:
            return (
                f"{consecutive} provider-side failures in a row "
                f"(last: {result.error}); not asking the rest"
            )
    return None


async def _ask_with_waits(
    item: GoldenItem,
    ask_one: AskOne,
    *,
    policy: RetryPolicy,
    sleep: Sleep,
    clock: Clock,
    on_result: Callable[[QuestionResult], None],
) -> tuple[QuestionResult, bool]:
    """One question. Returns its result and whether it failed on the provider's side. Raises
    ``_StopRunError`` (after reporting the question as failed) when the run must end here."""
    waits = 0
    while True:
        trace = RequestTrace(request_id=uuid4(), timer=StageTimer(clock))
        try:
            response = await ask_one(item, trace)
        except ProviderRateLimited as exc:
            wait_s = policy.default_wait_s if exc.retry_after_s is None else exc.retry_after_s
            cannot_wait = _cannot_wait(exc, wait_s, waits, policy)
            if cannot_wait is not None:
                raise _stop(item, trace, waits, exc, on_result, cannot_wait) from exc
            waits += 1
            await sleep(wait_s)
            continue
        except ProviderBadOutput as exc:
            # The model's fault, and already retried once by the pipeline: a result, not a stop.
            return _failed(item, trace, waits, exc), False
        except (ProviderRequestRejected, NoActiveIndexError, IndexMismatchError) as exc:
            raise _stop(item, trace, waits, exc, on_result, f"{type(exc).__name__}") from exc
        except psycopg.Error as exc:
            raise _stop(item, trace, waits, exc, on_result, "database error") from exc
        except (ProviderError, EmbedderUnavailableError) as exc:
            return _failed(item, trace, waits, exc), True
        return _succeeded(item, trace, waits, response), False


def _cannot_wait(
    exc: ProviderRateLimited, wait_s: float, waits: int, policy: RetryPolicy
) -> str | None:
    """Why this 429 ends the run instead of being waited out, or ``None`` to wait and ask again."""
    if exc.is_quota:
        return "daily quota exhausted"
    if wait_s > policy.max_wait_s:
        return f"Retry-After {wait_s:g}s is over the {policy.max_wait_s:g}s limit"
    if waits >= policy.max_rate_limit_retries:
        return f"still rate limited after {waits} waits on one question"
    return None


def _stop(
    item: GoldenItem,
    trace: RequestTrace,
    waits: int,
    exc: Exception,
    on_result: Callable[[QuestionResult], None],
    why: str,
) -> _StopRunError:
    on_result(_failed(item, trace, waits, exc))
    return _StopRunError(f"{why} at {item.id}: {_reason(exc)}")


def _reason(exc: Exception) -> str:
    text = " ".join(str(exc).split())[:MAX_REASON_CHARS]
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _failed(item: GoldenItem, trace: RequestTrace, waits: int, exc: Exception) -> QuestionResult:
    return QuestionResult(
        id=item.id,
        type=item.type,
        answerable=item.answerable,
        outcome="failed",
        schema_valid=False,
        error=_reason(exc),
        validation_retries=trace.validation_retries,
        invalid_citation_count=trace.invalid_citation_count or 0,
        dropped_claim_count=trace.dropped_claim_count or 0,
        removed_url_count=trace.removed_url_count or 0,
        rate_limit_waits=waits,
    )


def _succeeded(
    item: GoldenItem, trace: RequestTrace, waits: int, response: AskResponse
) -> QuestionResult:
    dumped = response.model_dump(mode="json")
    # The pipeline builds the response from validated parts; this proves it still round-trips
    # through the public schema, which is what a client of /v1/ask relies on.
    try:
        AskResponse.model_validate(dumped)
        schema_valid, error = True, None
    except ValidationError as exc:
        schema_valid, error = False, f"AskResponse failed validation: {exc.error_count()} errors"
    return QuestionResult(
        id=item.id,
        type=item.type,
        answerable=item.answerable,
        outcome="ok" if schema_valid else "failed",
        schema_valid=schema_valid,
        status=response.status,
        error=error,
        cache_hit=response.meta.cache_hit,
        validation_retries=trace.validation_retries,
        invalid_citation_count=trace.invalid_citation_count or 0,
        dropped_claim_count=trace.dropped_claim_count or 0,
        removed_url_count=trace.removed_url_count or 0,
        citation_count=len(response.citations),
        min_confidence=response.min_confidence,
        latency_total_ms=response.meta.latency_ms.get("total"),
        shadow_cost_usd=response.meta.shadow_cost_usd,
        rate_limit_waits=waits,
        response=dumped,
    )


def ask_results_path(now: datetime) -> Path:
    """``eval/results/<UTC timestamp>-ask.json`` (gitignored, like every results file)."""
    return RESULTS_DIR / f"{now.strftime('%Y%m%dT%H%M%SZ')}-ask.json"


def write_ask_batch(
    path: Path,
    summary: BatchSummary,
    results: Sequence[QuestionResult],
    *,
    started_at: datetime,
    golden_set_version: str,
    golden_set_sha256: str,
    mode: str,
    fake: bool,
) -> None:
    """The run as JSON: a header that says what ran, the summary and every question's result.

    ``fake`` marks a run with the stub provider, so a results file can never pass for real model
    output. No question text is written: ``id`` links each result back to the golden set.
    """
    document = {
        "kind": "ask_batch",
        "started_at": started_at.isoformat(),
        "golden_set_version": golden_set_version,
        "golden_set_sha256": golden_set_sha256,
        "mode": mode,
        "fake_provider": fake,
        "summary": summary.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
