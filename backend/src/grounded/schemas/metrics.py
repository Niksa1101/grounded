"""HTTP model of ``GET /v1/metrics/summary`` (Tech.md §13, §14; ticket 8.02, PRD FR-27).

The dashboard's whole contract, aggregates only. No model here has a field for question text, a
question hash, an IP hash or a request id, and ``extra="forbid"`` keeps it that way: a query that
selected one more column would fail validation instead of reaching the page (AGENTS.md §6.13).

**Denominators.** A rate is a ratio of request counts in the window, and it is ``null`` when its
denominator is 0 (no request of that kind yet), never 0: "no data" and "never happens" differ.

- *served*: outcome ``answered``, ``partial`` or ``insufficient_context`` (an answer was returned,
  from the cache or not). The denominator of ``cache_hit`` and of the per-1k cost.
- *generated*: not a cache hit and the LLM stage ran (``latency_llm_ms`` is not NULL). The
  denominator of ``fallback``, ``validation_retry`` and ``validation_failure``: those can only
  happen to a request that reached generation, so a burst of 429s or cache hits cannot dilute them.
- *timed*: not a cache hit, and the request entered the pipeline (outcome not ``rate_limited``,
  ``budget_exhausted`` or ``bad_request``). Latency is over these rows; a ``deadline_exceeded``
  request stays in, because a timeout is exactly what a p95 must show.

**Numbers.** Percentiles are ``percentile_cont`` (DB.md §8) rounded to whole milliseconds, half away
from zero. Money is a JSON number in USD, like ``meta.shadow_cost_usd`` of ``AskResponse``: the
sums are ``Decimal`` in the service and rounded to the 8 places of ``request_logs.shadow_cost_usd``
before they become a float, so the page and a hand calculation agree to the digit.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

# The windows the dashboard offers. A contract, not config: the page has one button per value.
Window = Literal["1d", "7d", "30d"]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RequestCounts(_Model):
    """User requests in the window (``source`` web or api; CI's eval rows are never counted).

    ``answered + partial + insufficient_context + throttled + errors == total``: ``throttled`` is
    ``rate_limited`` and ``budget_exhausted``, ``errors`` every other outcome (``bad_request``,
    ``validation_failed``, ``provider_unavailable``, ``internal_error``, ``deadline_exceeded``).
    ``cache_hits`` is a subset of the three served outcomes.
    """

    total: int
    answered: int
    partial: int
    insufficient_context: int
    throttled: int
    errors: int
    cache_hits: int


class Rates(_Model):
    """Ratios in [0, 1]; ``null`` on a zero denominator.

    - ``cache_hit``: cache hits / served requests.
    - ``fallback``: generated requests that called a provider other than the primary / generated.
    - ``validation_retry``: generated requests with ``validation_retries > 0`` / generated.
    - ``validation_failure``: generated requests with outcome ``validation_failed`` / generated.
    """

    cache_hit: float | None
    fallback: float | None
    validation_retry: float | None
    validation_failure: float | None


class Percentiles(_Model):
    """``n``: the rows the stage ran in (a NULL stage is ignored); p50/p95 are ``null`` at n = 0."""

    p50: int | None
    p95: int | None
    n: int


class StageLatency(_Model):
    """Milliseconds over the *timed* requests of the window (cache hits excluded)."""

    total: Percentiles
    embed: Percentiles
    retrieval: Percentiles
    rerank: Percentiles
    llm: Percentiles


class Cost(_Model):
    """Shadow cost in USD (Tech §14: what the requests would cost at paid list prices).

    ``shadow_usd_total`` sums every user request of the window, failed ones included (a failed
    request may have been billed). ``per_1k_questions_usd`` divides it by the served requests,
    cache hits included: what the demo really costs per thousand answers. ``per_1k_uncached_usd``
    divides the same total by the served requests that were not cache hits, comparable with the
    eval's ``cost_per_1k_usd`` (no cache there). Both are ``null`` on a zero denominator.
    """

    shadow_usd_total: float
    per_1k_questions_usd: float | None
    per_1k_uncached_usd: float | None


class DailyStats(_Model):
    """One UTC day of ``v_request_stats_daily`` (DB.md §8), as that view defines it: all user
    requests of the day, cache hits and throttled ones included. A day without requests is present
    with ``requests`` 0, ``shadow_usd`` 0 and null percentiles, so a chart has no gaps."""

    day: date
    requests: int
    p50_ms: int | None
    p95_ms: int | None
    shadow_usd: float


class EvalRunSummary(_Model):
    """The latest ``eval_runs`` row of one (suite, config) (DB.md §4): aggregates only."""

    suite: Literal["retrieval", "generation"]
    config: str
    created_at: datetime
    git_sha: str
    status: Literal["pass", "fail", "inconclusive"]
    case_count: int
    metrics: dict[str, float | int]
    report_url: str | None


class MetricsSummary(_Model):
    """``window`` is a rolling period ending at ``generated_at`` (the database's clock) for every
    field but ``daily``, which lists whole UTC days: the last 1, 7 or 30, today included."""

    window: Window
    generated_at: datetime
    requests: RequestCounts
    rates: Rates
    latency_ms: StageLatency
    cost: Cost
    daily: list[DailyStats]  # ordered by day
    evals: list[EvalRunSummary]  # ordered by suite, then config
