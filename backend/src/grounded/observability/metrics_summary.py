"""The read side of the dashboard: ``GET /v1/metrics/summary`` (Tech.md §14, DB.md §8; ticket 8.02).

**Views only.** The request numbers come from ``v_request_metrics`` (window aggregates) and
``v_request_stats_daily`` (the ``daily`` series), never from ``request_logs``: the views are the
query contract that leaves out ``question``, ``question_hash``, ``ip_hash``, the request id and
CI's ``source = 'eval'`` rows, and in production the runtime role can still read the table itself
(DB.md §8), so reading only the views is what keeps those columns off the page. The eval table
comes from ``eval_runs``, which holds aggregates only (DB.md §4).

**One window, one snapshot.** The window is a bound parameter (``make_interval(days => %s)``),
mapped from the enum in ``WINDOW_DAYS`` and nowhere else. The three queries run in one transaction,
so ``now()`` is the same instant in all of them and in ``generated_at``.

The denominators and the rounding rules are the ones documented on ``schemas/metrics.py``; the SQL
below computes the counts, and this module only divides.
"""

from __future__ import annotations

from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal
from types import MappingProxyType
from typing import Any, Final

from psycopg import AsyncConnection
from psycopg.rows import dict_row

from grounded.schemas.metrics import (
    Cost,
    DailyStats,
    EvalRunSummary,
    MetricsSummary,
    Percentiles,
    Rates,
    RequestCounts,
    StageLatency,
    Window,
)

WINDOW_DAYS: Final[Mapping[Window, int]] = MappingProxyType({"1d": 1, "7d": 7, "30d": 30})

_USD_PLACES = Decimal("0.00000001")  # numeric(12, 8), as request_logs.shadow_cost_usd
_STAGES = ("total", "embed", "retrieval", "rerank", "llm")

# served: an answer was returned. generated: the LLM stage ran (never true for a cache hit).
# timed: the request entered the pipeline and was not a cache hit (the latency rows).
_WINDOW_SQL = """
WITH w AS (
    SELECT
        m.*,
        outcome IN ('answered', 'partial', 'insufficient_context')       AS served,
        outcome IN ('rate_limited', 'budget_exhausted')                  AS throttled,
        NOT cache_hit AND latency_llm_ms IS NOT NULL                     AS generated,
        NOT cache_hit
            AND outcome NOT IN ('rate_limited', 'budget_exhausted', 'bad_request') AS timed
    FROM v_request_metrics AS m
    WHERE created_at > now() - make_interval(days => %(days)s)
)
SELECT
    now()                                                         AS generated_at,
    count(*)                                                      AS total,
    count(*) FILTER (WHERE outcome = 'answered')                  AS answered,
    count(*) FILTER (WHERE outcome = 'partial')                   AS partial,
    count(*) FILTER (WHERE outcome = 'insufficient_context')      AS insufficient_context,
    count(*) FILTER (WHERE throttled)                             AS throttled,
    count(*) FILTER (WHERE NOT served AND NOT throttled)          AS errors,
    count(*) FILTER (WHERE cache_hit)                             AS cache_hits,
    count(*) FILTER (WHERE served)                                AS served,
    count(*) FILTER (WHERE served AND NOT cache_hit)              AS served_uncached,
    count(*) FILTER (WHERE served AND cache_hit)                  AS served_cached,
    count(*) FILTER (WHERE generated)                             AS generated,
    count(*) FILTER (WHERE generated AND fallback_used)           AS fallbacks,
    count(*) FILTER (WHERE generated AND validation_retries > 0)  AS validation_retries,
    count(*) FILTER (WHERE generated AND outcome = 'validation_failed') AS validation_failures,
    coalesce(sum(shadow_cost_usd), 0)                             AS shadow_usd_total,
    round((percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_total_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS total_p50,
    round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_total_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS total_p95,
    count(latency_total_ms) FILTER (WHERE timed)                  AS total_n,
    round((percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_embed_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS embed_p50,
    round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_embed_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS embed_p95,
    count(latency_embed_ms) FILTER (WHERE timed)                  AS embed_n,
    round((percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_retrieval_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS retrieval_p50,
    round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_retrieval_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS retrieval_p95,
    count(latency_retrieval_ms) FILTER (WHERE timed)              AS retrieval_n,
    round((percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_rerank_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS rerank_p50,
    round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_rerank_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS rerank_p95,
    count(latency_rerank_ms) FILTER (WHERE timed)                 AS rerank_n,
    round((percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_llm_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS llm_p50,
    round((percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_llm_ms)
        FILTER (WHERE timed))::numeric)::integer                  AS llm_p95,
    count(latency_llm_ms) FILTER (WHERE timed)                    AS llm_n
FROM w
"""

# The last N whole UTC days, today included, each present even without a row in the view.
_DAILY_SQL = """
WITH days AS (
    SELECT (now() AT TIME ZONE 'utc')::date - g AS day
    FROM generate_series(0, %(days)s - 1) AS g
)
SELECT
    d.day,
    coalesce(s.requests, 0)                   AS requests,
    round(s.p50_ms::numeric)::integer         AS p50_ms,
    round(s.p95_ms::numeric)::integer         AS p95_ms,
    coalesce(s.shadow_cost_usd, 0)            AS shadow_usd
FROM days AS d
LEFT JOIN v_request_stats_daily AS s ON s.day = d.day
ORDER BY d.day
"""

# The latest row per (suite, config); id breaks a tie on created_at, so the pick is deterministic.
_EVALS_SQL = """
SELECT DISTINCT ON (suite, config_name)
    suite, config_name AS config, created_at, git_sha, status, case_count, metrics, report_url
FROM eval_runs
ORDER BY suite, config_name, created_at DESC, id DESC
"""


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _usd(value: Decimal) -> float:
    return float(value.quantize(_USD_PLACES, rounding=ROUND_HALF_UP))


def _per_1k(total: Decimal, denominator: int) -> float | None:
    return None if denominator == 0 else _usd(total * 1000 / denominator)


async def fetch_metrics_summary(conn: AsyncConnection[Any], window: Window) -> MetricsSummary:
    """Everything the dashboard shows for ``window``, read in one transaction of ``conn``."""
    params = {"days": WINDOW_DAYS[window]}
    async with conn.transaction(), conn.cursor(row_factory=dict_row) as cur:
        await cur.execute(_WINDOW_SQL, params)
        agg = await cur.fetchone()
        await cur.execute(_DAILY_SQL, params)
        daily = await cur.fetchall()
        await cur.execute(_EVALS_SQL)
        evals = await cur.fetchall()
    if agg is None:  # an aggregate without GROUP BY always returns one row
        raise RuntimeError("the window aggregate returned no row")

    total_usd: Decimal = agg["shadow_usd_total"]
    generated: int = agg["generated"]
    return MetricsSummary(
        window=window,
        generated_at=agg["generated_at"],
        requests=RequestCounts(
            total=agg["total"],
            answered=agg["answered"],
            partial=agg["partial"],
            insufficient_context=agg["insufficient_context"],
            throttled=agg["throttled"],
            errors=agg["errors"],
            cache_hits=agg["cache_hits"],
        ),
        rates=Rates(
            cache_hit=_ratio(agg["served_cached"], agg["served"]),
            fallback=_ratio(agg["fallbacks"], generated),
            validation_retry=_ratio(agg["validation_retries"], generated),
            validation_failure=_ratio(agg["validation_failures"], generated),
        ),
        latency_ms=StageLatency(
            **{
                stage: Percentiles(
                    p50=agg[f"{stage}_p50"], p95=agg[f"{stage}_p95"], n=agg[f"{stage}_n"]
                )
                for stage in _STAGES
            }
        ),
        cost=Cost(
            shadow_usd_total=_usd(total_usd),
            per_1k_questions_usd=_per_1k(total_usd, agg["served"]),
            per_1k_uncached_usd=_per_1k(total_usd, agg["served_uncached"]),
        ),
        daily=[
            DailyStats(
                day=row["day"],
                requests=row["requests"],
                p50_ms=row["p50_ms"],
                p95_ms=row["p95_ms"],
                shadow_usd=_usd(row["shadow_usd"]),
            )
            for row in daily
        ],
        evals=[EvalRunSummary.model_validate(row) for row in evals],
    )
