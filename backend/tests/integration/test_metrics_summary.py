"""``GET /v1/metrics/summary`` on the local pgvector DB (ticket 8.02, Tech §13/§14, DB.md §8).

The service tests run ``fetch_metrics_summary`` on one async connection inside a transaction that
starts by deleting every ``request_logs`` and ``eval_runs`` row and is rolled back at the end, so
rows other tests committed cannot change an aggregate. Rows are seeded relative to the database's
``now()``, which is constant inside a transaction, so the window edges are exact. The route test at
the end commits its rows (the app reads through its own pool) and truncates them afterwards.

Every seeded row carries the same question, question hash, IP hash and a known request id, and the
privacy tests search the serialized JSON for each of them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import date, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
import pytest
from psycopg.types.json import Jsonb

from grounded.observability.metrics_summary import fetch_metrics_summary
from grounded.schemas.metrics import MetricsSummary
from tests.support import app_client, make_settings

pytestmark = pytest.mark.integration

QUESTION = "How do I configure the secret-question-text dependency?"
QUESTION_HASH = "5e3c0a1f" * 8
IP_HASH = "c0ffee42" * 8
REQUEST_ID = UUID("0badc0de-0000-4000-8000-00000000beef")

_INSERT = """
INSERT INTO request_logs (
    created_at, source, question, question_hash, ip_hash, http_status, outcome, cache_hit,
    fallback_used, validation_retries, provider, model, latency_total_ms, latency_embed_ms,
    latency_retrieval_ms, latency_rerank_ms, latency_llm_ms, input_tokens, output_tokens,
    shadow_cost_usd)
VALUES (
    now() - %(age)s, %(source)s, %(question)s, %(question_hash)s, %(ip_hash)s, %(http_status)s,
    %(outcome)s, %(cache_hit)s, %(fallback_used)s, %(validation_retries)s, %(provider)s, %(model)s,
    %(latency_total_ms)s, %(latency_embed_ms)s, %(latency_retrieval_ms)s, %(latency_rerank_ms)s,
    %(latency_llm_ms)s, %(input_tokens)s, %(output_tokens)s, %(shadow_cost_usd)s)
"""

_INSERT_EVAL = """
INSERT INTO eval_runs (
    id, created_at, suite, config_name, git_sha, branch, golden_set_version, index_config_hash,
    status, metrics, case_count, report_url)
VALUES (
    %(id)s, now() - %(age)s, %(suite)s, %(config)s, %(git_sha)s, 'main', 'v1', 'cfg',
    %(status)s, %(metrics)s, %(case_count)s, %(report_url)s)
"""


def _row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "age": timedelta(hours=1),
        "source": "web",
        "question": QUESTION,
        "question_hash": QUESTION_HASH,
        "ip_hash": IP_HASH,
        "http_status": 200,
        "outcome": "answered",
        "cache_hit": False,
        "fallback_used": False,
        "validation_retries": 0,
        "provider": "gemini",
        "model": "gemini-test",
        "latency_total_ms": 100,
        "latency_embed_ms": None,
        "latency_retrieval_ms": None,
        "latency_rerank_ms": None,
        "latency_llm_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "shadow_cost_usd": None,
    }
    unknown = set(overrides) - set(row)
    assert not unknown, unknown
    row.update(overrides)
    return row


def _generated(**overrides: Any) -> dict[str, Any]:
    """An uncached request that went through every stage but rerank."""
    stages = {"latency_embed_ms": 10, "latency_retrieval_ms": 20, "latency_llm_ms": 60}
    return _row(**{**stages, **overrides})


async def insert(conn: psycopg.AsyncConnection[Any], **overrides: Any) -> None:
    await conn.execute(_INSERT, _row(**overrides))


async def insert_generated(conn: psycopg.AsyncConnection[Any], **overrides: Any) -> None:
    await conn.execute(_INSERT, _generated(**overrides))


async def insert_eval(conn: psycopg.AsyncConnection[Any], **overrides: Any) -> None:
    row: dict[str, Any] = {
        "id": UUID(int=1),
        "age": timedelta(days=1),
        "suite": "generation",
        "config": "hybrid",
        "git_sha": "a" * 40,
        "status": "pass",
        "metrics": Jsonb({"faithfulness": 0.9, "n": 30}),
        "case_count": 30,
        "report_url": "https://github.com/example/actions/runs/1",
    }
    row.update(overrides)
    await conn.execute(_INSERT_EVAL, row)


@pytest.fixture
async def conn(test_database_url: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    """One transaction, rolled back: it starts with empty ``request_logs`` and ``eval_runs``."""
    async with await psycopg.AsyncConnection.connect(test_database_url) as connection:
        await connection.execute("DELETE FROM request_logs")
        await connection.execute("DELETE FROM eval_runs")
        yield connection
        await connection.rollback()


async def test_an_empty_database_has_zero_counts_and_null_rates(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    summary = await fetch_metrics_summary(conn, "7d")

    assert summary.window == "7d"
    assert summary.requests.model_dump() == {
        "total": 0,
        "answered": 0,
        "partial": 0,
        "insufficient_context": 0,
        "throttled": 0,
        "errors": 0,
        "cache_hits": 0,
    }
    assert summary.rates.model_dump() == {
        "cache_hit": None,
        "fallback": None,
        "validation_retry": None,
        "validation_failure": None,
    }
    for stage in summary.latency_ms.model_dump().values():
        assert stage == {"p50": None, "p95": None, "n": 0}
    assert summary.cost.model_dump() == {
        "shadow_usd_total": 0.0,
        "per_1k_questions_usd": None,
        "per_1k_uncached_usd": None,
    }
    assert len(summary.daily) == 7
    assert all(d.requests == 0 and d.p50_ms is None and d.shadow_usd == 0 for d in summary.daily)
    assert summary.evals == []


async def test_the_window_is_a_rolling_period_with_an_open_lower_edge(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    for age in (
        timedelta(hours=2),
        timedelta(days=3),
        timedelta(days=7),  # exactly on the 7d edge: outside (created_at > now() - 7 days)
        timedelta(days=10),
        timedelta(days=40),
    ):
        await insert(conn, age=age)

    totals = {w: (await fetch_metrics_summary(conn, w)).requests.total for w in ("1d", "7d", "30d")}

    assert totals == {"1d": 1, "7d": 2, "30d": 4}


async def test_eval_rows_are_left_out_everywhere(conn: psycopg.AsyncConnection[Any]) -> None:
    await insert_generated(conn, latency_total_ms=100, shadow_cost_usd=Decimal("0.001"))
    await insert_generated(
        conn,
        source="eval",
        latency_total_ms=99_999,
        fallback_used=True,
        cache_hit=False,
        shadow_cost_usd=Decimal("5"),
    )
    await insert(conn, source="eval", cache_hit=True, age=timedelta(0))

    summary = await fetch_metrics_summary(conn, "7d")

    assert summary.requests.total == 1
    assert summary.requests.cache_hits == 0
    assert summary.latency_ms.total.p95 == 100
    assert summary.rates.fallback == 0.0
    assert summary.cost.shadow_usd_total == 0.001
    assert sum(d.requests for d in summary.daily) == 1


async def test_latency_leaves_out_cache_hits_and_requests_outside_the_pipeline(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    for total in (100, 200, 300, 400):
        await insert_generated(conn, latency_total_ms=total)
    # A timeout stays in: the p95 must show it.
    await insert_generated(
        conn,
        outcome="deadline_exceeded",
        http_status=504,
        latency_total_ms=500,
        latency_llm_ms=None,
    )
    # Never timed: a cache hit (retrieval ran, the index lookup), and requests outside the pipeline.
    await insert(conn, cache_hit=True, latency_total_ms=1, latency_retrieval_ms=1)
    await insert(conn, outcome="rate_limited", http_status=429, latency_total_ms=2)
    await insert(conn, outcome="budget_exhausted", http_status=503, latency_total_ms=3)
    await insert(conn, outcome="bad_request", http_status=422, latency_total_ms=4)
    # One reranked request: the rerank stage's n counts only the rows it ran in.
    await insert_generated(conn, latency_total_ms=300, latency_rerank_ms=41)
    await insert_generated(conn, latency_total_ms=300, latency_rerank_ms=42)

    latency = (await fetch_metrics_summary(conn, "7d")).latency_ms

    # 100, 200, 300, 300, 300, 400, 500: p50 300; p95 at rank 5.7 is 400 + 0.7 * 100.
    assert latency.total.model_dump() == {"p50": 300, "p95": 470, "n": 7}
    assert latency.embed.model_dump() == {"p50": 10, "p95": 10, "n": 7}
    assert latency.llm.n == 6  # the timed-out request never reached the LLM stage
    # p50 of 41, 42 is 41.5: rounded half away from zero; p95 is 41.95 -> 42.
    assert latency.rerank.model_dump() == {"p50": 42, "p95": 42, "n": 2}


async def test_counts_partition_the_total_and_rates_use_their_own_denominators(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    # Served, generated: 4 (one with a retry, one that fell back), plus one partial and one refusal.
    await insert_generated(conn)
    await insert_generated(conn, validation_retries=1)
    await insert_generated(conn, fallback_used=True)
    await insert_generated(conn, outcome="partial")
    await insert_generated(conn, outcome="insufficient_context")
    # Served from the cache: in the cache-hit rate, not in the generation rates.
    await insert(conn, cache_hit=True)
    await insert(conn, cache_hit=True, outcome="insufficient_context")
    # Generated but failed: in the generation rates, not served.
    await insert_generated(
        conn, outcome="validation_failed", http_status=502, validation_retries=1, fallback_used=True
    )
    # Never generated: throttled and errors before the LLM stage.
    await insert(conn, outcome="rate_limited", http_status=429)
    await insert(conn, outcome="bad_request", http_status=422)
    await insert(conn, outcome="internal_error", http_status=500)

    summary = await fetch_metrics_summary(conn, "7d")

    assert summary.requests.model_dump() == {
        "total": 11,
        "answered": 4,
        "partial": 1,
        "insufficient_context": 2,
        "throttled": 1,
        "errors": 3,
        "cache_hits": 2,
    }
    counts = summary.requests
    assert (
        counts.answered + counts.partial + counts.insufficient_context + counts.throttled
    ) + counts.errors == counts.total
    assert summary.rates.model_dump() == {
        "cache_hit": 2 / 7,  # over served
        "fallback": 2 / 6,  # over generated
        "validation_retry": 2 / 6,
        "validation_failure": 1 / 6,
    }


async def test_cost_has_a_blended_and_an_uncached_number(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    for cost in ("0.001", "0.002", "0.003"):
        await insert_generated(conn, shadow_cost_usd=Decimal(cost))
    await insert(conn, cache_hit=True, shadow_cost_usd=None)
    # A failed request was billed too: it is in the total, not in a denominator.
    await insert_generated(
        conn, outcome="validation_failed", http_status=502, shadow_cost_usd=Decimal("0.002")
    )

    cost = (await fetch_metrics_summary(conn, "7d")).cost

    assert cost.shadow_usd_total == 0.008
    assert cost.per_1k_questions_usd == 2.0  # 0.008 / 4 served * 1000
    assert cost.per_1k_uncached_usd == 2.66666667  # 0.008 / 3 uncached served * 1000, 8 places


async def test_daily_lists_every_utc_day_of_the_window_in_order(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    await insert(conn, age=timedelta(0), latency_total_ms=100, shadow_cost_usd=Decimal("0.001"))
    await insert(conn, age=timedelta(0), latency_total_ms=201, shadow_cost_usd=Decimal("0.002"))
    await insert(conn, age=timedelta(days=1), latency_total_ms=50)
    await insert(conn, age=timedelta(days=40))  # outside every window
    cur = await conn.execute("SELECT (now() AT TIME ZONE 'utc')::date")
    row = await cur.fetchone()
    assert row is not None
    today: date = row[0]

    daily = (await fetch_metrics_summary(conn, "7d")).daily

    assert [d.day for d in daily] == [today - timedelta(days=n) for n in range(6, -1, -1)]
    assert daily[-1].model_dump() == {
        "day": today,
        "requests": 2,
        "p50_ms": 151,  # 150.5, half away from zero
        "p95_ms": 196,  # 100 + 0.95 * 101 = 195.95
        "shadow_usd": 0.003,
    }
    assert (daily[-2].requests, daily[-2].p50_ms, daily[-2].shadow_usd) == (1, 50, 0.0)
    assert all(d.requests == 0 and d.p95_ms is None for d in daily[:-2])
    assert len((await fetch_metrics_summary(conn, "1d")).daily) == 1
    assert len((await fetch_metrics_summary(conn, "30d")).daily) == 30


async def test_evals_are_the_latest_row_per_suite_and_config_with_a_tie_breaker(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    await insert_eval(conn, id=UUID(int=1), age=timedelta(days=3), status="fail")
    await insert_eval(conn, id=UUID(int=2), age=timedelta(days=1), git_sha="b" * 40)
    # Same (suite, config) and same instant: the larger id wins, every time.
    await insert_eval(conn, id=UUID(int=3), age=timedelta(days=1), git_sha="c" * 40)
    await insert_eval(conn, id=UUID(int=4), config="no_rag", status="inconclusive", report_url=None)
    await insert_eval(
        conn, id=UUID(int=5), suite="retrieval", config="dense", age=timedelta(days=9)
    )

    evals = (await fetch_metrics_summary(conn, "1d")).evals  # not limited by the window

    assert [(e.suite, e.config, e.git_sha) for e in evals] == [
        ("generation", "hybrid", "c" * 40),
        ("generation", "no_rag", "a" * 40),
        ("retrieval", "dense", "a" * 40),
    ]
    first = evals[0].model_dump(mode="json")
    assert set(first) == {
        "suite",
        "config",
        "created_at",
        "git_sha",
        "status",
        "case_count",
        "metrics",
        "report_url",
    }
    assert first["metrics"] == {"faithfulness": 0.9, "n": 30}
    assert evals[1].report_url is None


async def seed_everything(conn: psycopg.AsyncConnection[Any]) -> None:
    """One row of every kind, all carrying the private values (and one with a known id)."""
    await insert_generated(conn, shadow_cost_usd=Decimal("0.001"))
    await insert(conn, cache_hit=True)
    await insert(conn, outcome="rate_limited", http_status=429)
    await insert(conn, source="eval")
    await conn.execute(
        "UPDATE request_logs SET id = %s WHERE id = (SELECT id FROM request_logs LIMIT 1)",
        (REQUEST_ID,),
    )
    await insert_eval(conn)


def assert_private_values_absent(text: str) -> None:
    for secret in (QUESTION, "secret-question-text", QUESTION_HASH, IP_HASH, str(REQUEST_ID)):
        assert secret not in text


async def test_the_serialized_summary_holds_no_question_hash_ip_or_request_id(
    conn: psycopg.AsyncConnection[Any],
) -> None:
    await seed_everything(conn)

    summary = await fetch_metrics_summary(conn, "30d")

    assert summary.requests.total == 3
    assert_private_values_absent(summary.model_dump_json())


# --- the route, end to end ------------------------------------------------------------------------

SECRET = "proxy-secret-for-tests"


@pytest.fixture
def committed_db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as sync:
            sync.execute("TRUNCATE request_logs, eval_runs")

    wipe()
    yield test_database_url
    wipe()


async def test_the_route_returns_the_summary_and_writes_no_request_log_row(
    committed_db: str,
) -> None:
    async with await psycopg.AsyncConnection.connect(committed_db) as seeding:
        await seed_everything(seeding)  # committed on exit
    settings = make_settings(database_url=committed_db, proxy_shared_secret=SECRET)

    async with app_client(settings, direct_api=False) as client:
        response = await client.get(
            "/v1/metrics/summary", params={"window": "30d"}, headers={"X-Proxy-Secret": SECRET}
        )
        default = await client.get("/v1/metrics/summary", headers={"X-Proxy-Secret": SECRET})

    assert response.status_code == 200, response.text
    summary = MetricsSummary.model_validate_json(response.text)
    assert summary.window == "30d"
    assert summary.requests.total == 3
    assert len(summary.daily) == 30
    assert [e.config for e in summary.evals] == ["hybrid"]
    assert_private_values_absent(response.text)
    assert default.status_code == 200
    assert default.json()["window"] == "7d"
    with psycopg.connect(committed_db) as sync:
        assert sync.execute("SELECT count(*) FROM request_logs").fetchone() == (4,)
