"""The Postgres rate limiter against the real schema (5.04, Tech §12, DB.md §7.4, PRD D31).

Windows are placed by an injected clock, so no test waits for a minute to pass and none depends on
when it runs. The route-level behavior (the 429, the log row, the eval switch) is in
``test_ask_protection.py``.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from grounded.infra.db import create_pool
from grounded.infra.ratelimit import RateLimiter
from tests.support import WallClock, make_settings

pytestmark = pytest.mark.integration

IP = "a" * 64
OTHER_IP = "b" * 64
ROOT = Path(__file__).resolve().parents[3]


def at(hour: int, minute: int, second: int) -> WallClock:
    return WallClock(datetime(2026, 3, 14, hour, minute, second, tzinfo=UTC))


@pytest.fixture(autouse=True)
def clean_counters(test_database_url: str) -> Iterator[None]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE rate_limit_counters")

    wipe()
    yield
    wipe()


@asynccontextmanager
async def limiter_of(
    url: str, *, per_minute: int, per_day: int, clock: WallClock | None = None, pool_size: int = 5
) -> AsyncGenerator[RateLimiter]:
    settings = make_settings(database_url=url, db_pool_min_size=1, db_pool_max_size=pool_size)
    pool = create_pool(settings)
    await pool.open(wait=True)
    try:
        yield RateLimiter(pool, per_minute=per_minute, per_day=per_day, clock=clock)
    finally:
        await pool.close()


def hits(url: str, kind: str, ip_hash: str = IP) -> int | None:
    with psycopg.connect(url) as conn:
        row = conn.execute(
            "SELECT sum(hits) FROM rate_limit_counters WHERE ip_hash = %s AND window_kind = %s",
            (ip_hash, kind),
        ).fetchone()
    assert row is not None
    return None if row[0] is None else int(row[0])


async def test_the_minute_limit_hits_exactly_at_the_configured_count(
    test_database_url: str,
) -> None:
    clock = at(10, 0, 30)
    async with limiter_of(test_database_url, per_minute=3, per_day=100, clock=clock) as limiter:
        decisions = [await limiter.check(IP) for _ in range(4)]

    assert [d.allowed for d in decisions] == [True, True, True, False]
    rejected = decisions[3]
    assert (rejected.window, rejected.retry_after_s) == ("minute", 30)  # to 10:01:00


async def test_a_new_minute_starts_at_zero(test_database_url: str) -> None:
    clock = at(10, 0, 59)
    async with limiter_of(test_database_url, per_minute=2, per_day=100, clock=clock) as limiter:
        first = [(await limiter.check(IP)).allowed for _ in range(3)]
        clock.advance(seconds=1)  # 10:01:00, the next window
        second = [(await limiter.check(IP)).allowed for _ in range(3)]

    assert first == [True, True, False]
    assert second == [True, True, False]


async def test_the_day_limit_hits_exactly_at_the_configured_count(test_database_url: str) -> None:
    clock = at(10, 0, 10)
    async with limiter_of(test_database_url, per_minute=2, per_day=3, clock=clock) as limiter:
        assert [(await limiter.check(IP)).allowed for _ in range(2)] == [True, True]
        clock.advance(minutes=1)  # a new minute window, the same day: the 3rd of 3 is admitted
        assert (await limiter.check(IP)).allowed is True
        rejected = await limiter.check(IP)  # minute: 2nd, fine; day: 4th of 3

    assert (rejected.allowed, rejected.window) == (False, "day")
    # From 10:01:10 to the next UTC midnight: 13 h 58 min 50 s.
    assert rejected.retry_after_s == 13 * 3600 + 58 * 60 + 50


async def test_a_new_utc_day_starts_at_zero(test_database_url: str) -> None:
    clock = at(23, 59, 50)
    async with limiter_of(test_database_url, per_minute=100, per_day=1, clock=clock) as limiter:
        assert (await limiter.check(IP)).allowed is True
        assert (await limiter.check(IP)).allowed is False
        clock.advance(seconds=10)  # 00:00:00 of the next day
        assert (await limiter.check(IP)).allowed is True


async def test_different_ip_hashes_are_independent(test_database_url: str) -> None:
    async with limiter_of(
        test_database_url, per_minute=1, per_day=1, clock=at(10, 0, 0)
    ) as limiter:
        assert (await limiter.check(IP)).allowed is True
        assert (await limiter.check(IP)).allowed is False
        assert (await limiter.check(OTHER_IP)).allowed is True


async def test_minute_rejected_requests_do_not_touch_the_day_counter(
    test_database_url: str,
) -> None:
    async with limiter_of(
        test_database_url, per_minute=2, per_day=100, clock=at(10, 0, 0)
    ) as limiter:
        for _ in range(7):
            await limiter.check(IP)

    assert hits(test_database_url, "day") == 2  # only the two the minute window admitted
    assert hits(test_database_url, "minute") == 7  # every attempt is counted in its own window


async def test_a_request_without_a_client_key_is_admitted_and_counted_nowhere(
    test_database_url: str,
) -> None:
    async with limiter_of(test_database_url, per_minute=1, per_day=1) as limiter:
        assert [(await limiter.check(None)).allowed for _ in range(5)] == [True] * 5

    with psycopg.connect(test_database_url) as conn:
        assert conn.execute("SELECT count(*) FROM rate_limit_counters").fetchone() == (0,)


async def test_the_default_clock_is_the_databases(test_database_url: str) -> None:
    async with limiter_of(test_database_url, per_minute=1, per_day=1) as limiter:
        assert (await limiter.check(IP)).allowed is True

    with psycopg.connect(test_database_url) as conn:
        [(start, db_now)] = conn.execute(
            "SELECT window_start, now() FROM rate_limit_counters WHERE window_kind = 'minute'"
        ).fetchall()
    assert timedelta(0) <= db_now - start < timedelta(minutes=1)
    assert (start.second, start.microsecond) == (0, 0)


@pytest.mark.parametrize("kind", ["minute", "day"])
async def test_parallel_requests_at_the_limit_minus_one_admit_exactly_one(
    test_database_url: str, kind: str
) -> None:
    limit = 5
    # One window at a time is the one under test; the other one cannot reject.
    per_minute, per_day = (limit, 1000) if kind == "minute" else (1000, limit)
    async with limiter_of(
        test_database_url,
        per_minute=per_minute,
        per_day=per_day,
        clock=at(10, 0, 0),
        pool_size=10,
    ) as limiter:
        for _ in range(limit - 1):
            assert (await limiter.check(IP)).allowed is True

        decisions = await asyncio.gather(*(limiter.check(IP) for _ in range(10)))

    assert sum(d.allowed for d in decisions) == 1
    assert hits(test_database_url, kind) == limit - 1 + 10


# --- the grant of DB.md §5 --------------------------------------------------------------------

_ROLE = "grounded_app_grant_test"


def app_grant() -> str:
    """The ``GRANT`` line for ``rate_limit_counters`` from DB.md §5, aimed at a throwaway role, so
    the test fails when the documented grant stops being enough for the limiter."""
    text = (ROOT / "docs" / "DB.md").read_text(encoding="utf-8")
    match = re.search(r"^GRANT [^\n]*? ON rate_limit_counters TO app;$", text, re.MULTILINE)
    assert match is not None, "DB.md §5 has no GRANT for rate_limit_counters"
    return match.group(0).replace(" TO app;", f" TO {_ROLE};")


@pytest.fixture
def app_role(test_database_url: str) -> Iterator[str]:
    with psycopg.connect(test_database_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(_ROLE)))
        conn.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(_ROLE)))
        conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(_ROLE)))
        conn.execute(sql.SQL(app_grant()))  # pyright: ignore[reportArgumentType]
    yield _ROLE
    with psycopg.connect(test_database_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(_ROLE)))
        conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(_ROLE)))


def test_the_grant_is_select_insert_update_and_nothing_else() -> None:
    assert app_grant().startswith("GRANT SELECT, INSERT, UPDATE ON rate_limit_counters TO ")


def test_the_app_role_can_run_the_upsert_but_cannot_delete(
    test_database_url: str, app_role: str
) -> None:
    upsert = """
        INSERT INTO rate_limit_counters AS rl (ip_hash, window_kind, window_start, hits)
        VALUES ('h', 'minute', '2026-03-14T10:00:00Z', 1)
        ON CONFLICT (ip_hash, window_kind, window_start) DO UPDATE SET hits = rl.hits + 1
        RETURNING rl.hits
    """
    with psycopg.connect(test_database_url, autocommit=True) as conn:
        conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(app_role)))
        assert conn.execute(upsert).fetchone() == (1,)
        assert conn.execute(upsert).fetchone() == (2,)  # the conflict branch, UPDATE + SELECT
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM rate_limit_counters")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("TRUNCATE rate_limit_counters")
