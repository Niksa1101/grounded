"""The daily rerank cap against the real schema (6.03, Tech §8, DB.md §7.1).

The same atomic reserve as the LLM budget (tests/integration/test_budget.py), on the
``rerank_calls`` column: atomicity under concurrency, the cap of 0, independence from the LLM
counter, and the ``app`` role's grant.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from grounded.infra.budget import DailyBudget, RerankCap, RerankCapReachedError
from grounded.infra.db import create_pool
from tests.support import make_settings

pytestmark = pytest.mark.integration

TODAY = "(now() AT TIME ZONE 'utc')::date"
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE daily_usage")

    wipe()
    yield test_database_url
    wipe()


def usage_row(url: str) -> dict[str, Any] | None:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT * FROM daily_usage WHERE day = {TODAY}")
        return cur.fetchone()


def rerank_calls_today(url: str) -> int:
    row = usage_row(url)
    return 0 if row is None else int(row["rerank_calls"])


def set_rerank_calls(url: str, calls: int) -> None:
    with psycopg.connect(url) as conn:
        conn.execute(
            f"INSERT INTO daily_usage (day, rerank_calls) VALUES ({TODAY}, %s) "
            "ON CONFLICT (day) DO UPDATE SET rerank_calls = EXCLUDED.rerank_calls",
            (calls,),
        )


@asynccontextmanager
async def pool_of(url: str, *, pool_size: int = 5) -> AsyncGenerator[Any]:
    pool = create_pool(
        make_settings(database_url=url, db_pool_min_size=1, db_pool_max_size=pool_size)
    )
    await pool.open(wait=True)
    try:
        yield pool
    finally:
        await pool.close()


async def test_reserve_counts_up_to_the_limit_and_then_refuses(db: str) -> None:
    async with pool_of(db) as pool:
        cap = RerankCap(pool, limit=3)
        for _ in range(3):
            await cap.reserve()
        with pytest.raises(RerankCapReachedError):
            await cap.reserve()
        with pytest.raises(RerankCapReachedError):
            await cap.reserve()

    assert rerank_calls_today(db) == 3  # a refused reserve does not count itself


async def test_parallel_reserves_at_the_limit_minus_one_admit_exactly_one(db: str) -> None:
    limit = 5
    set_rerank_calls(db, limit - 1)
    async with pool_of(db, pool_size=10) as pool:
        cap = RerankCap(pool, limit=limit)
        results = await asyncio.gather(*(cap.reserve() for _ in range(10)), return_exceptions=True)

    assert [r for r in results if r is None] == [None]  # exactly one reserve returned
    assert all(isinstance(r, RerankCapReachedError) for r in results if r is not None)
    assert rerank_calls_today(db) == limit  # and the counter never went past the cap


async def test_parallel_reserves_on_a_day_without_a_row_never_pass_the_cap(db: str) -> None:
    # The first reserve of a day takes the insert branch, which has no limit check of its own: ten
    # callers racing for it must still be admitted exactly ``limit`` times in all.
    limit = 4
    async with pool_of(db, pool_size=10) as pool:
        cap = RerankCap(pool, limit=limit)
        results = await asyncio.gather(*(cap.reserve() for _ in range(10)), return_exceptions=True)

    assert len([r for r in results if r is None]) == limit
    assert all(isinstance(r, RerankCapReachedError) for r in results if r is not None)
    assert rerank_calls_today(db) == limit


async def test_a_cap_of_zero_refuses_everything_and_writes_nothing(db: str) -> None:
    async with pool_of(db) as pool:
        cap = RerankCap(pool, limit=0)
        for _ in range(2):
            with pytest.raises(RerankCapReachedError):
                await cap.reserve()

    # Not even the insert branch ran: it would have taken a unit of a cap of 0.
    assert usage_row(db) is None


async def test_the_rerank_cap_and_the_llm_budget_count_separately(db: str) -> None:
    async with pool_of(db) as pool:
        cap, budget = RerankCap(pool, limit=2), DailyBudget(pool, limit=10)
        await cap.reserve()
        await budget.reserve()
        await budget.reserve()
        await cap.reserve()
        with pytest.raises(RerankCapReachedError):
            await cap.reserve()
        await budget.reserve()  # the rerank cap is spent, the LLM budget is not

    row = usage_row(db)
    assert row is not None
    assert (row["rerank_calls"], row["llm_calls"]) == (2, 3)


async def test_a_database_error_propagates_and_is_not_a_cap_reached(db: str) -> None:
    # The pipeline tells these apart: the cap means "do not call", the error means "could not
    # count, so do not call either" and is logged as a different cause.
    async with pool_of(db) as pool:
        cap = RerankCap(pool, limit=3)
    with pytest.raises(psycopg.Error):  # the pool is closed now
        await cap.reserve()


# --- the grant of DB.md §5 ---------------------------------------------------------------------

_ROLE = "grounded_app_rerank_cap_grant_test"


def app_grant() -> str:
    """The ``GRANT`` line for ``daily_usage`` from DB.md §5, aimed at a throwaway role."""
    text = (ROOT / "docs" / "DB.md").read_text(encoding="utf-8")
    match = [
        ln for ln in text.splitlines() if ln.startswith("GRANT") and "ON daily_usage TO app;" in ln
    ]
    assert len(match) == 1, "DB.md §5 must have exactly one GRANT for daily_usage"
    return match[0].replace(" TO app;", f" TO {_ROLE};")


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


def test_the_app_role_can_reserve_rerank_units_but_cannot_delete(db: str, app_role: str) -> None:
    from grounded.infra import budget as module

    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(app_role)))
        for expected in (1, 2):  # the insert branch, then the conflict branch (UPDATE + RETURNING)
            row = conn.execute(module._RESERVE_RERANK, {"limit": 2}).fetchone()  # pyright: ignore[reportPrivateUsage, reportArgumentType]
            assert row == (expected,)
        assert conn.execute(module._RESERVE_RERANK, {"limit": 2}).fetchone() is None  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM daily_usage")
