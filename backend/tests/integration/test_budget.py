"""The global daily LLM budget against the real schema (5.05, Tech §12, DB.md §7.1, PRD D31).

The first half drives ``DailyBudget`` directly (atomicity, the advisory check, tokens, the
documented grant). The second half goes through the real app: the 503, the log row, the order
against the answer cache and the embedder, retries, and the eval switch.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from grounded.generation.providers.base import Usage
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.budget import BudgetExhaustedError, DailyBudget
from grounded.infra.db import create_pool
from grounded.ingest.embed import FakeEmbedder
from grounded.schemas.llm import LLMAnswer, LLMClaim
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import (
    EMBEDDING_DIM,
    EVAL_SETTINGS,
    app_client,
    insert_index_version,
    make_settings,
)

pytestmark = pytest.mark.integration

QUESTION = "Where does the quokka sleep?"
OTHER_QUESTION = "Where does the wombat sleep?"
TODAY = "(now() AT TIME ZONE 'utc')::date"
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")
            conn.execute("TRUNCATE daily_usage")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    yield test_database_url
    wipe()


def usage_row(url: str) -> dict[str, Any] | None:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT * FROM daily_usage WHERE day = {TODAY}")
        return cur.fetchone()


def set_calls(url: str, calls: int) -> None:
    with psycopg.connect(url) as conn:
        conn.execute(
            f"INSERT INTO daily_usage (day, llm_calls) VALUES ({TODAY}, %s) "
            "ON CONFLICT (day) DO UPDATE SET llm_calls = EXCLUDED.llm_calls",
            (calls,),
        )


def calls_today(url: str) -> int:
    row = usage_row(url)
    return 0 if row is None else int(row["llm_calls"])


@asynccontextmanager
async def budget_of(url: str, *, limit: int, pool_size: int = 5) -> AsyncGenerator[DailyBudget]:
    pool = create_pool(
        make_settings(database_url=url, db_pool_min_size=1, db_pool_max_size=pool_size)
    )
    await pool.open(wait=True)
    try:
        yield DailyBudget(pool, limit=limit)
    finally:
        await pool.close()


# --- DailyBudget -------------------------------------------------------------------------------


async def test_reserve_counts_up_to_the_limit_and_then_refuses(db: str) -> None:
    async with budget_of(db, limit=3) as budget:
        for _ in range(3):
            await budget.reserve()
        with pytest.raises(BudgetExhaustedError):
            await budget.reserve()
        with pytest.raises(BudgetExhaustedError):
            await budget.reserve()

    assert calls_today(db) == 3  # a refused reserve does not count itself


async def test_parallel_reserves_at_the_limit_minus_one_admit_exactly_one(db: str) -> None:
    limit = 5
    async with budget_of(db, limit=limit, pool_size=10) as budget:
        for _ in range(limit - 1):
            await budget.reserve()

        results = await asyncio.gather(
            *(budget.reserve() for _ in range(10)), return_exceptions=True
        )

    assert [r for r in results if r is None] == [None]  # exactly one reserve returned
    assert all(isinstance(r, BudgetExhaustedError) for r in results if r is not None)
    assert calls_today(db) == limit  # and the counter never went past the cap


async def test_the_advisory_check_reads_but_reserves_nothing(db: str) -> None:
    async with budget_of(db, limit=2) as budget:
        await budget.check_available()  # no row for today: zero calls
        assert usage_row(db) is None  # and the check did not create one

        set_calls(db, 1)
        await budget.check_available()
        assert calls_today(db) == 1

        set_calls(db, 2)
        with pytest.raises(BudgetExhaustedError):
            await budget.check_available()
        assert calls_today(db) == 2


async def test_tokens_accumulate_on_the_days_row(db: str) -> None:
    async with budget_of(db, limit=10) as budget:
        await budget.reserve()
        await budget.add_tokens(Usage(input_tokens=100, output_tokens=40))
        await budget.add_tokens(Usage(input_tokens=7, output_tokens=3, thinking_tokens=2))

    row = usage_row(db)
    assert row is not None
    assert (row["llm_calls"], row["input_tokens"], row["output_tokens"]) == (1, 107, 43)


async def test_adding_tokens_is_best_effort(db: str, caplog: pytest.LogCaptureFixture) -> None:
    # The answer exists once the call returned: a statistic that cannot be written is logged.
    async with budget_of(db, limit=10) as budget:
        await budget.add_tokens(Usage(input_tokens=1, output_tokens=1))  # no row yet: no-op
        assert usage_row(db) is None
    with caplog.at_level("ERROR", logger="grounded.infra.budget"):
        await budget.add_tokens(Usage(input_tokens=1, output_tokens=1))  # the pool is closed now
    assert [r.getMessage() for r in caplog.records] == ["adding tokens to daily_usage failed"]
    with pytest.raises(psycopg.Error):  # the reserve, unlike the bookkeeping, must fail loudly
        await budget.reserve()


# --- the grant of DB.md §5 ---------------------------------------------------------------------

_ROLE = "grounded_app_budget_grant_test"


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


def test_the_app_role_can_reserve_peek_and_add_tokens_but_cannot_delete(
    db: str, app_role: str
) -> None:
    from grounded.infra import budget as module

    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(app_role)))
        for _ in range(2):  # the insert branch, then the conflict branch (UPDATE + SELECT)
            assert conn.execute(module._RESERVE, {"limit": 5}).fetchone() is not None  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        assert conn.execute(module._PEEK).fetchone() == (2,)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        conn.execute(module._ADD_TOKENS, {"input": 1, "output": 1})  # pyright: ignore[reportPrivateUsage, reportArgumentType]
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("DELETE FROM daily_usage")


# --- through the app ---------------------------------------------------------------------------


def good() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[claim])


def request_rows(url: str) -> list[dict[str, Any]]:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM request_logs ORDER BY created_at, id")
        return cur.fetchall()


async def ask(client: httpx.AsyncClient, question: str = QUESTION) -> httpx.Response:
    return await client.post("/v1/ask", json={"question": question})


def capped(db: str, budget: int, **overrides: Any) -> Any:
    return make_settings(database_url=db, daily_llm_budget=budget, **overrides)


async def test_an_answered_request_reserves_one_call_and_adds_its_tokens(db: str) -> None:
    llm = FakeLLMProvider([good()])  # reports 100 input and 50 output tokens
    async with app_client(capped(db, 10), provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 200
    row = usage_row(db)
    assert row is not None
    assert (row["llm_calls"], row["input_tokens"], row["output_tokens"]) == (1, 100, 50)


async def test_a_retry_is_another_call_and_reserves_again(db: str) -> None:
    llm = FakeLLMProvider(["{", good()])  # invalid JSON, then a good answer
    async with app_client(capped(db, 10), provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 200
    assert len(llm.calls) == 2
    row = usage_row(db)
    assert row is not None
    # Both attempts were billed, the invalid one included.
    assert (row["llm_calls"], row["input_tokens"], row["output_tokens"]) == (2, 200, 100)


async def test_a_retry_that_finds_the_budget_gone_ends_in_budget_exhausted(db: str) -> None:
    llm = FakeLLMProvider(["{"])  # a second call would exhaust the script and fail the test
    async with app_client(capped(db, 1), provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "budget_exhausted"
    assert len(llm.calls) == 1
    [row] = request_rows(db)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "budget_exhausted",
        503,
        "budget_exhausted",
    )
    assert row["input_tokens"] == 100  # what the first attempt cost is still on the row
    assert calls_today(db) == 1


async def test_an_exhausted_day_is_a_503_that_embeds_and_generates_nothing(db: str) -> None:
    set_calls(db, 5)
    embedder, llm = FakeEmbedder(dim=EMBEDDING_DIM), FakeLLMProvider([])
    async with app_client(capped(db, 5), embedder=embedder, provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "budget_exhausted"
    assert body["error"]["retry_after_s"] is None
    assert "Retry-After" not in response.headers
    assert embedder.calls == []  # the advisory check came before the embed
    assert llm.calls == []
    [row] = request_rows(db)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "budget_exhausted",
        503,
        "budget_exhausted",
    )
    assert str(row["id"]) == body["request_id"]
    assert row["question"] == QUESTION
    assert calls_today(db) == 5  # nothing was reserved


async def test_the_reserve_enforces_the_cap_when_the_advisory_check_lets_a_request_through(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def always_free(self: DailyBudget) -> None:
        return None

    monkeypatch.setattr(DailyBudget, "check_available", always_free)  # as if it lost the race
    set_calls(db, 5)
    embedder, llm = FakeEmbedder(dim=EMBEDDING_DIM), FakeLLMProvider([])
    async with app_client(capped(db, 5), embedder=embedder, provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "budget_exhausted"
    assert len(embedder.calls) == 1  # it got as far as the embedding, and no further
    assert llm.calls == []
    assert calls_today(db) == 5


async def test_a_cache_hit_is_served_on_an_exhausted_day_and_touches_no_usage(db: str) -> None:
    embedder, llm = FakeEmbedder(dim=EMBEDDING_DIM), FakeLLMProvider([good()])
    async with app_client(capped(db, 1), embedder=embedder, provider=llm) as client:
        miss = await ask(client)
        assert miss.status_code == 200
        before = usage_row(db)
        assert before is not None
        assert before["llm_calls"] == 1  # the day is now exhausted

        hit = await ask(client, "  where DOES the quokka sleep ?? ")  # same key after normalizing
        new = await ask(client, OTHER_QUESTION)

    assert hit.status_code == 200
    assert hit.json()["meta"]["cache_hit"] is True
    assert usage_row(db) == before  # not a call, not a token, not even updated_at
    assert len(llm.calls) == 1
    assert len(embedder.calls) == 1
    assert new.status_code == 503  # while a question that is not cached is refused
    assert new.json()["error"]["code"] == "budget_exhausted"


@pytest.mark.parametrize(
    ("overrides", "statuses"),
    [({"app_env": "test"}, [200, 503]), (EVAL_SETTINGS, [200, 200])],
    ids=["test", "eval"],
)
async def test_eval_mode_bypasses_the_budget(
    db: str, tmp_path: Path, overrides: dict[str, Any], statuses: list[int]
) -> None:
    # A budget of 1 and two different questions (the second cannot be an answer-cache hit).
    settings = capped(db, 1, cache_dir=tmp_path / "cache", **overrides)
    async with app_client(settings) as client:
        sent = [(await ask(client, q)).status_code for q in (QUESTION, OTHER_QUESTION)]

    assert sent == statuses
    if overrides is EVAL_SETTINGS:
        assert usage_row(db) is None  # nothing was reserved and nothing was added
    else:
        assert calls_today(db) == 1


async def test_without_a_cap_nothing_is_counted(db: str) -> None:
    # DAILY_LLM_BUDGET unset (dev and test): no budget is built, so daily_usage is not written.
    async with app_client(make_settings(database_url=db)) as client:
        sent = [(await ask(client, q)).status_code for q in (QUESTION, OTHER_QUESTION)]

    assert sent == [200, 200]
    assert usage_row(db) is None
