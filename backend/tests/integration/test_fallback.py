"""The provider router inside ``POST /v1/ask``: fallback, breaker, answer cache, budget, log row and
HTTP errors, end to end (7.04, Tech §10, PRD D57).

A real pgvector database and the real app, with only fakes outside it: a scripted "gemini" and a
scripted "groq" (``FakeLLMProvider`` under those names, so the router's Gemini and Groq rules and
the price list apply), a fake embedder, and two hand-moved clocks. ``Hand.mono`` is the router's
monotonic clock (the breaker's time: nothing here sleeps for a breaker), ``Hand.wall`` its wall
clock, and ``Env.clock`` the pipeline's own clock (stage timing and the time left before the
request deadline). Every request below uses its own question unless it means to repeat one,
because an answer from the primary is cached (Tech §11).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Iterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, LiteralString, NamedTuple
from uuid import uuid4

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row
from pydantic import BaseModel

import grounded.runtime
from grounded.generation.providers.base import GenerationResult, Usage
from grounded.generation.providers.fake import FakeLLMProvider, ScriptStep
from grounded.generation.router import RouterClock
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.ingest.embed import FakeEmbedder, TaskType, Vector
from grounded.main import create_app
from grounded.runtime import ProviderConfigError, open_runtime
from grounded.schemas.llm import LLMAnswer, LLMClaim
from grounded.settings import Settings
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import (
    FakeClock,
    WallClock,
    app_client,
    insert_index_version,
    make_settings,
)

pytestmark = pytest.mark.integration

GEMINI_MODEL = "gemini-3.5-flash-lite"
GROQ_MODEL = "openai/gpt-oss-120b"  # both are priced in pricing.toml
Q1 = "Where does the quokka sleep?"
Q2 = "Where does the wombat sleep?"
Q3 = "What does the numbat eat?"
Q4 = "How does the bilby dig?"
Q5 = "When does the echidna wake?"
GEMINI_USAGE = Usage(input_tokens=1_000, output_tokens=500)
GROQ_USAGE = Usage(input_tokens=2_000, output_tokens=300)


@dataclass
class Hand:
    """The router's clocks, moved by the test."""

    mono: FakeClock
    wall: WallClock

    @property
    def clock(self) -> RouterClock:
        return RouterClock(monotonic=self.mono, wall=self.wall)


class Env(NamedTuple):
    url: str
    clock: FakeClock  # the pipeline's clock
    hand: Hand


@pytest.fixture
def env(test_database_url: str) -> Iterator[Env]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")
            conn.execute("TRUNCATE daily_usage")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    hand = Hand(FakeClock(), WallClock(datetime(2026, 10, 14, 12, 0, tzinfo=UTC)))
    yield Env(test_database_url, FakeClock(), hand)
    wipe()


def good() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[claim])


def limited(retry_after_s: float | None = 30.0) -> ProviderRateLimited:
    return ProviderRateLimited("429", retry_after_s=retry_after_s, is_quota=False)


def gemini(*script: ScriptStep) -> FakeLLMProvider:
    return FakeLLMProvider(script, name="gemini", model=GEMINI_MODEL, usage=GEMINI_USAGE)


def groq(*script: ScriptStep) -> FakeLLMProvider:
    return FakeLLMProvider(script, name="groq", model=GROQ_MODEL, usage=GROQ_USAGE)


class TimedProvider(FakeLLMProvider):
    """Every call takes ``takes_s`` on the pipeline's clock (the failed ones too)."""

    def __init__(self, clock: FakeClock, takes_s: float, script: list[ScriptStep]) -> None:
        super().__init__(script, name="gemini", model=GEMINI_MODEL, usage=GEMINI_USAGE)
        self._clock = clock
        self._takes_s = takes_s

    async def generate[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> GenerationResult[T]:
        self._clock.advance(self._takes_s)
        return await super().generate(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )


def client_for(
    env: Env, primary: FakeLLMProvider, *fallbacks: FakeLLMProvider, **settings: Any
) -> AbstractAsyncContextManager[httpx.AsyncClient]:
    return app_client(
        make_settings(database_url=env.url, **settings),
        provider=primary,
        fallback_providers=fallbacks,
        clock=env.clock,
        router_clock=env.hand.clock,
    )


async def ask(client: httpx.AsyncClient, question: str) -> httpx.Response:
    return await client.post("/v1/ask", json={"question": question})


def table(env: Env, query: LiteralString) -> list[dict[str, Any]]:
    with psycopg.connect(env.url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        return cur.fetchall()


def rows(env: Env) -> list[dict[str, Any]]:
    return table(env, "SELECT * FROM request_logs ORDER BY created_at, id")


def cached(env: Env) -> list[dict[str, Any]]:
    return table(env, "SELECT * FROM answer_cache ORDER BY created_at, cache_key")


def llm_calls(env: Env) -> int:
    [row] = table(
        env, "SELECT llm_calls FROM daily_usage WHERE day = (now() AT TIME ZONE 'utc')::date"
    )
    return int(row["llm_calls"])


def embedding_cost(question: str) -> Decimal:
    """The query embedding is charged as one token per character (Tech §14), $0.15 per 1M."""
    return Decimal(len(question)) * Decimal("0.15") / Decimal(1_000_000)


# --- (a) a 429 on the primary: the fallback answers ----


async def test_a_429_on_the_primary_is_answered_by_the_fallback_and_logged(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    primary, fallback = gemini(limited()), groq(good())
    async with client_for(env, primary, fallback, daily_llm_budget=50) as client:
        response = await ask(client, Q1)

    assert response.status_code == 200
    meta = response.json()["meta"]
    assert (meta["provider"], meta["model"], meta["fallback_used"]) == ("groq", GROQ_MODEL, True)
    assert meta["cache_hit"] is False
    # Only the call that answered billed tokens: Groq's, at Groq's price (the 429 billed nothing).
    assert meta["tokens"] == {"input": 2_000, "output": 300}
    assert meta["shadow_cost_usd"] == pytest.approx(float(Decimal("0.00048") + embedding_cost(Q1)))
    assert (len(primary.calls), len(fallback.calls)) == (1, 1)

    [row] = rows(env)
    assert str(row["id"]) == meta["request_id"]
    assert (row["outcome"], row["http_status"]) == ("answered", 200)
    assert (row["provider"], row["model"], row["fallback_used"]) == ("groq", GROQ_MODEL, True)
    assert (row["input_tokens"], row["output_tokens"]) == (2_000, 300)
    assert row["validation_retries"] == 0
    # The unit of each call was reserved: the 429's and the fallback's.
    assert llm_calls(env) == 2

    # The fallback rate is visible in the logs too: on the summary line, and on the router's lines,
    # which carry the request id. Never the question.
    request_id = meta["request_id"]
    [summary] = [r for r in caplog.records if r.getMessage() == "request completed"]
    assert (summary.__dict__["fallback_used"], summary.__dict__["provider"]) == (True, "groq")
    router_lines = [r for r in caplog.records if r.name == "grounded.generation.router"]
    assert {r.getMessage() for r in router_lines} >= {"breaker opened", "provider attempt failed"}
    assert {r.__dict__["request_id"] for r in router_lines} == {request_id}
    assert Q1 not in caplog.text


# --- (b) the breaker skips the primary until its time is up ----


async def test_the_next_request_skips_an_open_breaker_and_a_trial_closes_it(env: Env) -> None:
    primary = gemini(limited(30.0), good(), good())  # the 429, then the trial, then a plain call
    fallback = groq(good(), good())
    async with client_for(env, primary, fallback, daily_llm_budget=50) as client:
        first = await ask(client, Q1)
        assert first.json()["meta"]["fallback_used"] is True
        assert len(primary.calls) == 1

        # 10 s later the breaker (30 s) is still open: Gemini is not called, Groq answers.
        env.hand.mono.advance(10)
        second = await ask(client, Q2)
        assert second.status_code == 200
        assert second.json()["meta"]["provider"] == "groq"
        assert len(primary.calls) == 1  # zero calls to the primary while the breaker is open
        assert len(fallback.calls) == 2
        # The skip reserved nothing: one unit for the 429, one for each Groq answer.
        assert llm_calls(env) == 3

        # Past the open time one trial goes to Gemini; it answers and closes the breaker.
        env.hand.mono.advance(25)
        third = await ask(client, Q3)
        assert third.status_code == 200
        meta = third.json()["meta"]
        assert (meta["provider"], meta["model"], meta["fallback_used"]) == (
            "gemini",
            GEMINI_MODEL,
            False,
        )
        assert len(primary.calls) == 2

        fourth = await ask(client, Q4)  # closed again: Gemini takes the next request as usual
        assert fourth.json()["meta"]["fallback_used"] is False
        assert len(primary.calls) == 3
        assert len(fallback.calls) == 2

    assert [(r["provider"], r["fallback_used"]) for r in rows(env)] == [
        ("groq", True),
        ("groq", True),
        ("gemini", False),
        ("gemini", False),
    ]
    assert (primary.remaining, fallback.remaining) == (0, 0)
    # The fallback rate of the log: the count the dashboard view of DB.md §8 makes, as plain SQL.
    [rate] = table(
        env,
        "SELECT count(*) FILTER (WHERE fallback_used) AS fallbacks, count(*) AS requests "
        "FROM request_logs",
    )
    assert (rate["fallbacks"], rate["requests"]) == (2, 4)


# --- (c) every breaker open: a 503 with the wait ----


async def test_every_breaker_open_is_a_503_with_the_remaining_wait(env: Env) -> None:
    primary, fallback = gemini(limited(30.0)), groq(limited(45.0))
    async with client_for(env, primary, fallback) as client:
        first = await ask(client, Q1)

        # Both were called and both rate limited: the smallest open time is 30 s.
        assert first.status_code == 503
        assert first.headers["Retry-After"] == "30"
        error = first.json()["error"]
        assert (error["code"], error["retry_after_s"]) == ("provider_unavailable", 30)
        assert "429" not in first.text  # nothing of the provider reaches the client

        env.hand.mono.advance(5)
        second = await ask(client, Q2)  # both skipped: neither is called
        assert second.status_code == 503
        assert second.headers["Retry-After"] == "25"
        assert second.json()["error"]["retry_after_s"] == 25
        assert (len(primary.calls), len(fallback.calls)) == (1, 1)

    first_row, second_row = rows(env)
    for row in (first_row, second_row):
        assert (row["outcome"], row["http_status"], row["error_code"]) == (
            "provider_unavailable",
            503,
            "provider_unavailable",
        )
    # A failed request that tried the fallback says so and names the last provider called; one that
    # called nobody has the primary and no tokens (Tech §14).
    assert (first_row["provider"], first_row["fallback_used"]) == ("groq", True)
    assert (second_row["provider"], second_row["fallback_used"]) == ("gemini", False)
    assert second_row["input_tokens"] is None


async def test_a_503_has_no_retry_after_when_no_wait_is_known(env: Env) -> None:
    # One 5xx each does not open a breaker (three in a row do): nothing says when to retry, so no
    # header is sent and ``retry_after_s`` is null, rather than a number made up.
    primary, fallback = gemini(ProviderUnavailable("down")), groq(ProviderUnavailable("down"))
    async with client_for(env, primary, fallback) as client:
        response = await ask(client, Q1)

    assert response.status_code == 503
    assert "Retry-After" not in response.headers
    assert response.json()["error"]["retry_after_s"] is None
    [row] = rows(env)
    assert (row["outcome"], row["provider"], row["fallback_used"]) == (
        "provider_unavailable",
        "groq",
        True,
    )


# --- (d) bad output twice: one retry, then the fallback ----


async def test_bad_output_twice_on_the_primary_costs_two_calls_and_then_falls_back(
    env: Env,
) -> None:
    primary, fallback = gemini("{", "{"), groq(good())  # a third scripted Gemini call would fail
    async with client_for(env, primary, fallback, daily_llm_budget=50) as client:
        response = await ask(client, Q1)

    assert response.status_code == 200
    meta = response.json()["meta"]
    assert (meta["provider"], meta["fallback_used"]) == ("groq", True)
    # No double retry: the primary was called exactly twice (the call and its one retry).
    assert len(primary.calls) == 2
    first, retry = primary.calls
    assert retry.user.startswith(first.user)
    assert "Your previous output was invalid because" in retry.user
    assert len(fallback.calls) == 1
    assert fallback.calls[0].user == first.user  # a fallback starts from the original message

    [row] = rows(env)
    assert (row["validation_retries"], row["provider"], row["fallback_used"]) == (1, "groq", True)
    # The usage of all three calls: two billed bad outputs on Gemini and the Groq answer.
    assert (row["input_tokens"], row["output_tokens"]) == (4_000, 1_300)
    assert llm_calls(env) == 3
    # Each at the price of the model that billed it: 2 x (1,000 in, 500 out) at Gemini's list price
    # and (2,000 in, 300 out) at Groq's, plus the embedding.
    gemini_cost = Decimal(2) * (Decimal("0.0003") + Decimal("0.00125"))
    groq_cost = Decimal("0.0003") + Decimal("0.00018")
    assert row["shadow_cost_usd"] == (gemini_cost + groq_cost + embedding_cost(Q1)).quantize(
        Decimal("0.00000001")
    )
    assert meta["tokens"] == {"input": 4_000, "output": 1_300}


async def test_bad_output_is_not_a_breaker_event(env: Env) -> None:
    # The provider answered, so it is alive: the next request calls it again (no skip).
    primary, fallback = gemini("{", "{", good()), groq(good())
    async with client_for(env, primary, fallback) as client:
        first = await ask(client, Q1)
        second = await ask(client, Q2)

    assert first.json()["meta"]["fallback_used"] is True
    assert second.json()["meta"]["fallback_used"] is False
    assert len(primary.calls) == 3


async def test_bad_output_on_every_provider_stays_a_502(env: Env) -> None:
    primary, fallback = gemini("{", "{"), groq("{", "{")
    async with client_for(env, primary, fallback) as client:
        response = await ask(client, Q1)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "validation_failed"
    [row] = rows(env)
    assert (row["outcome"], row["validation_retries"], row["fallback_used"]) == (
        "validation_failed",
        2,
        True,
    )
    assert (row["input_tokens"], row["output_tokens"]) == (6_000, 1_600)


# --- (e) the answer cache ----


async def test_a_fallback_answer_is_not_cached_and_a_cached_answer_is_served_while_open(
    env: Env,
) -> None:
    primary = gemini(good(), limited(30.0))  # Q1 is answered, then the 429 on Q2
    fallback = groq(good(), good())
    async with client_for(env, primary, fallback) as client:
        first = await ask(client, Q1)
        assert first.json()["meta"]["fallback_used"] is False
        assert len(cached(env)) == 1  # a primary answer is stored

        second = await ask(client, Q2)  # falls back: not stored
        assert second.json()["meta"]["fallback_used"] is True
        assert len(cached(env)) == 1

        again = await ask(client, Q2)  # the breaker is open, so Groq again, and again not stored
        assert again.json()["meta"]["fallback_used"] is True
        assert again.json()["meta"]["cache_hit"] is False
        assert len(cached(env)) == 1
        assert len(fallback.calls) == 2  # a repeated question pays a fallback call again

        # The cached answer is still served while the primary's breaker is open: no call at all.
        hit = await ask(client, Q1)
        assert hit.status_code == 200
        meta = hit.json()["meta"]
        assert (meta["cache_hit"], meta["fallback_used"], meta["provider"]) == (
            True,
            False,
            "gemini",
        )
        assert (len(primary.calls), len(fallback.calls)) == (2, 2)

    [stored] = cached(env)
    assert stored["generator_model"] == GEMINI_MODEL  # the key names the primary, and only it
    assert stored["hit_count"] == 1


# --- the budget: a unit for every call that is made, none for one that is not ----


async def test_the_fallback_reserves_budget_and_stops_when_none_is_left(env: Env) -> None:
    # A budget of 1: the 429'd call used it, so the fallback's reserve finds none and Groq is not
    # called. The reserve comes before the call (5.05), for the fallback as for the first call.
    primary, fallback = gemini(limited()), groq(good())
    async with client_for(env, primary, fallback, daily_llm_budget=1) as client:
        response = await ask(client, Q1)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "budget_exhausted"
    assert fallback.calls == []
    [row] = rows(env)
    assert row["outcome"] == "budget_exhausted"
    assert llm_calls(env) == 1


# --- (f) the request deadline ----


async def test_a_retry_that_does_not_fit_before_the_deadline_is_a_504(env: Env) -> None:
    # 20 s deadline, 12 s per call, and the first call "takes" 9 s: 11 s are left, so the retry
    # (12 s) is not started. Not a provider failure and not a 503: the request ran out of time.
    primary = TimedProvider(env.clock, 9.0, ["{", good()])
    fallback = groq(good())
    async with client_for(
        env, primary, fallback, request_deadline_s=20.0, daily_llm_budget=50
    ) as client:
        response = await ask(client, Q1)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "deadline_exceeded"
    assert "Retry-After" not in response.headers
    assert (len(primary.calls), fallback.calls) == (1, [])  # the first entry's refusal ends it all
    [row] = rows(env)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "deadline_exceeded",
        504,
        "deadline_exceeded",
    )
    assert row["validation_retries"] == 0  # the retry was never made
    assert (row["input_tokens"], row["output_tokens"]) == (1_000, 500)  # the first call's usage
    assert row["latency_llm_ms"] == 9_000
    assert llm_calls(env) == 1  # the refused attempt reserved nothing


async def test_a_deadline_shorter_than_one_call_is_a_504_before_any_call(env: Env) -> None:
    primary, fallback = gemini(good()), groq(good())
    async with client_for(
        env, primary, fallback, request_deadline_s=5.0, daily_llm_budget=50
    ) as client:
        response = await ask(client, Q1)

    assert response.status_code == 504
    assert (primary.calls, fallback.calls) == ([], [])
    [row] = rows(env)
    assert row["outcome"] == "deadline_exceeded"
    assert row["latency_llm_ms"] is not None  # the stage was entered
    assert row["input_tokens"] is None
    # No budget unit for a call that was not made: the day's row was never even created.
    assert table(env, "SELECT * FROM daily_usage") == []


class TimedEmbedder(FakeEmbedder):
    """The query embedding takes ``takes_s`` on the pipeline's clock: it stands for everything a
    request does before the LLM stage, which is otherwise 0 s on the fake clock."""

    def __init__(self, clock: FakeClock, takes_s: float, *, dim: int) -> None:
        super().__init__(dim=dim)
        self._clock = clock
        self._takes_s = takes_s

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        self._clock.advance(self._takes_s)
        return await super().embed(texts, task_type)


def timed_client(
    env: Env, before_llm_s: float, primary: FakeLLMProvider, fallback: FakeLLMProvider
) -> AbstractAsyncContextManager[httpx.AsyncClient]:
    # The two numbers the arithmetic below depends on, pinned (they are the defaults, D53).
    settings = make_settings(database_url=env.url, request_deadline_s=25.0, llm_timeout_s=12.0)
    return app_client(
        settings,
        embedder=TimedEmbedder(env.clock, before_llm_s, dim=settings.embedding_dim),
        provider=primary,
        fallback_providers=[fallback],
        clock=env.clock,
        router_clock=env.hand.clock,
    )


def timeouts(env: Env, count: int) -> TimedProvider:
    """A Gemini whose every call runs into its whole 12 s timeout."""
    return TimedProvider(env.clock, 12.0, [ProviderTimeout("12 s") for _ in range(count)])


async def test_a_primary_timeout_leaves_no_time_for_the_fallback_and_is_a_504(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    # Pinned as it is today (Tech §10, "The deadline rule"). A timed-out primary used its whole
    # LLM_TIMEOUT_S, so the fallback starts only if the request reached the LLM stage within
    # REQUEST_DEADLINE_S - 2 x LLM_TIMEOUT_S, 1 s with the defaults. After 1.5 s, 11.5 s of 25 are
    # left: Groq is not called and the request is the router's 504, not a fallback. The timeout
    # still counts toward Gemini's breaker: the third one opens it, and the next request goes
    # straight to Groq without calling Gemini.
    caplog.set_level(logging.INFO)
    primary, fallback = timeouts(env, 3), groq(good())
    async with timed_client(env, 1.5, primary, fallback) as client:
        refused = [await ask(client, question) for question in (Q1, Q2, Q3)]
        answered = await ask(client, Q4)

    assert [response.status_code for response in refused] == [504, 504, 504]
    assert {response.json()["error"]["code"] for response in refused} == {"deadline_exceeded"}
    refusals = [
        r
        for r in caplog.records
        if r.getMessage() == "router refused an attempt, too little time left"
    ]
    assert [r.__dict__["time_left_s"] for r in refusals] == [pytest.approx(11.5)] * 3
    assert answered.status_code == 200
    meta = answered.json()["meta"]
    assert (meta["provider"], meta["fallback_used"]) == ("groq", True)
    assert (len(primary.calls), len(fallback.calls)) == (3, 1)  # the fourth skipped Gemini

    *failed, last = rows(env)
    assert [(r["outcome"], r["provider"], r["fallback_used"]) for r in failed] == [
        ("deadline_exceeded", "gemini", False)
    ] * 3
    assert {r["latency_llm_ms"] for r in failed} == {12_000}  # the timeout, then no second call
    assert (last["outcome"], last["provider"], last["fallback_used"]) == ("answered", "groq", True)


async def test_a_primary_timeout_falls_back_when_the_llm_stage_starts_within_1_s(
    env: Env,
) -> None:
    # The other side of the same arithmetic: after 0.5 s, 12.5 s are left, a whole Groq attempt.
    primary, fallback = timeouts(env, 1), groq(good())
    async with timed_client(env, 0.5, primary, fallback) as client:
        response = await ask(client, Q1)

    assert response.status_code == 200
    meta = response.json()["meta"]
    assert (meta["provider"], meta["fallback_used"]) == ("groq", True)
    assert (len(primary.calls), len(fallback.calls)) == (1, 1)


# --- the primary alone, and the rest of the contract ----


async def test_without_a_fallback_a_429_is_a_503_with_the_breaker_wait(env: Env) -> None:
    primary = gemini(limited(30.0), good())
    async with client_for(env, primary) as client:
        first = await ask(client, Q1)
        second = await ask(client, Q2)
        env.hand.mono.advance(31)
        third = await ask(client, Q3)

    assert (first.status_code, second.status_code, third.status_code) == (503, 503, 200)
    assert first.headers["Retry-After"] == "30"
    assert len(primary.calls) == 2  # the second request was skipped, the third was the trial
    assert third.json()["meta"]["fallback_used"] is False
    assert [r["fallback_used"] for r in rows(env)] == [False, False, False]


async def test_a_rejected_request_on_the_primary_is_not_hidden_by_the_fallback_row(
    env: Env,
) -> None:
    # A bad key on the primary fails over (no breaker), and the answer says so.
    primary = gemini(ProviderRequestRejected("secret-detail", status_code=403))
    fallback = groq(good())
    async with client_for(env, primary, fallback) as client:
        response = await ask(client, Q1)

    assert response.status_code == 200
    assert response.json()["meta"]["fallback_used"] is True
    assert "secret-detail" not in response.text


async def test_bad_output_that_is_not_retryable_goes_straight_to_the_fallback(env: Env) -> None:
    blocked = ProviderBadOutput("Gemini returned no candidate", retryable=False)
    primary, fallback = gemini(blocked), groq(good())
    async with client_for(env, primary, fallback) as client:
        response = await ask(client, Q1)

    assert response.status_code == 200
    assert response.json()["meta"]["fallback_used"] is True
    assert len(primary.calls) == 1  # a blocked prompt is not retried
    [row] = rows(env)
    assert row["validation_retries"] == 0


# --- (g) startup: the first entry must be buildable, a later one is skipped with a warning ----


@asynccontextmanager
async def started(settings: Settings) -> AsyncGenerator[httpx.AsyncClient]:
    """The app of ``settings`` with its providers built from them (no provider injected), plus a
    client. ``build_provider`` is patched by the caller where a real adapter would be made."""
    app = create_app(settings, embedder=FakeEmbedder(dim=settings.embedding_dim))
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 123))
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as http:
            yield http


async def test_an_unbuildable_fallback_is_skipped_with_a_warning_and_the_primary_answers(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    primary = gemini(good())
    monkeypatch.setattr(grounded.runtime, "build_provider", lambda _settings: primary)
    settings = make_settings(
        database_url=env.url,
        allow_direct_api=True,
        generator_providers=["gemini", "groq"],
        gemini_api_key="gemini-secret-key",
        groq_api_key="groq-secret-key",  # a key but no GROQ_MODEL: the Groq entry cannot be built
        groq_model=None,
    )
    caplog.set_level(logging.INFO)
    async with started(settings) as client:
        response = await ask(client, Q1)

    assert response.status_code == 200
    assert response.json()["meta"]["provider"] == "gemini"
    [skipped] = [r for r in caplog.records if r.getMessage().startswith("fallback provider")]
    assert skipped.levelno == logging.WARNING
    assert skipped.__dict__["provider"] == "groq"
    assert "GROQ_MODEL" in skipped.__dict__["reason"]
    assert "secret-key" not in caplog.text  # the name of the variable, never a value
    # The startup line says which providers the router runs on: only the primary, here.
    [ready] = [r for r in caplog.records if r.getMessage() == "generator providers ready"]
    assert ready.__dict__["providers"] == ["gemini"]


async def test_a_missing_groq_key_is_named_in_the_warning(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(grounded.runtime, "build_provider", lambda _settings: gemini(good()))
    settings = make_settings(
        database_url=env.url, generator_providers=["gemini", "groq"], groq_model=GROQ_MODEL
    )
    async with started(settings):
        pass

    [skipped] = [r for r in caplog.records if r.getMessage().startswith("fallback provider")]
    assert "GROQ_API_KEY" in skipped.__dict__["reason"]


async def test_groq_as_the_first_entry_stops_startup(env: Env) -> None:
    settings = make_settings(
        database_url=env.url,
        generator_providers=["groq", "gemini"],
        groq_api_key="k-123",
        groq_model=GROQ_MODEL,
    )
    with pytest.raises(ProviderConfigError, match="cannot be the first entry"):
        async with started(settings):
            pass


async def test_an_unbuildable_first_entry_stops_startup(env: Env) -> None:
    settings = make_settings(database_url=env.url, generator_providers=["gemini", "groq"])
    with pytest.raises(ProviderConfigError, match="GEMINI_API_KEY"):
        async with started(settings):
            pass


def forbid_fallback_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("no fallback provider may be built here")

    monkeypatch.setattr(grounded.runtime, "build_fallback_providers", forbidden)


async def test_primary_only_builds_no_fallback_from_the_settings(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The CLI (and eval mode) run the first entry alone, even when the settings name a Groq key.
    monkeypatch.setattr(grounded.runtime, "build_provider", lambda _settings: gemini(good()))
    forbid_fallback_builds(monkeypatch)
    settings = make_settings(
        database_url=env.url,
        generator_providers=["gemini", "groq"],
        groq_api_key="k-123",
        groq_model=GROQ_MODEL,
    )

    async with open_runtime(
        settings, embedder=FakeEmbedder(dim=settings.embedding_dim), primary_only=True
    ) as runtime:
        response = await runtime.pipeline.ask(Q1, uuid4())

    assert (response.meta.provider, response.meta.fallback_used) == ("gemini", False)


async def test_an_injected_provider_stands_for_the_whole_list(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A test that injects the primary never gets a real Groq built next to it.
    forbid_fallback_builds(monkeypatch)
    settings = make_settings(
        database_url=env.url,
        generator_providers=["gemini", "groq"],
        groq_api_key="k-123",
        groq_model=GROQ_MODEL,
    )

    async with open_runtime(
        settings, embedder=FakeEmbedder(dim=settings.embedding_dim), provider=gemini(good())
    ) as runtime:
        response = await runtime.pipeline.ask(Q1, uuid4())

    assert response.meta.provider == "gemini"
