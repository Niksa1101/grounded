"""The request deadline of ``POST /v1/ask`` against the real schema (5.08, Tech §6, §13, §18).

A request that outlives ``REQUEST_DEADLINE_S`` ends as a 504 ``deadline_exceeded`` with a
``request_id`` and a ``request_logs`` row that keeps the stage timings reached so far. The
deadlines here are a fraction of a second (long enough for the fast stages to finish first on a
warm pool, which is what each test does before it asks), and the slow work is an awaitable that
never finishes, so it ends by cancellation: those deadlines are the only real waits.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

from grounded.generation.providers.fake import FakeLLMProvider
from grounded.ingest.embed import FakeEmbedder, TaskType, Vector
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
DEADLINE_S = 0.2
TODAY = "(now() AT TIME ZONE 'utc')::date"


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


def good() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[claim])


class HangingProvider(FakeLLMProvider):
    """Never answers: the call waits until it is cancelled, and says so."""

    def __init__(self) -> None:
        super().__init__()
        self.cancelled = False

    async def generate(self, **kwargs: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.started = True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    started = False


class SlowProvider(FakeLLMProvider):
    """Answers, but after ``delay_s`` (real, a few hundredths of a second)."""

    def __init__(self, delay_s: float) -> None:
        super().__init__([good()])
        self._delay_s = delay_s

    async def generate(self, **kwargs: Any) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        await asyncio.sleep(self._delay_s)
        return await super().generate(**kwargs)


class HangingEmbedder(FakeEmbedder):
    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        self.calls.append((list(texts), task_type))
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def rows(url: str) -> list[dict[str, Any]]:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM request_logs ORDER BY created_at, id")
        return cur.fetchall()


def usage(url: str) -> dict[str, Any] | None:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT * FROM daily_usage WHERE day = {TODAY}")
        return cur.fetchone()


async def ask(client: httpx.AsyncClient) -> httpx.Response:
    await client.get("/readyz")  # warm the pool: the first connection is the slowest stage
    return await client.post("/v1/ask", json={"question": QUESTION})


def settings_for(db: str, **overrides: Any) -> Any:
    return make_settings(database_url=db, request_deadline_s=DEADLINE_S, **overrides)


async def test_a_request_that_outlives_the_deadline_is_a_504_with_a_request_id_and_a_row(
    db: str,
) -> None:
    llm = HangingProvider()
    async with app_client(settings_for(db), provider=llm) as client:
        response = await ask(client)

        assert response.status_code == 504
        body = response.json()
        assert body["error"]["code"] == "deadline_exceeded"
        assert body["error"]["retry_after_s"] is None
        assert "Retry-After" not in response.headers
        request_id = UUID(body["request_id"])
        assert "Traceback" not in response.text

        # Not poisoned: the app and its pool still answer.
        assert (await client.get("/readyz")).status_code == 200

    assert llm.started
    assert llm.cancelled  # the call in flight was cancelled, not left running
    [row] = rows(db)
    assert row["id"] == request_id
    # DB.md §4 has no deadline outcome; the row reuses internal_error and says the rest itself.
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "internal_error",
        504,
        "deadline_exceeded",
    )
    assert row["question"] == QUESTION
    assert row["source"] == "api"
    # The stages reached are on the row, the one that was cut off included.
    assert row["latency_embed_ms"] is not None
    assert row["latency_retrieval_ms"] is not None
    assert row["latency_llm_ms"] is not None
    assert row["latency_total_ms"] >= round(DEADLINE_S * 1000) - 5  # the real clock
    assert row["input_tokens"] is None  # no usage was ever reported


async def test_a_deadline_during_the_embed_stage_stops_before_any_llm_call(db: str) -> None:
    embedder, llm = HangingEmbedder(dim=EMBEDDING_DIM), FakeLLMProvider([])
    async with app_client(
        settings_for(db, daily_llm_budget=10), embedder=embedder, provider=llm
    ) as client:
        response = await ask(client)

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "deadline_exceeded"
    assert len(embedder.calls) == 1
    assert llm.calls == []
    assert usage(db) is None  # no reserve before the LLM stage: nothing was spent
    [row] = rows(db)
    assert row["latency_embed_ms"] is not None
    assert row["latency_llm_ms"] is None  # never reached: NULL, not 0
    assert row["error_code"] == "deadline_exceeded"


async def test_a_budget_unit_reserved_before_a_call_that_timed_out_stays_spent(db: str) -> None:
    # The reserve is its own committed statement and the provider may have counted the request
    # (a call that fails after the reserve keeps its unit, Tech §12), so cancelling the call does
    # not give the unit back. No tokens are added: the call never reported any.
    llm = HangingProvider()
    async with app_client(settings_for(db, daily_llm_budget=10), provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 504
    row = usage(db)
    assert row is not None
    assert (row["llm_calls"], row["input_tokens"], row["output_tokens"]) == (1, 0, 0)


async def test_eval_mode_has_no_deadline(db: str, tmp_path: Path) -> None:
    # The same slow reply (60 ms against a 20 ms deadline) is a 200 in eval mode: eval runs have
    # their own, longer call timeout and never hit the deadline (Tech §15.6).
    settings = make_settings(
        database_url=db, cache_dir=tmp_path, request_deadline_s=0.02, **EVAL_SETTINGS
    )
    async with app_client(settings, provider=SlowProvider(0.06)) as client:
        response = await ask(client)

    assert response.status_code == 200
    [row] = rows(db)
    assert (row["outcome"], row["source"]) == ("answered", "eval")


async def test_a_request_inside_the_deadline_is_unaffected(db: str) -> None:
    async with app_client(
        make_settings(database_url=db, request_deadline_s=5.0), provider=FakeLLMProvider([good()])
    ) as client:
        response = await ask(client)
    assert response.status_code == 200
