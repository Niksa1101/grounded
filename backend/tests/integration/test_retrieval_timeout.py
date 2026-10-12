"""The retrieval stage timeout of ``POST /v1/ask`` against the real schema (R.21, Tech §6, D54a).

``RETRIEVAL_TIMEOUT_S`` bounds the pool checkout, the hybrid search and the context build. A search
that outlives it ends as a 504 ``deadline_exceeded`` with outcome ``deadline_exceeded`` (migration
0004) before any LLM call. The slow search is an awaitable that never finishes, so it ends by
cancellation; the request deadline is set far away (30 s), so only the stage timeout can fire.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

import grounded.generation.pipeline
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.retrieval.hybrid import hybrid_search
from grounded.retrieval.types import RetrievedChunk
from grounded.schemas.llm import LLMAnswer, LLMClaim
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import EVAL_SETTINGS, app_client, insert_index_version, make_settings

pytestmark = pytest.mark.integration

QUESTION = "Where does the quokka sleep?"
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


class SearchSpy:
    """Stands in for ``hybrid_search``: hangs, or sleeps and then runs the real search."""

    def __init__(self, *, hang: bool = False, delay_s: float = 0.0) -> None:
        self._hang = hang
        self._delay_s = delay_s

    async def __call__(self, *args: Any, **kwargs: Any) -> list[RetrievedChunk]:
        if self._hang:
            await asyncio.Event().wait()
        await asyncio.sleep(self._delay_s)
        return await hybrid_search(*args, **kwargs)


def rows(url: str) -> list[dict[str, Any]]:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM request_logs ORDER BY created_at, id")
        return cur.fetchall()


def usage(url: str) -> dict[str, Any] | None:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(f"SELECT * FROM daily_usage WHERE day = {TODAY}")
        return cur.fetchone()


async def ask(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post("/v1/ask", json={"question": QUESTION})


async def test_a_search_that_outlives_the_retrieval_timeout_is_a_504_before_any_llm_call(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    search = SearchSpy(hang=True)
    monkeypatch.setattr(grounded.generation.pipeline, "hybrid_search", search)
    llm = FakeLLMProvider([])
    settings = make_settings(
        database_url=db, retrieval_timeout_s=0.05, request_deadline_s=30.0, daily_llm_budget=10
    )
    async with app_client(settings, provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 504
    body = response.json()
    assert body["error"]["code"] == "deadline_exceeded"
    assert "Retry-After" not in response.headers
    [row] = rows(db)
    assert row["id"] == UUID(body["request_id"])
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "deadline_exceeded",
        504,
        "deadline_exceeded",
    )
    assert row["latency_embed_ms"] is not None
    assert row["latency_retrieval_ms"] is not None  # the stage that was cut off is recorded
    assert row["latency_llm_ms"] is None  # never reached: NULL, not 0
    assert llm.calls == []
    assert usage(db) is None  # no budget unit was reserved: the LLM stage never started


async def test_eval_mode_has_no_retrieval_timeout(
    db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A 100 ms search against a 20 ms timeout is a 200 in eval mode (Tech §15.6, PRD D54a).
    monkeypatch.setattr(grounded.generation.pipeline, "hybrid_search", SearchSpy(delay_s=0.1))
    settings = make_settings(
        database_url=db, cache_dir=tmp_path, retrieval_timeout_s=0.02, **EVAL_SETTINGS
    )
    async with app_client(settings, provider=FakeLLMProvider([good()])) as client:
        response = await ask(client)

    assert response.status_code == 200
    [row] = rows(db)
    assert (row["outcome"], row["source"]) == ("answered", "eval")


async def test_a_search_inside_the_retrieval_timeout_is_unaffected(db: str) -> None:
    settings = make_settings(database_url=db, retrieval_timeout_s=30.0)
    async with app_client(settings, provider=FakeLLMProvider([good()])) as client:
        response = await ask(client)

    assert response.status_code == 200
    [row] = rows(db)
    assert row["outcome"] == "answered"
    assert row["latency_retrieval_ms"] is not None
