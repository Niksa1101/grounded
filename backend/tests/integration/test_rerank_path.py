"""Rerank in the request path of ``POST /v1/ask`` against the real schema (6.03, Tech §6, §8).

Real pgvector index (the 13-chunk hybrid corpus: a question retrieves all of them, so the fake
reranker always gets 13 candidates), fake embedder, scripted fake LLM and ``FakeReranker``. No test
reaches Cohere. The shape of each test is: what the model was given (the ``<source>`` URLs in
prompt order, which is the context order), what the response and the ``request_logs`` row say, and
what the daily cap counted.

``rerank-v4.0-pro`` is the model of these settings because it is priced in ``pricing.toml`` (the
pipeline refuses an unpriced one); the fake reranker's own model name is not used.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Iterator, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.budget import RerankCap
from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.infra.timing import StageTimer
from grounded.ingest.embed import FakeEmbedder
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.rerank import CachedReranker, FakeReranker, Hang, PacedReranker
from grounded.runtime import ProviderConfigError, open_runtime
from grounded.schemas.api import AskResponse
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
THIRD_QUESTION = "Where does the quokka nap?"
MODEL = "rerank-v4.0-pro"
CANDIDATES = len(CORPUS)  # K_FUSED (40) is larger than the corpus: every chunk is a candidate
PART_URL = "https://fastapi.tiangolo.com/hybrid/#part-"
TODAY = "(now() AT TIME ZONE 'utc')::date"
# The question's 28 characters at the embedding price, plus one rerank call at $2.50 per 1K.
EMBEDDING_COST = Decimal("0.00000420")
RERANK_CALL_COST = Decimal("0.00250000")


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


def rerank_settings(db: str, **overrides: Any) -> Any:
    return make_settings(
        **{
            "database_url": db,
            "rerank_provider": "cohere",
            "rerank_model": MODEL,
            "rerank_timeout_s": 5.0,
            **overrides,
        }
    )


def good() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[claim])


def favoring(boost: dict[int, float], *, base: float = 0.1) -> list[float]:
    """One score per candidate (RRF order): ``base`` for all but the boosted indexes."""
    return [boost.get(i, base) for i in range(CANDIDATES)]


def urls_in(user_prompt: str) -> list[str]:
    """The URLs of the ``<source>`` blocks, in prompt order (= label order = context order)."""
    return re.findall(r'<source id="c\d" section="[^"]*" url="([^"]*)">', user_prompt)


def url_of(document: str) -> str:
    """The chunk URL behind a document the reranker was given ("Hybrid > Part N" breadcrumb)."""
    match = re.search(r"Hybrid > Part (\d+)", document)
    assert match is not None, document
    return f"{PART_URL}{match.group(1)}"


def rows(url: str) -> list[dict[str, Any]]:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM request_logs ORDER BY created_at, id")
        return cur.fetchall()


def row_of(url: str, response: httpx.Response) -> dict[str, Any]:
    [row] = [r for r in rows(url) if str(r["id"]) == response.json()["meta"]["request_id"]]
    return row


def rerank_calls_today(url: str) -> int | None:
    """Today's ``rerank_calls``, or ``None`` when ``daily_usage`` has no row for today."""
    with psycopg.connect(url) as conn:
        found = conn.execute(f"SELECT rerank_calls FROM daily_usage WHERE day = {TODAY}")
        row = found.fetchone()
    return None if row is None else int(row[0])


def set_rerank_calls(url: str, calls: int) -> None:
    with psycopg.connect(url) as conn:
        conn.execute(
            f"INSERT INTO daily_usage (day, rerank_calls) VALUES ({TODAY}, %s) "
            "ON CONFLICT (day) DO UPDATE SET rerank_calls = EXCLUDED.rerank_calls",
            (calls,),
        )


def cached_answers(url: str) -> int:
    with psycopg.connect(url) as conn:
        row = conn.execute("SELECT count(*) FROM answer_cache").fetchone()
    assert row is not None
    return int(row[0])


def clear_answer_cache(url: str) -> None:
    with psycopg.connect(url) as conn:
        conn.execute("TRUNCATE answer_cache")


async def ask(client: httpx.AsyncClient, question: str = QUESTION) -> httpx.Response:
    return await client.post("/v1/ask", json={"question": question})


async def plain_prompt(db: str) -> str:
    """The user prompt of the same question with rerank off: the RRF context, as before 6.03."""
    llm = FakeLLMProvider([good()])
    async with app_client(make_settings(database_url=db), provider=llm) as client:
        assert (await ask(client)).status_code == 200
    clear_answer_cache(db)  # the plain answer must not be served to the requests under test
    return llm.calls[0].user


# --- success -----------------------------------------------------------------------------------


async def test_a_reranked_request_builds_its_context_from_the_reranked_order(db: str) -> None:
    settings = rerank_settings(db)
    reranker, llm = FakeReranker([favoring({12: 0.99, 7: 0.8})]), FakeLLMProvider([good()])
    async with app_client(settings, provider=llm, reranker=reranker) as client:
        response = await ask(client)

    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    [call] = reranker.calls
    assert call.query == QUESTION
    assert len(call.documents) == CANDIDATES
    candidates = [url_of(document) for document in call.documents]  # RRF order
    # The best two by score, then the 0.1 ties in their RRF order (the first K_CONTEXT = 5 in all).
    expected = [candidates[12], candidates[7], candidates[0], candidates[1], candidates[2]]
    assert urls_in(llm.calls[0].user) == expected
    assert [str(c.url) for c in body.citations] == [expected[0]]  # c1 is the top reranked chunk

    meta = body.meta
    assert meta.rerank_used is True
    assert "rerank" in meta.latency_ms
    assert (
        meta.retrieval_config_hash == RetrievalConfig.from_settings(settings, "hybrid").config_hash
    )
    # The rerank component of the claim that cites c1 is c1's score (the weight stays 0, below).
    assert body.claims[0].confidence_components["rerank"] == pytest.approx(0.99)

    row = row_of(db, response)
    assert (row["rerank_used"], row["rerank_error"]) == (True, None)
    assert row["latency_rerank_ms"] is not None
    assert row["retrieval_config_hash"] == meta.retrieval_config_hash
    assert row["shadow_cost_usd"] == EMBEDDING_COST + RERANK_CALL_COST
    assert meta.shadow_cost_usd == pytest.approx(float(EMBEDDING_COST + RERANK_CALL_COST))
    assert rerank_calls_today(db) == 1


async def test_the_rerank_flag_changes_the_config_hash_and_off_leaves_it_alone(db: str) -> None:
    off_settings, on_settings = make_settings(database_url=db), rerank_settings(db)
    async with app_client(off_settings, provider=FakeLLMProvider([good()])) as client:
        off = await ask(client)
    async with app_client(
        on_settings, provider=FakeLLMProvider([good()]), reranker=FakeReranker()
    ) as client:
        on = await ask(client)

    off_hash = off.json()["meta"]["retrieval_config_hash"]
    on_hash = on.json()["meta"]["retrieval_config_hash"]
    assert off_hash == RetrievalConfig.from_settings(off_settings, "hybrid").config_hash
    assert on_hash == RetrievalConfig.from_settings(on_settings, "hybrid").config_hash
    assert on_hash != off_hash


# --- degradation -------------------------------------------------------------------------------

ERRORS: list[tuple[ProviderError, str]] = [
    (ProviderRateLimited("429", retry_after_s=None, is_quota=False), "rate_limited"),
    (ProviderTimeout("slow"), "timeout"),
    (ProviderUnavailable("503"), "unavailable"),
    (ProviderRequestRejected("401", status_code=401), "rejected"),
    (ProviderBadOutput("not the documented shape"), "bad_output"),
]


@pytest.mark.parametrize(("error", "code"), ERRORS, ids=[code for _, code in ERRORS])
async def test_a_failed_rerank_degrades_to_the_rrf_context_and_the_request_succeeds(
    db: str, error: ProviderError, code: str
) -> None:
    baseline = await plain_prompt(db)
    reranker, llm = FakeReranker([error]), FakeLLMProvider([good()])
    async with app_client(rerank_settings(db), provider=llm, reranker=reranker) as client:
        response = await ask(client)

    assert response.status_code == 200
    assert len(reranker.calls) == 1
    assert llm.calls[0].user == baseline  # exactly the context of a request that never reranked
    body = AskResponse.model_validate(response.json())
    assert body.status == "answered"
    assert body.meta.rerank_used is False
    assert body.claims[0].confidence_components["rerank"] == 0.0  # no chunk has a rerank score
    row = row_of(db, response)
    assert (row["rerank_used"], row["rerank_error"], row["outcome"]) == (False, code, "answered")
    assert row["latency_rerank_ms"] is not None  # the stage ran, and failed
    # An attempted call is charged: the cap counted it and the shadow cost has it.
    assert rerank_calls_today(db) == 1
    assert row["shadow_cost_usd"] == EMBEDDING_COST + RERANK_CALL_COST


async def test_a_hung_reranker_degrades_at_the_stage_timeout(db: str) -> None:
    baseline = await plain_prompt(db)
    reranker, llm = FakeReranker([Hang()]), FakeLLMProvider([good()])
    settings = rerank_settings(db, rerank_timeout_s=0.05)
    async with app_client(settings, provider=llm, reranker=reranker) as client:
        response = await ask(client)

    assert response.status_code == 200
    assert llm.calls[0].user == baseline
    row = row_of(db, response)
    assert (row["rerank_used"], row["rerank_error"]) == (False, "timeout")
    assert row["latency_rerank_ms"] is not None  # the time spent waiting is on the row
    assert rerank_calls_today(db) == 1


async def test_a_cancelled_rerank_is_not_swallowed_as_a_degradation(db: str) -> None:
    class Cancelled(FakeReranker):
        async def _scores(self, query: str, chunks: Sequence[Any]) -> list[float]:
            raise asyncio.CancelledError

    llm = FakeLLMProvider([good()])
    trace = RequestTrace(request_id=uuid4(), timer=StageTimer(time.perf_counter))
    async with open_runtime(
        rerank_settings(db),
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=llm,
        reranker=Cancelled(),
    ) as runtime:
        with pytest.raises(asyncio.CancelledError):
            await runtime.pipeline.ask(QUESTION, trace.request_id, trace=trace)

    assert llm.calls == []  # the request did not go on without a ranking
    assert (trace.rerank_used, trace.rerank_error) == (False, None)  # and recorded no "degradation"
    assert trace.rerank_calls == 1  # the attempt is counted: it was started (and the unit spent)
    assert rerank_calls_today(db) == 1
    assert cached_answers(db) == 0


# --- the daily cap -----------------------------------------------------------------------------


async def test_the_cap_reached_makes_no_call_and_uses_the_rrf_order(db: str) -> None:
    baseline = await plain_prompt(db)
    set_rerank_calls(db, 2)
    reranker, llm = FakeReranker(), FakeLLMProvider([good()])
    settings = rerank_settings(db, rerank_daily_cap=2)
    async with app_client(settings, provider=llm, reranker=reranker) as client:
        response = await ask(client)

    assert response.status_code == 200
    assert reranker.calls == []  # no call, and no spend of a scarce quota
    assert llm.calls[0].user == baseline
    assert response.json()["meta"]["rerank_used"] is False
    assert "rerank" not in response.json()["meta"]["latency_ms"]  # the stage did not run
    row = row_of(db, response)
    assert (row["rerank_used"], row["rerank_error"]) == (False, "cap_reached")
    assert row["latency_rerank_ms"] is None  # NULL, not 0: the reranker was never timed
    assert row["shadow_cost_usd"] == EMBEDDING_COST  # a call that was not made is not charged
    assert rerank_calls_today(db) == 2  # a refused reserve does not count itself


async def test_a_cap_of_zero_never_calls_and_never_touches_daily_usage(db: str) -> None:
    reranker, llm = FakeReranker(), FakeLLMProvider([good()])
    async with app_client(
        rerank_settings(db, rerank_daily_cap=0), provider=llm, reranker=reranker
    ) as client:
        response = await ask(client)

    assert response.status_code == 200
    assert reranker.calls == []
    assert row_of(db, response)["rerank_error"] == "cap_reached"
    assert rerank_calls_today(db) is None


async def test_every_attempted_call_counts_against_the_cap_and_the_last_unit_is_the_last(
    db: str,
) -> None:
    # Cap 2: a successful call and a failed one use the two units; the third request finds none.
    reranker = FakeReranker([favoring({12: 0.9}), ProviderUnavailable("503")])
    llm = FakeLLMProvider([good(), good(), good()])
    async with app_client(
        rerank_settings(db, rerank_daily_cap=2), provider=llm, reranker=reranker
    ) as client:
        first = await ask(client, QUESTION)
        second = await ask(client, OTHER_QUESTION)
        third = await ask(client, THIRD_QUESTION)

    assert [row_of(db, r)["rerank_error"] for r in (first, second, third)] == [
        None,
        "unavailable",
        "cap_reached",
    ]
    assert len(reranker.calls) == 2
    assert rerank_calls_today(db) == 2


async def test_parallel_requests_never_make_more_calls_than_the_cap(db: str) -> None:
    questions = [f"Where does quokka number {n} sleep?" for n in range(6)]
    reranker, llm = FakeReranker(), FakeLLMProvider([good() for _ in questions])
    async with app_client(
        rerank_settings(db, rerank_daily_cap=3), provider=llm, reranker=reranker
    ) as client:
        responses = await asyncio.gather(*(ask(client, q) for q in questions))

    assert [r.status_code for r in responses] == [200] * 6
    assert len(reranker.calls) == 3
    assert rerank_calls_today(db) == 3
    outcomes = sorted(str(row["rerank_error"]) for row in rows(db))
    assert outcomes == ["None"] * 3 + ["cap_reached"] * 3


async def test_a_failing_reserve_degrades_without_a_call(
    db: str, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The unit cannot be counted, so no call is made (fail-closed): the quota is the scarce thing.
    async def broken(self: RerankCap) -> None:
        raise psycopg.OperationalError(f"connection lost while reading {QUESTION}")

    monkeypatch.setattr(RerankCap, "reserve", broken)
    baseline = await plain_prompt(db)
    reranker, llm = FakeReranker(), FakeLLMProvider([good()])
    with caplog.at_level("WARNING", logger="grounded.generation.pipeline"):
        async with app_client(rerank_settings(db), provider=llm, reranker=reranker) as client:
            response = await ask(client)

    assert response.status_code == 200
    assert reranker.calls == []
    assert llm.calls[0].user == baseline
    assert row_of(db, response)["rerank_error"] == "unavailable"
    [record] = [r for r in caplog.records if r.getMessage().startswith("rerank degraded")]
    assert (record.rerank_error, record.error) == ("unavailable", "OperationalError")  # type: ignore[attr-defined]
    assert QUESTION not in str(record.__dict__)  # the database's message is not logged


# --- logging -----------------------------------------------------------------------------------


async def test_a_degradation_is_logged_with_the_code_and_the_type_only(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "sk-SECRET-body-from-the-provider"
    reranker = FakeReranker([ProviderUnavailable(f"Cohere 503: {secret}")])
    with caplog.at_level("INFO"):
        async with app_client(
            rerank_settings(db), provider=FakeLLMProvider([good()]), reranker=reranker
        ) as client:
            response = await ask(client)

    request_id = response.json()["meta"]["request_id"]
    [warning] = [r for r in caplog.records if r.getMessage().startswith("rerank degraded")]
    assert warning.levelname == "WARNING"
    assert (warning.request_id, warning.rerank_error, warning.error) == (  # type: ignore[attr-defined]
        request_id,
        "unavailable",
        "ProviderUnavailable",
    )
    assert warning.status_code is None  # type: ignore[attr-defined]  # only a rejection has one
    everything = "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert QUESTION not in everything  # AGENTS.md §6.13
    assert secret not in everything  # nor the provider's text
    [summary] = [r for r in caplog.records if r.getMessage() == "request completed"]
    assert (summary.rerank_used, summary.rerank_error) == (False, "unavailable")  # type: ignore[attr-defined]


async def test_a_rejected_rerank_logs_its_http_status(
    db: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The runbook reads the status to tell a bad key (401) from a model the key cannot use (403,
    # 404): a number, never the provider's text.
    secret = "sk-SECRET-body-from-the-provider"
    rejected = ProviderRequestRejected(f"Cohere rerank 404: {secret}", status_code=404)
    with caplog.at_level("INFO"):
        async with app_client(
            rerank_settings(db),
            provider=FakeLLMProvider([good()]),
            reranker=FakeReranker([rejected]),
        ) as client:
            await ask(client)

    [warning] = [r for r in caplog.records if r.getMessage().startswith("rerank degraded")]
    assert (warning.rerank_error, warning.error, warning.status_code) == (  # type: ignore[attr-defined]
        "rejected",
        "ProviderRequestRejected",
        404,
    )
    everything = "\n".join(f"{r.getMessage()} {r.__dict__}" for r in caplog.records)
    assert QUESTION not in everything  # AGENTS.md §6.13
    assert secret not in everything


# --- the caches --------------------------------------------------------------------------------


async def test_an_answer_cache_hit_skips_rerank_entirely(db: str) -> None:
    reranker, llm = FakeReranker([favoring({12: 0.99})]), FakeLLMProvider([good()])
    async with app_client(rerank_settings(db), provider=llm, reranker=reranker) as client:
        miss = await ask(client)
        hit = await ask(client, "  where DOES the quokka sleep ?? ")  # same key once normalized

    assert miss.json()["meta"]["cache_hit"] is False
    meta = hit.json()["meta"]
    assert meta["cache_hit"] is True
    assert meta["rerank_used"] is False  # this request did no rerank; the stored answer had one
    assert len(reranker.calls) == 1  # zero calls for the hit
    assert len(llm.calls) == 1
    assert rerank_calls_today(db) == 1  # and no unit of the cap
    row = row_of(db, hit)
    assert (row["cache_hit"], row["rerank_used"], row["rerank_error"]) == (True, False, None)
    assert row["latency_rerank_ms"] is None
    # What the hit served is the reranked answer: its claim carries the rerank score of c1.
    assert hit.json()["claims"][0]["confidence_components"]["rerank"] == pytest.approx(0.99)


async def test_a_degraded_answer_is_not_stored_in_the_answer_cache(db: str) -> None:
    reranker = FakeReranker([ProviderUnavailable("503"), favoring({12: 0.99})])
    llm = FakeLLMProvider([good(), good()])
    async with app_client(rerank_settings(db), provider=llm, reranker=reranker) as client:
        degraded = await ask(client)
        assert cached_answers(db) == 0  # the RRF-order answer is not filed under the rerank key
        recovered = await ask(client)

    assert degraded.json()["meta"]["rerank_used"] is False
    assert recovered.json()["meta"]["cache_hit"] is False  # a miss, not the degraded answer
    assert recovered.json()["meta"]["rerank_used"] is True
    assert len(llm.calls) == 2
    assert cached_answers(db) == 1  # the reranked one is stored


async def test_a_reranked_answer_and_a_plain_one_never_share_a_cache_entry(db: str) -> None:
    on_llm, off_llm = FakeLLMProvider([good()]), FakeLLMProvider([good()])
    async with app_client(rerank_settings(db), provider=on_llm, reranker=FakeReranker()) as client:
        reranked = await ask(client)
    async with app_client(make_settings(database_url=db), provider=off_llm) as client:
        plain = await ask(client)  # the same question, same index, rerank off

    assert reranked.json()["meta"]["cache_hit"] is False
    assert plain.json()["meta"]["cache_hit"] is False
    assert len(off_llm.calls) == 1  # it did not get the reranked entry
    with psycopg.connect(db) as conn:
        hashes = conn.execute("SELECT retrieval_config_hash FROM answer_cache").fetchall()
    assert len({h[0] for h in hashes}) == 2  # two entries, told apart by the config hash


# --- off, eval, the trace ----------------------------------------------------------------------


async def test_rerank_off_changes_nothing(db: str) -> None:
    llm = FakeLLMProvider([good()])
    async with app_client(make_settings(database_url=db), provider=llm) as client:
        response = await ask(client)

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["rerank_used"] is False
    assert set(body["meta"]["latency_ms"]) == {"total", "embed", "retrieval", "llm"}
    row = row_of(db, response)
    assert (row["rerank_used"], row["rerank_error"], row["latency_rerank_ms"]) == (
        False,
        None,
        None,
    )
    assert row["shadow_cost_usd"] == EMBEDDING_COST
    assert rerank_calls_today(db) is None  # no cap exists, daily_usage is untouched


async def test_eval_mode_has_no_cap_but_keeps_the_stage_timeout(db: str, tmp_path: Path) -> None:
    # A cap of 0 would refuse everything elsewhere. In eval mode the manual run has its own call
    # budget (Tech §15.6), so both requests reach the reranker; the second one hangs and degrades.
    settings = rerank_settings(
        db, cache_dir=tmp_path, rerank_daily_cap=0, rerank_timeout_s=0.05, **EVAL_SETTINGS
    )
    reranker = FakeReranker([favoring({12: 0.9}), Hang()])
    async with app_client(
        settings, provider=FakeLLMProvider([good(), good()]), reranker=reranker
    ) as client:
        used = await ask(client, QUESTION)
        timed_out = await ask(client, OTHER_QUESTION)

    assert len(reranker.calls) == 2
    first, second = row_of(db, used), row_of(db, timed_out)
    assert (first["source"], first["rerank_used"], first["rerank_error"]) == ("eval", True, None)
    assert (second["rerank_used"], second["rerank_error"]) == (False, "timeout")
    assert rerank_calls_today(db) is None  # nothing reserved, daily_usage untouched


@pytest.mark.parametrize("paced", [False, True])
async def test_the_stage_timeout_allows_for_the_eval_pacers_wait_and_for_nothing_else(
    db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paced: bool
) -> None:
    """The eval pacer waits for its turn inside ``rerank()``, so the stage scope around it is
    widened by that wait (6.04); the adapter keeps its own timeout on the call. No sleeping here:
    the scope's length is read off a spy, and any other reranker gets exactly ``RERANK_TIMEOUT_S``
    as before."""
    delays: list[float | None] = []
    real_timeout = asyncio.timeout

    def spy(delay: float | None) -> Any:
        delays.append(delay)
        return real_timeout(delay)

    monkeypatch.setattr(asyncio, "timeout", spy)
    settings = rerank_settings(db, cache_dir=tmp_path, rerank_timeout_s=0.05, **EVAL_SETTINGS)
    with KVCache(tmp_path / "rerank.sqlite") as cache:
        reranker: Any = FakeReranker()
        if paced:
            reranker = CachedReranker(PacedReranker(reranker, min_interval_s=6.0), cache)
        async with open_runtime(
            settings,
            embedder=FakeEmbedder(dim=EMBEDDING_DIM),
            provider=FakeLLMProvider([good()]),
            reranker=reranker,
        ) as runtime:
            await runtime.pipeline.ask(QUESTION, uuid4())

    expected = 0.05 + (6.0 if paced else 0.0)
    assert [d for d in delays if d is not None and d < 10] == [pytest.approx(expected)]


async def test_the_confidence_weight_for_rerank_stays_zero_and_the_score_is_only_reported(
    db: str,
) -> None:
    assert make_settings().confidence_w_rerank == 0.0
    claims: list[dict[str, Any]] = []
    for score in (0.99, 0.9):  # same order, a different score for the chunk that c1 is
        clear_answer_cache(db)
        reranker = FakeReranker([favoring({12: score, 7: 0.3})])
        async with app_client(
            rerank_settings(db), provider=FakeLLMProvider([good()]), reranker=reranker
        ) as client:
            claims.append((await ask(client)).json()["claims"][0])

    assert [c["confidence_components"]["rerank"] for c in claims] == pytest.approx([0.99, 0.9])
    assert claims[0]["confidence"] == claims[1]["confidence"]  # the weight is 0: no influence


async def test_in_process_callers_see_the_rerank_outcome_on_the_trace(db: str) -> None:
    """The CLI and the evals read the trace, not the API response (which has no error code)."""
    settings = rerank_settings(db)

    async def run(reranker: FakeReranker) -> RequestTrace:
        trace = RequestTrace(request_id=uuid4(), timer=StageTimer(time.perf_counter))
        async with open_runtime(
            settings,
            embedder=FakeEmbedder(dim=EMBEDDING_DIM),
            provider=FakeLLMProvider([good()]),
            reranker=reranker,
        ) as runtime:
            await runtime.pipeline.ask(QUESTION, trace.request_id, trace=trace)
        clear_answer_cache(db)
        return trace

    used = await run(FakeReranker([favoring({12: 0.99})]))
    degraded = await run(FakeReranker([ProviderTimeout("slow")]))

    assert (used.rerank_used, used.rerank_error, used.rerank_calls) == (True, None, 1)
    assert used.rerank_model == MODEL
    assert used.context_chunks["c1"].rerank_score == pytest.approx(0.99)
    assert len(used.retrieved_section_ids) == CANDIDATES  # the candidates, in RRF order
    assert (degraded.rerank_used, degraded.rerank_error) == (False, "timeout")
    assert all(c.rerank_score is None for c in degraded.context_chunks.values())
    assert UUID(str(used.request_id)) != degraded.request_id


# --- start-up ----------------------------------------------------------------------------------


async def test_cohere_without_a_key_or_a_model_stops_the_runtime_before_any_request(
    db: str,
) -> None:
    embedder, llm = FakeEmbedder(dim=EMBEDDING_DIM), FakeLLMProvider()
    with pytest.raises(ProviderConfigError, match="COHERE_API_KEY"):
        async with open_runtime(
            make_settings(database_url=db, rerank_provider="cohere", rerank_model=MODEL),
            embedder=embedder,
            provider=llm,
        ):
            pass
    with pytest.raises(ProviderConfigError, match="RERANK_MODEL"):
        async with open_runtime(
            make_settings(database_url=db, rerank_provider="cohere", cohere_api_key="key"),
            embedder=embedder,
            provider=llm,
        ):
            pass
