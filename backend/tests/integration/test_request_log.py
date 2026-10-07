"""One ``request_logs`` row per ``POST /v1/ask``, whatever the outcome (3.12, Tech §14).

Real pgvector index and table, fake embedder and provider, and a hand-driven clock: the fake
embedder "takes" 20 ms, each generation attempt 300 ms, the database work none, so every latency
below is exact instead of "roughly".
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from decimal import Decimal
from typing import Any, NamedTuple
from uuid import UUID

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row
from pydantic import BaseModel

import grounded.observability.request_log
from grounded.generation.pipeline import AskMode, AskPipeline
from grounded.generation.providers.base import GenerationResult, Usage
from grounded.generation.providers.fake import FakeLLMProvider, ScriptStep
from grounded.infra.hashing import question_hash
from grounded.infra.logging import JsonFormatter
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
)
from grounded.ingest.embed import FakeEmbedder, TaskType, Vector
from grounded.observability.cost import PricingError
from grounded.observability.request_log import RequestTrace
from grounded.runtime import open_runtime
from grounded.schemas.api import AskResponse
from grounded.schemas.llm import LLMAnswer, LLMClaim
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import (
    EMBEDDING_DIM,
    FakeClock,
    app_client,
    insert_index_version,
    make_settings,
)

pytestmark = pytest.mark.integration

QUESTION = "Where does the quokka sleep?"
CONFIG_HASH = "1" * 64
EMBED_S = 0.020
LLM_S = 0.300
# The one priced generator model (pricing.toml, PRD D46); the fake provider is renamed to match.
GEMINI = "gemini-3.5-flash-lite"


class TimedEmbedder(FakeEmbedder):
    def __init__(self, clock: FakeClock) -> None:
        super().__init__(dim=EMBEDDING_DIM)
        self._clock = clock

    async def embed(self, texts: Any, task_type: TaskType) -> list[Vector]:
        self._clock.advance(EMBED_S)
        return await super().embed(texts, task_type)


class TimedProvider(FakeLLMProvider):
    """Every attempt takes ``LLM_S``, the failed ones included."""

    def __init__(self, clock: FakeClock, script: list[ScriptStep], **kwargs: Any) -> None:
        super().__init__(script, **kwargs)
        self._clock = clock

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
        self._clock.advance(LLM_S)
        return await super().generate(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )


class Env(NamedTuple):
    url: str
    version_id: int
    clock: FakeClock


@pytest.fixture
def env(test_database_url: str) -> Iterator[Env]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash=CONFIG_HASH)
        insert_page(conn, version, CORPUS)
    yield Env(test_database_url, version, FakeClock())
    wipe()


def claim(text: str, *labels: str) -> LLMClaim:
    return LLMClaim(text=text, citation_ids=list(labels), self_confidence=0.9)


def good(status: str = "answered") -> LLMAnswer:
    return LLMAnswer.model_validate(
        {
            "status": status,
            "answer_markdown": "Quokkas sleep. [c1] Odd. [c9]",
            "claims": [claim("Quokkas sleep.", "c1").model_dump()],
        }
    )


def uncited() -> LLMAnswer:
    """Valid JSON whose only label is not a source of this request: bad output (Tech §9.5)."""
    return LLMAnswer(
        status="answered",
        answer_markdown="Quokkas sleep. [c9]",
        claims=[claim("Quokkas sleep.", "c9")],
    )


def provider(env: Env, script: list[ScriptStep]) -> TimedProvider:
    # 1,000 input and 500 output tokens per attempt, priced as the real Gemini model.
    return TimedProvider(
        env.clock,
        script,
        name="gemini",
        model=GEMINI,
        usage=Usage(input_tokens=1_000, output_tokens=500),
    )


async def post(
    env: Env,
    script: list[ScriptStep],
    json: dict[str, Any] | None = None,
    **client_args: Any,
) -> tuple[httpx.Response, TimedProvider]:
    llm = provider(env, script)
    settings = make_settings(database_url=env.url)
    async with app_client(
        settings, embedder=TimedEmbedder(env.clock), provider=llm, clock=env.clock, **client_args
    ) as client:
        response = await client.post(
            "/v1/ask", json=json if json is not None else {"question": QUESTION}
        )
    return response, llm


def rows(env: Env) -> list[dict[str, Any]]:
    with psycopg.connect(env.url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM request_logs ORDER BY created_at, id")
        return cur.fetchall()


def one_row(env: Env) -> dict[str, Any]:
    [row] = rows(env)
    return row


# --- answered ---------------------------------------------------------------------------------


async def test_an_answered_request_writes_one_complete_row(env: Env) -> None:
    response, _ = await post(env, [good()])

    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    row = one_row(env)
    meta = body.meta

    assert row["id"] == meta.request_id
    assert (row["outcome"], row["http_status"], row["error_code"]) == ("answered", 200, None)
    assert (row["source"], row["ip_hash"], row["cache_hit"]) == ("api", None, False)
    assert row["question"] == QUESTION
    assert row["question_hash"] == question_hash(QUESTION)
    assert (row["provider"], row["model"]) == ("gemini", GEMINI)
    assert (row["fallback_used"], row["rerank_used"], row["rerank_error"]) == (False, False, None)
    assert row["prompt_version"] == meta.prompt_version
    assert row["index_version_id"] == env.version_id
    assert row["retrieval_config_hash"] == meta.retrieval_config_hash
    # The latencies are exact on the injected clock, and the response says the same.
    assert (row["latency_embed_ms"], row["latency_retrieval_ms"], row["latency_llm_ms"]) == (
        20,
        0,
        300,
    )
    assert row["latency_rerank_ms"] is None  # never ran: NULL, not 0
    assert row["latency_total_ms"] == 320
    assert meta.latency_ms == {"total": 320, "embed": 20, "retrieval": 0, "llm": 300}
    # 1,000 in * 0.30/1e6 + 500 out * 2.50/1e6 + 28 embed chars * 0.15/1e6 = 0.0015542
    assert (row["input_tokens"], row["output_tokens"]) == (1_000, 500)
    assert meta.tokens == {"input": 1_000, "output": 500}
    assert row["shadow_cost_usd"] == Decimal("0.00155420")
    assert meta.shadow_cost_usd == pytest.approx(0.0015542)
    # Citation bookkeeping: c1 is valid, c9 was removed and counted.
    assert (row["citation_count"], row["invalid_citation_count"]) == (1, 1)
    assert row["validation_retries"] == 0
    assert body.min_confidence is not None
    assert row["min_claim_confidence"] == pytest.approx(body.min_confidence)


async def test_removed_urls_are_counted_on_the_summary_line(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    linked = LLMAnswer(
        status="answered",
        answer_markdown="Quokkas sleep, see [the docs](https://example.com/q). [c1]",
        claims=[claim("Quokkas sleep.", "c1")],
    )
    response, _ = await post(env, [linked])

    assert response.status_code == 200
    assert "https://example.com" not in response.json()["answer_markdown"]
    [summary] = [r for r in caplog.records if r.getMessage() == "request completed"]
    assert summary.__dict__["removed_url_count"] == 1
    assert summary.__dict__["dropped_claim_count"] == 0


async def test_a_retried_request_records_the_retry_and_both_attempts(env: Env) -> None:
    response, llm = await post(env, [uncited(), good()])

    assert response.status_code == 200
    assert len(llm.calls) == 2
    row = one_row(env)
    assert row["validation_retries"] == 1
    assert (row["input_tokens"], row["output_tokens"]) == (2_000, 1_000)
    assert row["latency_llm_ms"] == 600  # both attempts
    # 2,000 * 0.30/1e6 + 1,000 * 2.50/1e6 + 0.0000042
    assert row["shadow_cost_usd"] == Decimal("0.00310420")


async def test_a_partial_answer_is_logged_as_partial(env: Env) -> None:
    await post(env, [good("partial")])
    assert one_row(env)["outcome"] == "partial"


async def test_a_refusal_is_logged_without_citations_or_min_confidence(env: Env) -> None:
    refusal = LLMAnswer(
        status="insufficient_context", answer_markdown="Not covered. [c9]", claims=[]
    )
    response, _ = await post(env, [refusal])

    assert response.status_code == 200
    row = one_row(env)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "insufficient_context",
        200,
        None,
    )
    assert (row["citation_count"], row["invalid_citation_count"]) == (0, 1)
    assert row["min_claim_confidence"] is None


# --- failures ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload", [{"question": "ab"}, {"question": "x" * 501}, {}, {"question": 42}]
)
async def test_a_bad_request_is_logged_without_its_text(env: Env, payload: dict[str, Any]) -> None:
    response, llm = await post(env, [], json=payload)

    assert response.status_code == 422
    assert llm.calls == []
    row = one_row(env)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "bad_request",
        422,
        "bad_request",
    )
    assert row["question"] is None  # the rejected text is not kept
    text = payload.get("question")
    assert row["question_hash"] == question_hash(text if isinstance(text, str) else "")
    assert row["id"] == UUID(response.json()["request_id"])
    # Nothing ran, so there is nothing to attribute: no provider, versions, tokens or stages.
    for column in (
        "provider",
        "model",
        "prompt_version",
        "index_version_id",
        "retrieval_config_hash",
        "latency_embed_ms",
        "latency_retrieval_ms",
        "latency_llm_ms",
        "input_tokens",
        "output_tokens",
        "citation_count",
    ):
        assert row[column] is None, column
    assert row["shadow_cost_usd"] == 0
    assert row["latency_total_ms"] == 0


async def test_output_that_fails_validation_is_logged_with_what_it_spent(env: Env) -> None:
    # Two replies that are not JSON. Both were billed, and the adapter puts each reply's usage on
    # the ProviderBadOutput it raises (Phase 3 review #6), so the row counts both.
    response, llm = await post(env, ["{", "{"])

    assert response.status_code == 502
    assert len(llm.calls) == 2
    row = one_row(env)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "validation_failed",
        502,
        "validation_failed",
    )
    assert row["validation_retries"] == 1
    assert row["latency_llm_ms"] == 600
    # 2,000 in * 0.30/1e6 + 1,000 out * 2.50/1e6 + 28 embed chars * 0.15/1e6 = 0.0031042
    assert (row["input_tokens"], row["output_tokens"]) == (2_000, 1_000)
    assert row["shadow_cost_usd"] == Decimal("0.00310420")
    assert row["question"] == QUESTION


async def test_bad_output_without_reported_usage_charges_only_the_embedding(env: Env) -> None:
    # An adapter that could not read any usage raises ProviderBadOutput with 0 tokens: nothing is
    # invented, and only the embedding (0.0000042) is charged.
    bad = ProviderBadOutput("bad", raw="{", validation_error="oops")
    response, _ = await post(env, [bad, bad])

    assert response.status_code == 502
    row = one_row(env)
    assert (row["input_tokens"], row["output_tokens"]) == (None, None)
    assert row["shadow_cost_usd"] == Decimal("0.00000420")


async def test_an_answer_with_no_valid_citation_twice_keeps_its_tokens_and_counts(
    env: Env,
) -> None:
    response, _ = await post(env, [uncited(), uncited()])

    assert response.status_code == 502
    row = one_row(env)
    assert row["outcome"] == "validation_failed"
    assert (row["input_tokens"], row["output_tokens"]) == (2_000, 1_000)  # two answers were billed
    # The label c9 is invalid in the text and again in the claim (citations.py counts both).
    assert row["invalid_citation_count"] == 2
    assert row["citation_count"] is None  # no answer was produced


async def test_an_unavailable_provider_is_logged_with_the_time_it_took(env: Env) -> None:
    response, _ = await post(env, [ProviderRateLimited("slow", retry_after_s=2.0, is_quota=False)])

    assert response.status_code == 503
    row = one_row(env)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "provider_unavailable",
        503,
        "provider_unavailable",
    )
    assert row["latency_llm_ms"] == 300
    assert row["latency_total_ms"] == 320
    assert (row["input_tokens"], row["output_tokens"]) == (None, None)
    assert row["index_version_id"] == env.version_id


async def test_an_internal_error_is_logged(env: Env) -> None:
    # The embedder built another index: refused before anything is spent.
    settings = make_settings(database_url=env.url)
    embedder = FakeEmbedder(model="other-embedding", dim=EMBEDDING_DIM)
    async with app_client(
        settings, embedder=embedder, clock=env.clock, raise_app_exceptions=False
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})

    assert response.status_code == 500
    row = one_row(env)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "internal_error",
        500,
        "internal_error",
    )
    assert row["question"] == QUESTION
    assert row["index_version_id"] == env.version_id
    assert row["latency_embed_ms"] is None  # refused before the embedding


async def test_a_rejected_provider_request_is_a_500_internal_error_without_details(
    env: Env,
) -> None:
    # A 4xx other than 429 (a bad key, a bad request) is our fault and not retryable; until the
    # router exists it is "anything else" in Tech §13: 500 internal_error, logged with request_id.
    rejected = ProviderRequestRejected(
        "Gemini API 400 INVALID_ARGUMENT: secret-detail", status_code=400
    )
    response, llm = await post(env, [rejected], raise_app_exceptions=False)

    assert response.status_code == 500
    body = response.json()
    assert body["error"] == {
        "code": "internal_error",
        "message": "Internal error.",
        "retry_after_s": None,
    }
    assert "secret-detail" not in response.text
    assert len(llm.calls) == 1  # not bad output: no retry
    row = one_row(env)
    assert (row["outcome"], row["http_status"], row["error_code"]) == (
        "internal_error",
        500,
        "internal_error",
    )


async def test_an_answer_without_a_status_on_the_trace_is_an_internal_error(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pipeline bug: an answer is returned but the trace has no status. It used to be an assert
    # (stripped under python -O); now it is raised, logged and written as an internal_error row.
    original = AskPipeline.ask

    async def forgetful(
        self: AskPipeline,
        question: str,
        request_id: UUID,
        *,
        mode: AskMode = AskMode.HYBRID,
        trace: RequestTrace | None = None,
    ) -> AskResponse:
        response = await original(self, question, request_id, mode=mode, trace=trace)
        assert trace is not None
        trace.status = None
        return response

    monkeypatch.setattr(AskPipeline, "ask", forgetful)
    response, _ = await post(env, [good()], raise_app_exceptions=False)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert one_row(env)["outcome"] == "internal_error"


async def test_no_active_index_is_logged_as_an_internal_error(env: Env) -> None:
    with psycopg.connect(env.url, autocommit=True) as conn:
        conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")
    response, _ = await post(env, [])

    assert response.status_code == 500
    row = one_row(env)
    assert (row["outcome"], row["error_code"]) == ("internal_error", "internal_error")
    assert row["index_version_id"] is None


async def test_every_request_writes_exactly_one_row(env: Env) -> None:
    llm = provider(env, [good(), ProviderRateLimited("slow", retry_after_s=1.0, is_quota=False)])
    settings = make_settings(database_url=env.url)
    async with app_client(
        settings, embedder=TimedEmbedder(env.clock), provider=llm, clock=env.clock
    ) as client:
        # Two different questions: the same one twice would be answered from the cache (3.13).
        for payload in (
            {"question": QUESTION},
            {"question": "Where does the wombat sleep?"},
            {"question": "ab"},
        ):
            await client.post("/v1/ask", json=payload)
    assert [r["outcome"] for r in rows(env)] == ["answered", "provider_unavailable", "bad_request"]


# --- the log never breaks a response, and never leaks the question ----------------------------


SECRET = "ZEBRA-SECRET-4711"


def rendered_output(caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]) -> str:
    """Everything that reached a log handler or stdout/stderr, as the JSON formatter prints it."""
    captured = capsys.readouterr()
    formatter = JsonFormatter()
    records = "\n".join(formatter.format(record) for record in caplog.records)
    return "\n".join([captured.out, captured.err, records])


@pytest.mark.parametrize(
    "case", ["answered", "refused", "bad_request", "provider", "validation", "internal"]
)
async def test_question_text_never_reaches_the_logs(
    env: Env,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    caplog.set_level(logging.DEBUG)
    question = f"What is the {SECRET} setting?"
    bad = ProviderBadOutput("bad", raw=question, validation_error=question)
    script: dict[str, list[ScriptStep]] = {
        "answered": [good()],
        "refused": [LLMAnswer(status="insufficient_context", answer_markdown="No.", claims=[])],
        "bad_request": [],
        "provider": [ProviderRateLimited("slow", retry_after_s=1.0, is_quota=False)],
        "validation": [bad, bad],
        "internal": [],
    }
    payload = {"question": (question + " ") * 40 if case == "bad_request" else question}
    client_args = {"raise_app_exceptions": False}
    if case == "internal":
        with psycopg.connect(env.url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")
    await post(env, script[case], json=payload, **client_args)

    output = rendered_output(caplog, capsys)
    assert "request completed" in output  # the capture works, and the summary line is there
    assert SECRET not in output
    assert one_row(env)["question_hash"] in output  # the hash stands in for the text


async def test_a_failed_log_write_is_logged_without_the_question_and_does_not_break_the_response(
    env: Env,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A CHECK violation: Postgres puts the whole failing row, question included, into the DETAIL
    # of this error, which is exactly what must not be logged.
    broken = grounded.observability.request_log._INSERT.replace("%(outcome)s", "'bogus'")  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(grounded.observability.request_log, "_INSERT", broken)
    caplog.set_level(logging.DEBUG)
    question = f"What is the {SECRET} setting?"

    response, _ = await post(env, [good()], json={"question": question})

    assert response.status_code == 200  # the answer still goes out
    request_id = response.json()["meta"]["request_id"]
    assert rows(env) == []
    [failure] = [r for r in caplog.records if r.getMessage() == "request log write failed"]
    assert failure.levelno == logging.ERROR
    assert failure.__dict__["request_id"] == request_id
    assert failure.__dict__["sqlstate"] == "23514"  # check_violation
    assert failure.exc_info is None
    assert SECRET not in rendered_output(caplog, capsys)


async def test_a_database_that_is_down_does_not_break_the_log_path_either(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Nothing listens on port 1. The request fails (500) and so does the log write: both are
    # reported, neither raises out of the handler.
    settings = make_settings(
        database_url="postgresql://grounded:grounded@127.0.0.1:1/grounded",
        db_pool_min_size=0,
        db_pool_timeout_s=0.3,
    )
    async with app_client(settings, raise_app_exceptions=False) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 500
    assert any(r.getMessage() == "request log write failed" for r in caplog.records)


# --- startup ----------------------------------------------------------------------------------


async def test_an_unpriced_generator_model_stops_startup(env: Env) -> None:
    unpriced = FakeLLMProvider([], name="gemini", model="gemini-not-in-pricing")
    with pytest.raises(PricingError, match="gemini-not-in-pricing"):
        async with open_runtime(
            make_settings(database_url=env.url),
            embedder=FakeEmbedder(dim=EMBEDDING_DIM),
            provider=unpriced,
        ):
            pytest.fail("the runtime must not open")


async def test_an_unpriced_embedding_model_stops_startup(env: Env) -> None:
    settings = make_settings(database_url=env.url, embedding_model="embedding-not-in-pricing")
    with pytest.raises(PricingError, match="embedding-not-in-pricing"):
        async with open_runtime(
            settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=FakeLLMProvider([])
        ):
            pytest.fail("the runtime must not open")
