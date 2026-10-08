"""The answer cache end to end (3.13, Tech §11, DB §7.2): real pgvector database and tables, fake
embedder and provider. Every test counts the calls that reach the embedder and the provider, because
"a hit does no work" is the whole point of the cache."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from typing import Any, LiteralString, NamedTuple
from uuid import uuid4

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

import grounded.runtime
from grounded.generation.confidence import ConfidenceConfig
from grounded.generation.params import GenerationParams
from grounded.generation.pipeline import AskMode
from grounded.generation.prompts import load_answer_prompt
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.answer_cache import AnswerCache, CacheKey
from grounded.infra.db import create_pool
from grounded.infra.provider_errors import ProviderBadOutput, ProviderRateLimited
from grounded.infra.timing import StageTimer
from grounded.ingest.embed import FakeEmbedder
from grounded.observability.request_log import RequestTrace
from grounded.runtime import open_runtime
from grounded.schemas.api import AskResponse
from grounded.schemas.llm import AnswerStatus, LLMAnswer, LLMClaim
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
CONFIG_HASH = "1" * 64


class Env(NamedTuple):
    url: str
    version_id: int


@pytest.fixture
def env(test_database_url: str) -> Iterator[Env]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash=CONFIG_HASH)
        insert_page(conn, version, CORPUS)
    yield Env(test_database_url, version)
    wipe()


def good(status: AnswerStatus = "answered") -> LLMAnswer:
    return LLMAnswer(
        status=status,
        answer_markdown="Quokkas sleep. [c1] Odd. [c9]",
        claims=[LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)],
    )


def refusal() -> LLMAnswer:
    return LLMAnswer(status="insufficient_context", answer_markdown="Not covered.", claims=[])


async def post(
    env: Env,
    llm: FakeLLMProvider,
    questions: tuple[str, ...] = (QUESTION,),
    *,
    embedder: FakeEmbedder | None = None,
    **settings: Any,
) -> list[httpx.Response]:
    """One app (so one pool and one provider), one POST per question, in order."""
    embedder = embedder or FakeEmbedder(dim=EMBEDDING_DIM)
    responses: list[httpx.Response] = []
    async with app_client(
        make_settings(database_url=env.url, **settings), embedder=embedder, provider=llm
    ) as client:
        for question in questions:
            responses.append(await client.post("/v1/ask", json={"question": question}))
    return responses


def table(env: Env, sql: LiteralString) -> list[dict[str, Any]]:
    with psycopg.connect(env.url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql)
        return cur.fetchall()


def cached(env: Env) -> list[dict[str, Any]]:
    return table(env, "SELECT * FROM answer_cache ORDER BY created_at, cache_key")


def logs(env: Env) -> list[dict[str, Any]]:
    return table(env, "SELECT * FROM request_logs ORDER BY created_at, id")


def body_of(response: httpx.Response) -> AskResponse:
    assert response.status_code == 200, response.text
    return AskResponse.model_validate(response.json())


# --- a hit ------------------------------------------------------------------------------------


async def test_a_repeated_question_is_answered_without_embedding_or_generation(env: Env) -> None:
    llm = FakeLLMProvider([good()])  # a second generation would exhaust the script and fail
    embedder = FakeEmbedder(dim=EMBEDDING_DIM)
    # The same question, written differently: it normalizes to the same key (Tech §11).
    first, second = await post(
        env, llm, (QUESTION, "  where DOES the quokka sleep ?? "), embedder=embedder
    )

    miss, hit = body_of(first), body_of(second)
    assert len(llm.calls) == 1
    assert len(embedder.calls) == 1

    assert (miss.meta.cache_hit, hit.meta.cache_hit) == (False, True)
    # The answer is the stored one, and only ``meta`` is this request's own.
    assert hit.model_dump(exclude={"meta"}) == miss.model_dump(exclude={"meta"})
    assert hit.meta.request_id != miss.meta.request_id
    assert hit.meta.tokens == {"input": 0, "output": 0}
    assert hit.meta.shadow_cost_usd == 0.0
    assert (hit.meta.provider, hit.meta.model) == ("fake", "fake-model")
    assert (hit.meta.prompt_version, hit.meta.index_version) == (
        miss.meta.prompt_version,
        miss.meta.index_version,
    )
    assert hit.meta.retrieval_config_hash == miss.meta.retrieval_config_hash
    assert (hit.meta.fallback_used, hit.meta.rerank_used) == (False, False)
    assert hit.meta.latency_ms["embed"] == 0
    assert hit.meta.latency_ms["llm"] == 0

    # The request log says the same, and the counter of the row moved.
    first_row, second_row = logs(env)
    assert (first_row["cache_hit"], second_row["cache_hit"]) == (False, True)
    assert second_row["id"] == hit.meta.request_id
    assert (second_row["outcome"], second_row["http_status"]) == ("answered", 200)
    assert second_row["question_hash"] == first_row["question_hash"]
    assert second_row["input_tokens"] is None
    assert second_row["output_tokens"] is None
    assert second_row["shadow_cost_usd"] == 0
    assert second_row["latency_embed_ms"] is None
    assert second_row["latency_llm_ms"] is None
    assert second_row["citation_count"] == len(hit.citations)
    assert second_row["min_claim_confidence"] == pytest.approx(hit.min_confidence)
    assert second_row["index_version_id"] == env.version_id
    [row] = cached(env)
    assert row["hit_count"] == 1
    assert row["last_hit_at"] is not None


@pytest.mark.parametrize("status", ["answered", "partial", "insufficient_context"])
async def test_every_valid_outcome_is_cached(env: Env, status: AnswerStatus) -> None:
    llm = FakeLLMProvider([refusal() if status == "insufficient_context" else good(status)])
    first, second = await post(env, llm, (QUESTION, QUESTION))

    assert body_of(first).status == body_of(second).status == status
    assert body_of(second).meta.cache_hit
    assert len(llm.calls) == 1
    assert [row["outcome"] for row in logs(env)] == [status, status]


async def test_the_cache_is_shared_between_app_instances(env: Env) -> None:
    # It lives in Postgres, not in the process: a second instance (a cold serverless start) hits.
    await post(env, FakeLLMProvider([good()]))
    llm = FakeLLMProvider([])
    [response] = await post(env, llm)

    assert body_of(response).meta.cache_hit
    assert llm.calls == []


async def test_a_stored_row_has_the_columns_and_the_retention_limit(env: Env) -> None:
    await post(env, FakeLLMProvider([good()]), answer_cache_ttl_days=7)

    [row] = table(
        env,
        "SELECT *, expires_at - created_at AS ttl FROM answer_cache",
    )
    assert row["normalized_question"] == "where does the quokka sleep"
    assert row["index_version_id"] == env.version_id
    assert row["generator_model"] == "fake-model"
    assert row["prompt_version"] == load_answer_prompt().version
    assert row["hit_count"] == 0
    assert row["last_hit_at"] is None
    assert "meta" not in row["response"]
    assert row["response"]["status"] == "answered"
    assert row["ttl"].days == 7  # ANSWER_CACHE_TTL_DAYS, and never above 30 (DB §4)


# --- what is never cached or served -----------------------------------------------------------


async def test_errors_are_never_cached(env: Env) -> None:
    bad = ProviderBadOutput("bad", raw="{", validation_error="oops")
    llm = FakeLLMProvider(
        [ProviderRateLimited("slow", retry_after_s=1.0, is_quota=False), bad, bad, good()]
    )
    statuses = [r.status_code for r in await post(env, llm, (QUESTION,) * 3)]

    assert statuses == [503, 502, 200]  # each request tried the provider again
    assert len(llm.calls) == 4
    [row] = cached(env)  # only the successful answer
    assert row["response"]["status"] == "answered"
    assert [r["cache_hit"] for r in logs(env)] == [False, False, False]


async def test_a_bad_request_never_reaches_the_cache(env: Env) -> None:
    llm = FakeLLMProvider([])
    [response] = await post(env, llm, ("ab",))

    assert response.status_code == 422
    assert cached(env) == []


async def test_an_expired_row_is_a_miss_and_is_replaced(env: Env) -> None:
    llm = FakeLLMProvider([good(), good()])
    await post(env, llm)
    with psycopg.connect(env.url, autocommit=True) as conn:
        conn.execute(
            "UPDATE answer_cache SET expires_at = now() - interval '1 day', hit_count = 5, "
            "created_at = now() - interval '31 days'"
        )

    [again] = await post(env, llm)

    assert not body_of(again).meta.cache_hit
    assert len(llm.calls) == 2
    # The stale row did not block the new answer (a plain DO NOTHING would have): it was replaced.
    [row] = table(env, "SELECT hit_count, expires_at > now() AS live, created_at FROM answer_cache")
    assert (row["hit_count"], row["live"]) == (0, True)
    [third] = await post(env, llm)
    assert body_of(third).meta.cache_hit
    assert len(llm.calls) == 2


async def test_a_live_row_is_not_overwritten_by_a_concurrent_writer(env: Env) -> None:
    # DO NOTHING semantics for a live row: the first writer's answer stays.
    [first] = await post(env, FakeLLMProvider([good()]))
    [before] = cached(env)
    key = CacheKey.build(
        before["normalized_question"],
        prompt_version=before["prompt_version"],
        index_version_id=before["index_version_id"],
        retrieval_config_hash=before["retrieval_config_hash"],
        generator_model=before["generator_model"],
        confidence=ConfidenceConfig.from_settings(make_settings()),
        generation=GenerationParams.from_settings(make_settings(), "fake"),
    )
    assert key.digest == before["cache_key"]  # the test rebuilt the very key the pipeline used
    rival = body_of(first).model_copy(update={"answer_markdown": "A rival answer."})

    settings = make_settings(database_url=env.url)
    pool = create_pool(settings)
    await pool.open(wait=True)
    try:
        await AnswerCache(pool, ttl_days=30).put(key, rival)
    finally:
        await pool.close()

    [after] = cached(env)
    assert after["response"] == before["response"]
    assert after["created_at"] == before["created_at"]


async def test_a_different_prompt_version_is_a_miss(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    llm = FakeLLMProvider([good(), good()])
    await post(env, llm)
    prompt = load_answer_prompt()
    monkeypatch.setattr(
        grounded.runtime,
        "load_answer_prompt",
        lambda: dataclasses.replace(prompt, version="answer_v1@00000000"),
    )

    [other] = await post(env, llm)

    assert not body_of(other).meta.cache_hit
    assert body_of(other).meta.prompt_version == "answer_v1@00000000"
    assert len(llm.calls) == 2
    assert len(cached(env)) == 2


async def test_a_different_generator_model_is_a_miss(env: Env) -> None:
    await post(env, FakeLLMProvider([good()], model="model-a"))
    llm = FakeLLMProvider([good()], model="model-b")

    [other] = await post(env, llm)

    assert not body_of(other).meta.cache_hit
    assert len(llm.calls) == 1


async def test_a_different_index_version_is_a_miss(env: Env) -> None:
    llm = FakeLLMProvider([good(), good()])
    await post(env, llm)
    with psycopg.connect(env.url, autocommit=True) as conn:
        conn.execute("UPDATE index_versions SET is_active = false")
        newer = insert_index_version(conn, active=True, config_hash="2" * 64)
        insert_page(conn, newer, CORPUS)

    [other] = await post(env, llm, active_index_ttl_s=0.001)

    assert not body_of(other).meta.cache_hit
    assert len(llm.calls) == 2


async def test_a_different_retrieval_config_is_a_miss(env: Env) -> None:
    llm = FakeLLMProvider([good(), good()])
    await post(env, llm)

    [other] = await post(env, llm, k_context=4)

    assert not body_of(other).meta.cache_hit
    assert len(llm.calls) == 2


async def test_changed_confidence_weights_are_a_miss_not_stale_numbers(env: Env) -> None:
    # PRD §12 (closed in 3.13): the stored answer carries server-computed confidence.
    llm = FakeLLMProvider([good(), good()])
    [before] = await post(env, llm)
    [after] = await post(env, llm, confidence_w_self=0.30)

    assert not body_of(after).meta.cache_hit
    assert len(llm.calls) == 2
    assert (
        body_of(after).claims[0].confidence != body_of(before).claims[0].confidence
    )  # recomputed with the new weights, not served from the old row


async def test_a_row_that_no_longer_fits_the_schema_is_a_miss_and_is_replaced(env: Env) -> None:
    llm = FakeLLMProvider([good(), good()])
    await post(env, llm)
    with psycopg.connect(env.url, autocommit=True) as conn:
        conn.execute('UPDATE answer_cache SET response = \'{"status": "answered"}\'::jsonb')

    [again] = await post(env, llm)

    assert not body_of(again).meta.cache_hit
    assert len(llm.calls) == 2
    assert [r["cache_hit"] for r in logs(env)] == [False, False]
    # The stale row was deleted and the fresh answer took its place (``put`` alone could not
    # overwrite a live row), so the next request is a hit again.
    [row] = cached(env)
    assert row["hit_count"] == 0
    AskResponse.model_validate({**row["response"], "meta": body_of(again).meta})
    [third] = await post(env, llm)
    assert body_of(third).meta.cache_hit
    assert len(llm.calls) == 2
    [row] = cached(env)
    assert row["hit_count"] == 1


async def test_a_row_that_is_not_a_json_object_is_a_miss_and_is_replaced(env: Env) -> None:
    llm = FakeLLMProvider([good(), good()])
    await post(env, llm)
    with psycopg.connect(env.url, autocommit=True) as conn:
        conn.execute("UPDATE answer_cache SET response = '[1, 2]'::jsonb")

    [again] = await post(env, llm)

    assert not body_of(again).meta.cache_hit
    assert len(llm.calls) == 2
    [row] = cached(env)
    assert isinstance(row["response"], dict)  # replaced by the fresh answer
    [third] = await post(env, llm)
    assert body_of(third).meta.cache_hit


async def test_the_cache_lookup_is_a_miss_when_the_cache_is_off(env: Env) -> None:
    # Eval mode has no cache; the lookup helper must answer "miss", not fail (it used to assert).
    settings = make_settings(database_url=env.url, **EVAL_SETTINGS)
    async with open_runtime(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=FakeLLMProvider([])
    ) as runtime:
        key = CacheKey.build(
            QUESTION,
            prompt_version="answer_v1@aaaaaaaa",
            index_version_id=env.version_id,
            retrieval_config_hash="r" * 64,
            generator_model="fake-model",
            confidence=ConfidenceConfig.from_settings(settings),
            generation=GenerationParams.from_settings(settings, "fake"),
        )
        trace = RequestTrace(request_id=uuid4(), timer=StageTimer())
        lookup = runtime.pipeline._cached_response  # pyright: ignore[reportPrivateUsage]
        assert await lookup(key, trace.request_id, trace, "0.0.1@abcdef12") is None
    assert cached(env) == []


@pytest.mark.parametrize(
    "change",
    [
        {"llm_temperature": 0.5},
        {"llm_max_output_tokens": 900},
        {"gemini_thinking_level": "low"},
    ],
    ids=["temperature", "max output tokens", "thinking level"],
)
async def test_changed_generation_params_are_a_miss(env: Env, change: dict[str, Any]) -> None:
    # PRD D47: an answer made with other generation settings is not served.
    llm = FakeLLMProvider([good(), good()])
    await post(env, llm)

    [other] = await post(env, llm, **change)

    assert not body_of(other).meta.cache_hit
    assert len(llm.calls) == 2
    assert len(cached(env)) == 2


# --- where the cache is off -------------------------------------------------------------------


async def test_eval_mode_bypasses_the_cache(env: Env) -> None:
    llm = FakeLLMProvider([good(), good()])
    first, second = await post(env, llm, (QUESTION, QUESTION), **EVAL_SETTINGS)

    assert not body_of(first).meta.cache_hit
    assert not body_of(second).meta.cache_hit
    assert len(llm.calls) == 2
    assert cached(env) == []


async def test_eval_mode_ignores_rows_written_outside_it(env: Env) -> None:
    await post(env, FakeLLMProvider([good()]))  # a dev/test run stored an answer
    llm = FakeLLMProvider([good()])

    [response] = await post(env, llm, **EVAL_SETTINGS)

    assert not body_of(response).meta.cache_hit
    assert len(llm.calls) == 1


async def test_no_rag_mode_neither_reads_nor_writes_the_cache(env: Env) -> None:
    await post(env, FakeLLMProvider([good()]))  # a stored hybrid answer for the same question
    [stored] = cached(env)
    llm = FakeLLMProvider(
        [
            LLMAnswer(
                status="answered",
                answer_markdown="Quokkas sleep.",
                claims=[LLMClaim(text="Quokkas sleep.", citation_ids=[], self_confidence=0.9)],
            )
        ]
        * 2
    )
    async with open_runtime(
        make_settings(database_url=env.url),
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=llm,
    ) as runtime:
        for _ in range(2):
            response = await runtime.pipeline.ask(QUESTION, uuid4(), mode=AskMode.NO_RAG)
            assert not response.meta.cache_hit

    assert len(llm.calls) == 2
    assert cached(env) == [stored]  # untouched: no new row, no hit counted
