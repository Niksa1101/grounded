"""`APP_ENV=eval` through the real pipeline (4.03, Tech §15.6): the eval LLM cache around a fake
provider, over a real pgvector index. The unit tests of the wrappers are in test_eval_wrappers.py;
what only the pipeline can show is here: a cached run costs the same on paper, a bad cached first
attempt is not served to the retry, and an eval request is logged as `source='eval'`."""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import psycopg
import pytest
from psycopg.rows import dict_row

from grounded.generation.providers.base import Usage
from grounded.generation.providers.eval_wrappers import (
    LLM_EVAL_CACHE_FILE,
    EvalLLM,
    EvalStats,
)
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import ProviderBadOutput
from grounded.infra.timing import StageTimer
from grounded.ingest.embed import FakeEmbedder
from grounded.observability.request_log import RequestTrace
from grounded.runtime import open_runtime
from grounded.schemas.api import AskResponse
from grounded.schemas.llm import LLMAnswer, LLMClaim
from grounded.settings import Settings
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
GEMINI = "gemini-3.5-flash-lite"  # the one priced generator model (pricing.toml, PRD D46)
USAGE = Usage(input_tokens=1_000, output_tokens=500, thinking_tokens=50)


class Env(NamedTuple):
    url: str
    cache_dir: Path


@pytest.fixture
def env(test_database_url: str, tmp_path: Path) -> Iterator[Env]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    yield Env(test_database_url, tmp_path / "cache")
    wipe()


def settings_of(env: Env) -> Settings:
    return make_settings(database_url=env.url, cache_dir=env.cache_dir, **EVAL_SETTINGS)


def provider(*script: Any) -> FakeLLMProvider:
    return FakeLLMProvider(script, name="gemini", model=GEMINI, usage=USAGE)


def good() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[claim])


def uncited() -> LLMAnswer:
    """Valid JSON whose only label is not a source of this request: bad output at the pipeline's
    own check (Tech §9.5), not at the adapter, so the wrapper stores it like any valid reply."""
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c9"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c9]", claims=[claim])


class Run(NamedTuple):
    response: AskResponse | None
    error: Exception | None
    trace: RequestTrace
    stats: EvalStats
    inner: FakeLLMProvider


async def run(env: Env, *script: Any) -> Run:
    """One "process": a fresh pool, a fresh provider with `script`, the same cache file on disk."""
    inner = provider(*script)
    settings = settings_of(env)
    trace = RequestTrace(request_id=uuid4(), timer=StageTimer())
    with KVCache(env.cache_dir / LLM_EVAL_CACHE_FILE) as kv:
        eval_llm = EvalLLM(settings, kv)
        async with open_runtime(
            settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=eval_llm.wrap(inner)
        ) as runtime:
            try:
                response = await runtime.pipeline.ask(QUESTION, trace.request_id, trace=trace)
            except ProviderBadOutput as exc:
                return Run(None, exc, trace, eval_llm.stats, inner)
    return Run(response, None, trace, eval_llm.stats, inner)


async def test_a_second_run_makes_no_provider_call_and_costs_the_same_on_paper(env: Env) -> None:
    first = await run(env, good())
    second = await run(env)  # an empty script: a live call would fail the test

    assert first.response is not None
    assert second.response is not None
    assert second.inner.calls == []
    # The call's timeout is the eval one (EVAL_LLM_TIMEOUT_S), not the request path's 12 s.
    assert [call.timeout_s for call in first.inner.calls] == [40.0]
    assert (first.stats.hits, first.stats.misses) == (0, 1)
    assert (second.stats.hits, second.stats.misses) == (1, 0)
    # Same answer, same tokens and shadow cost: cost and latency accounting works on cached runs.
    assert second.response.answer_markdown == first.response.answer_markdown
    assert second.response.claims == first.response.claims
    assert (
        second.response.meta.tokens == first.response.meta.tokens == {"input": 1_000, "output": 500}
    )
    assert second.response.meta.shadow_cost_usd == first.response.meta.shadow_cost_usd > 0
    # The pipeline tells the two apart: the trace says a cached call was in it.
    assert (first.trace.llm_cache_hits, second.trace.llm_cache_hits) == (0, 1)
    # The answer cache stays off in eval mode: this was the LLM cache, not that one.
    assert not second.response.meta.cache_hit


async def test_a_bad_cached_first_attempt_is_not_served_to_the_retry(env: Env) -> None:
    first = await run(env, uncited(), good())  # bad, then the retry that fixes it: both stored
    second = await run(env)  # nothing live

    assert first.response is not None
    assert second.response is not None
    assert first.trace.validation_retries == second.trace.validation_retries == 1
    assert second.inner.calls == []
    # Two cached calls: the bad first attempt and, under another key (the user message carries the
    # retry feedback), the good retry. The same bad entry is not replayed for the retry.
    assert second.trace.llm_cache_hits == 2
    assert second.response.answer_markdown == first.response.answer_markdown == "Quokkas sleep. [1]"


async def test_a_run_that_failed_validation_twice_fails_the_same_way_without_looping(
    env: Env,
) -> None:
    first = await run(env, uncited(), uncited())
    second = await run(env)

    assert isinstance(first.error, ProviderBadOutput)
    assert isinstance(second.error, ProviderBadOutput)  # the same outcome, replayed
    assert second.inner.calls == []
    assert second.trace.validation_retries == 1  # one retry, as live: it cannot loop
    assert second.trace.llm_cache_hits == 2


async def test_an_eval_request_is_logged_as_source_eval(env: Env) -> None:
    llm = provider(good())
    async with app_client(
        settings_of(env), embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=llm
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 200

    with psycopg.connect(env.url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT source, cache_hit, shadow_cost_usd FROM request_logs")
        [row] = cur.fetchall()
    assert (row["source"], row["cache_hit"]) == ("eval", False)
    assert row["shadow_cost_usd"] > Decimal(0)


async def test_a_provider_built_from_the_settings_is_wrapped_only_in_eval_mode(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    import grounded.runtime

    built = provider(good())
    monkeypatch.setattr(grounded.runtime, "build_provider", lambda settings: built)
    embedder = FakeEmbedder(dim=EMBEDDING_DIM)

    eval_settings = settings_of(env)
    async with open_runtime(eval_settings, embedder=embedder) as runtime:
        assert runtime.eval_llm is not None
        await runtime.pipeline.ask(QUESTION, uuid4())
        assert (runtime.eval_llm.stats.hits, runtime.eval_llm.stats.misses) == (0, 1)
    assert (env.cache_dir / LLM_EVAL_CACHE_FILE).is_file()

    dev_settings = make_settings(database_url=env.url, cache_dir=env.cache_dir / "dev")
    async with open_runtime(dev_settings, embedder=embedder) as runtime:
        assert runtime.eval_llm is None  # dev and prod have no eval cache
    assert not (env.cache_dir / "dev" / LLM_EVAL_CACHE_FILE).exists()
