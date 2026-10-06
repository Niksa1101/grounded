"""The ``no_rag`` mode (3.11): no embedding, no retrieval, no sources, no citation check.

The database has **no active index** in every test here, so any attempt to look up the index or to
retrieve would fail with ``NoActiveIndexError`` instead of passing quietly. ``hybrid_search`` is
also replaced by a function that fails the test, and the embedder records its calls.
"""

from __future__ import annotations

from uuid import uuid4

import psycopg
import pytest

import grounded.generation.pipeline
from grounded.generation.pipeline import (
    NO_RAG_INDEX_VERSION,
    NO_RAG_RETRIEVAL_CONFIG_HASH,
    AskMode,
)
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import ProviderBadOutput
from grounded.infra.timing import StageTimer
from grounded.ingest.embed import FakeEmbedder
from grounded.observability.request_log import RequestTrace
from grounded.runtime import open_runtime
from grounded.schemas.api import AskResponse
from grounded.schemas.llm import LLMAnswer, LLMClaim
from tests.support import EMBEDDING_DIM, make_settings

pytestmark = pytest.mark.integration

QUESTION = "Where does the quokka sleep?"


@pytest.fixture(autouse=True)
def no_index_and_no_retrieval(test_database_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with psycopg.connect(test_database_url, autocommit=True) as conn:
        conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    async def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("no_rag must not retrieve")

    monkeypatch.setattr(grounded.generation.pipeline, "hybrid_search", fail)
    return


def uncited(text: str = "Quokkas sleep.") -> LLMAnswer:
    return LLMAnswer(
        status="answered",
        answer_markdown=text,
        claims=[LLMClaim(text="Quokkas sleep.", citation_ids=[], self_confidence=0.9)],
    )


async def ask_no_rag(
    test_database_url: str, provider: FakeLLMProvider, embedder: FakeEmbedder
) -> AskResponse:
    settings = make_settings(database_url=test_database_url)
    async with open_runtime(settings, embedder=embedder, provider=provider) as runtime:
        return await runtime.pipeline.ask(QUESTION, uuid4(), mode=AskMode.NO_RAG)


async def test_no_rag_makes_no_embedding_and_no_retrieval_call(test_database_url: str) -> None:
    provider = FakeLLMProvider([uncited()])
    embedder = FakeEmbedder(dim=EMBEDDING_DIM)
    body = await ask_no_rag(test_database_url, provider, embedder)

    assert embedder.calls == []
    [call] = provider.calls
    assert QUESTION in call.user
    assert "<source" not in call.user
    assert "Sources:" not in call.user
    assert body.status == "answered"
    assert body.citations == []
    meta = body.meta
    assert meta.prompt_version.startswith("answer_no_rag_v1@")
    assert (meta.index_version, meta.retrieval_config_hash) == (
        NO_RAG_INDEX_VERSION,
        NO_RAG_RETRIEVAL_CONFIG_HASH,
    )
    assert (meta.latency_ms["embed"], meta.latency_ms["retrieval"]) == (0, 0)
    assert meta.rerank_used is False


async def test_no_rag_leaves_the_trace_without_an_index_or_an_embedding(
    test_database_url: str,
) -> None:
    # The request log row of such a request would have a NULL index_version_id (DB.md §4) and
    # cost nothing: nothing was embedded, and the fake provider is free.
    provider = FakeLLMProvider([uncited()])
    settings = make_settings(database_url=test_database_url)
    request_id = uuid4()
    trace = RequestTrace(request_id=request_id, timer=StageTimer())
    async with open_runtime(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as runtime:
        body = await runtime.pipeline.ask(QUESTION, request_id, mode=AskMode.NO_RAG, trace=trace)

    assert trace.index_version_id is None
    assert trace.retrieval_config_hash == NO_RAG_RETRIEVAL_CONFIG_HASH
    assert trace.embedding_model is None
    assert (trace.timer.stage_ms("embed"), trace.timer.stage_ms("retrieval")) == (None, None)
    assert trace.timer.stage_ms("llm") is not None
    assert body.meta.shadow_cost_usd == 0.0


async def test_no_rag_claims_have_no_citation_so_their_confidence_is_capped(
    test_database_url: str,
) -> None:
    provider = FakeLLMProvider([uncited()])
    body = await ask_no_rag(test_database_url, provider, FakeEmbedder(dim=EMBEDDING_DIM))

    [claim] = body.claims
    cap = make_settings().confidence_uncited_cap
    assert claim.citations == []
    assert claim.confidence <= cap
    assert claim.confidence_components["citations"] == 0.0
    assert claim.confidence_components["self_confidence"] == 0.9  # high, and it changes nothing
    assert body.min_confidence == claim.confidence


async def test_no_rag_never_retries_for_missing_citations(test_database_url: str) -> None:
    # A second step is scripted to prove it is never consumed: the answer has no citation at all,
    # which would be bad output (and a retry) in hybrid mode.
    provider = FakeLLMProvider([uncited(), uncited()])
    await ask_no_rag(test_database_url, provider, FakeEmbedder(dim=EMBEDDING_DIM))

    assert len(provider.calls) == 1
    assert provider.remaining == 1


async def test_no_rag_removes_and_ignores_stray_markers(test_database_url: str) -> None:
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="answered",
                answer_markdown="Quokkas sleep. [c1]",
                claims=[LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)],
            )
        ]
    )
    body = await ask_no_rag(test_database_url, provider, FakeEmbedder(dim=EMBEDDING_DIM))

    assert len(provider.calls) == 1
    assert body.answer_markdown == "Quokkas sleep. "
    assert (body.citations, [c.citations for c in body.claims]) == ([], [[]])


async def test_no_rag_still_retries_once_on_a_schema_failure_with_its_own_prompt(
    test_database_url: str,
) -> None:
    provider = FakeLLMProvider(["{", uncited()])
    body = await ask_no_rag(test_database_url, provider, FakeEmbedder(dim=EMBEDDING_DIM))

    first, retry = provider.calls
    assert retry.user.startswith(first.user)
    assert "Your previous output was invalid because" in retry.user
    assert retry.system == first.system
    assert body.status == "answered"


async def test_no_rag_schema_failure_twice_is_still_the_provider_error(
    test_database_url: str,
) -> None:
    provider = FakeLLMProvider(["{", "[]", uncited()])
    with pytest.raises(ProviderBadOutput):
        await ask_no_rag(test_database_url, provider, FakeEmbedder(dim=EMBEDDING_DIM))
    assert len(provider.calls) == 2


async def test_no_rag_refusal_is_a_clean_insufficient_context(test_database_url: str) -> None:
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="insufficient_context",
                answer_markdown="I cannot answer that. [c1]",
                claims=[],
                follow_up_questions=["How do I declare a path parameter?"],
            )
        ]
    )
    body = await ask_no_rag(test_database_url, provider, FakeEmbedder(dim=EMBEDDING_DIM))

    assert (body.status, body.claims, body.citations, body.min_confidence) == (
        "insufficient_context",
        [],
        [],
        None,
    )
    assert body.answer_markdown == "I cannot answer that. "
    assert body.follow_up_questions == ["How do I declare a path parameter?"]
