"""``POST /v1/ask`` end to end: real pgvector index, fake embedder, scripted fake provider."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import httpx
import psycopg
import pytest

from grounded.generation.confidence import COMPONENT_KEYS
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import ProviderBadOutput, ProviderRateLimited
from grounded.ingest.embed import FakeEmbedder, TaskType, Vector
from grounded.retrieval.config import RetrievalConfig
from grounded.schemas.api import AskResponse
from grounded.schemas.llm import LLMAnswer, LLMClaim
from tests.hybrid_corpus import CORPUS, PAGE, insert_page
from tests.support import EMBEDDING_DIM, app_client, insert_index_version, make_settings

pytestmark = pytest.mark.integration

QUESTION = "Where does the quokka sleep?"
CONFIG_HASH = "1" * 64


class Index(NamedTuple):
    url: str
    title: str


@pytest.fixture
def index(test_database_url: str) -> Iterator[Index]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash=CONFIG_HASH)
        insert_page(conn, version, CORPUS)
    yield Index(url=PAGE[1], title=PAGE[2])
    wipe()


def claim(text: str, *labels: str) -> LLMClaim:
    return LLMClaim(text=text, citation_ids=list(labels), self_confidence=0.9)


def sources_in(user_prompt: str) -> dict[str, str]:
    """label -> url of the ``<source>`` blocks the model was given."""
    return dict(re.findall(r'<source id="(c\d)" section="[^"]*" url="([^"]*)">', user_prompt))


async def test_ask_returns_a_schema_valid_response(test_database_url: str, index: Index) -> None:
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="answered",
                answer_markdown="Quokkas sleep. [c2] They nap, too. [c1][c2] Odd. [c9]",
                claims=[claim("Quokkas sleep.", "c2"), claim("They nap.", "c1", "c2", "c9")],
                follow_up_questions=["What do wombats do?"],
            )
        ]
    )
    settings = make_settings(database_url=test_database_url)
    async with app_client(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})

    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    [call] = provider.calls
    sources = sources_in(call.user)
    assert len(sources) == settings.k_context
    assert QUESTION in call.user

    # Markers are rewritten by first appearance; the unknown label c9 is removed everywhere.
    assert body.answer_markdown == "Quokkas sleep. [1] They nap, too. [2][1] Odd. "
    assert [c.citations for c in body.claims] == [[1], [2, 1]]
    assert body.follow_up_questions == ["What do wombats do?"]
    # Citations come from the DB, in display order: c2 is [1], c1 is [2].
    assert [c.n for c in body.citations] == [1, 2]
    assert [str(c.url) for c in body.citations] == [sources["c2"], sources["c1"]]
    for citation in body.citations:
        assert str(citation.url).startswith(index.url)
        assert citation.title == index.title
        assert citation.breadcrumb.startswith(f"{index.title} > Part ")
        assert citation.snippet

    # Confidence is computed by the server (3.10), never the model's self_confidence (0.9 here).
    first, second = body.claims
    for scored in body.claims:
        assert set(scored.confidence_components) == set(COMPONENT_KEYS)
        assert 0.0 < scored.confidence < 1.0
        assert scored.confidence != 0.9
        assert scored.confidence_components["self_confidence"] == 0.9
        assert scored.confidence_components["rerank"] == 0.0  # rerank is off
        assert scored.confidence_components["retrieval"] > 0.0  # real hybrid signals were used
    # One valid source gives n/(n+1) = 1/2, two give 2/3; the unknown label c9 is not a source.
    assert first.confidence_components["citations"] == pytest.approx(1 / 2)
    assert second.confidence_components["citations"] == pytest.approx(2 / 3)
    assert body.min_confidence == min(first.confidence, second.confidence)

    meta = body.meta
    assert (meta.provider, meta.model) == ("fake", "fake-model")
    assert (meta.fallback_used, meta.cache_hit, meta.rerank_used) == (False, False, False)
    assert re.fullmatch(r"answer_v1@[0-9a-f]{8}", meta.prompt_version)
    assert meta.index_version == f"0.0.1@{CONFIG_HASH[:8]}"
    expected_hash = RetrievalConfig.from_settings(settings, "hybrid").config_hash
    assert meta.retrieval_config_hash == expected_hash
    assert meta.tokens == {"input": 100, "output": 50}
    # The fake provider is free; the 28-character question is embedded: 28 * 0.15 / 1e6.
    assert meta.shadow_cost_usd == pytest.approx(4.2e-06)
    assert set(meta.latency_ms) >= {"total", "embed", "retrieval", "llm"}
    assert call.temperature == settings.llm_temperature
    assert call.max_output_tokens == settings.llm_max_output_tokens
    assert call.timeout_s == settings.llm_timeout_s


async def test_a_claim_without_a_valid_citation_is_capped_and_sets_min_confidence(
    test_database_url: str, index: Index
) -> None:
    # The answer cites c1 in the text (so it is valid), but the second claim cites only an unknown
    # label: its confidence is capped, however confident the model says it is.
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="answered",
                answer_markdown="Quokkas sleep. [c1] Odd.",
                claims=[claim("Quokkas sleep.", "c1"), claim("Odd.", "c9")],
            )
        ]
    )
    settings = make_settings(database_url=test_database_url)
    async with app_client(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    body = AskResponse.model_validate(response.json())
    cited, uncited = body.claims
    assert uncited.confidence <= settings.confidence_uncited_cap
    assert uncited.confidence_components["citations"] == 0.0
    assert uncited.confidence_components["self_confidence"] == 0.9
    assert cited.confidence > uncited.confidence
    assert body.min_confidence == uncited.confidence


async def test_insufficient_context_has_no_claims_and_no_min_confidence(
    test_database_url: str, index: Index
) -> None:
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="insufficient_context",
                answer_markdown="The documentation provided does not cover this.",
                claims=[],
            )
        ]
    )
    settings = make_settings(database_url=test_database_url)
    async with app_client(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    assert (body.status, body.claims, body.citations, body.min_confidence) == (
        "insufficient_context",
        [],
        [],
        None,
    )


async def test_provider_output_that_fails_validation_is_a_502(
    test_database_url: str, index: Index
) -> None:
    bad = ProviderBadOutput("bad", raw="{", validation_error="oops")
    provider = FakeLLMProvider([bad, bad])
    settings = make_settings(database_url=test_database_url)
    async with app_client(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "validation_failed"
    assert "oops" not in response.text
    assert len(provider.calls) == 2


async def test_rate_limited_provider_is_a_503_with_retry_after(
    test_database_url: str, index: Index
) -> None:
    error = ProviderRateLimited("slow down", retry_after_s=6.2, is_quota=False)
    settings = make_settings(database_url=test_database_url)
    provider = FakeLLMProvider([error])
    async with app_client(
        settings,
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=provider,
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "7"
    error_body = response.json()["error"]
    assert (error_body["code"], error_body["retry_after_s"]) == ("provider_unavailable", 7)
    assert len(provider.calls) == 1  # a rate limit is not bad output: no retry


class _RateLimitedEmbedder(FakeEmbedder):
    """What the request-path embedder does on a per-minute 429: raise at once, never wait."""

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        raise ProviderRateLimited("embedding quota per minute", retry_after_s=50.0, is_quota=False)


async def test_a_rate_limited_embedding_is_a_503_at_once_and_is_logged(
    test_database_url: str, index: Index
) -> None:
    settings = make_settings(database_url=test_database_url)
    provider = FakeLLMProvider([])
    async with app_client(
        settings, embedder=_RateLimitedEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "50"
    assert response.json()["error"]["code"] == "provider_unavailable"
    assert provider.calls == []  # no generation without a query vector
    with psycopg.connect(test_database_url) as conn:
        row = conn.execute(
            "SELECT outcome, http_status FROM request_logs WHERE id = %s",
            (response.json()["request_id"],),
        ).fetchone()
    assert row == ("provider_unavailable", 503)


async def test_embedder_that_built_another_index_is_refused(
    test_database_url: str, index: Index
) -> None:
    # Same dimension, different model: the vectors would be silently incomparable.
    settings = make_settings(database_url=test_database_url)
    embedder = FakeEmbedder(model="other-embedding", dim=EMBEDDING_DIM)
    async with app_client(settings, embedder=embedder, raise_app_exceptions=False) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"


async def test_ask_without_an_active_index_is_a_500_internal_error(test_database_url: str) -> None:
    with psycopg.connect(test_database_url, autocommit=True) as conn:
        conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")
    async with app_client(make_settings(database_url=test_database_url)) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "internal_error"
    assert "no index" in body["error"]["message"]


async def test_citations_carry_db_urls_with_anchors_and_a_snippet(
    test_database_url: str, index: Index
) -> None:
    provider = FakeLLMProvider(
        [LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[])]
    )
    async with app_client(
        make_settings(database_url=test_database_url),
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=provider,
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    body = AskResponse.model_validate(response.json())
    [citation] = body.citations
    assert "#" in str(citation.url)
    assert len(citation.snippet) <= 300


async def test_insufficient_context_claims_are_dropped(
    test_database_url: str, index: Index
) -> None:
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="insufficient_context",
                answer_markdown="The documentation does not cover this.",
                claims=[claim("Quokkas sleep.", "c1")],
            )
        ]
    )
    async with app_client(
        make_settings(database_url=test_database_url),
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=provider,
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    assert (body.claims, body.citations, body.min_confidence) == ([], [], None)


FEEDBACK = "Your previous output was invalid because"


def good_answer() -> LLMAnswer:
    return LLMAnswer(
        status="answered",
        answer_markdown="Quokkas sleep. [c1]",
        claims=[claim("Quokkas sleep.", "c1")],
    )


def uncited_answer() -> LLMAnswer:
    return LLMAnswer(
        status="answered",
        answer_markdown="Quokkas sleep. [c9]",
        claims=[claim("Quokkas sleep.", "c9")],
    )


async def ask(test_database_url: str, provider: FakeLLMProvider) -> httpx.Response:
    async with app_client(
        make_settings(database_url=test_database_url),
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=provider,
    ) as client:
        return await client.post("/v1/ask", json={"question": QUESTION})


async def test_invalid_json_is_retried_once_with_the_error_and_then_succeeds(
    test_database_url: str, index: Index
) -> None:
    provider = FakeLLMProvider(["{", good_answer()])
    response = await ask(test_database_url, provider)

    assert response.status_code == 200
    AskResponse.model_validate(response.json())
    first, retry = provider.calls
    assert FEEDBACK not in first.user
    # The retry is the original message plus the feedback section with the validation error.
    assert retry.user.startswith(first.user)
    # Compact feedback (Phase 3 review #5): where and what, no echoed input, no Pydantic links.
    assert retry.user.endswith(
        f"{FEEDBACK} output: Invalid JSON: EOF while parsing an object at line 1 column 1"
    )
    assert "input_value" not in retry.user
    assert "errors.pydantic.dev" not in retry.user
    assert (retry.system, retry.temperature) == (first.system, first.temperature)
    assert provider.remaining == 0


async def test_zero_valid_citations_is_retried_once_with_the_reason_and_then_succeeds(
    test_database_url: str, index: Index
) -> None:
    provider = FakeLLMProvider([uncited_answer(), good_answer()])
    response = await ask(test_database_url, provider)

    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    assert [c.n for c in body.citations] == [1]
    first, retry = provider.calls
    assert retry.user.startswith(first.user)
    assert f"{FEEDBACK} the status is 'answered' but the answer cites no valid source label" in (
        retry.user
    )
    # Both attempts were billed: their tokens add up.
    assert body.meta.tokens == {"input": 200, "output": 100}


async def test_invalid_json_twice_is_a_502_after_exactly_two_calls(
    test_database_url: str, index: Index
) -> None:
    # A third step is scripted to prove it is never reached.
    provider = FakeLLMProvider(["{", "[]", good_answer()])
    response = await ask(test_database_url, provider)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "validation_failed"
    assert len(provider.calls) == 2
    assert provider.remaining == 1


async def test_zero_valid_citations_twice_is_a_502_after_exactly_two_calls(
    test_database_url: str, index: Index
) -> None:
    provider = FakeLLMProvider([uncited_answer(), uncited_answer(), good_answer()])
    response = await ask(test_database_url, provider)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "validation_failed"
    assert len(provider.calls) == 2
    assert provider.remaining == 1


async def test_a_rate_limit_on_the_retry_is_a_503_with_retry_after(
    test_database_url: str, index: Index
) -> None:
    error = ProviderRateLimited("slow down", retry_after_s=3.0, is_quota=False)
    provider = FakeLLMProvider(["{", error])
    response = await ask(test_database_url, provider)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "3"
    assert len(provider.calls) == 2


# --- refusal (3.11, Tech §9.7) ------------------------------------------------------------------


async def test_a_refusal_is_citation_free_keeps_its_follow_ups_and_is_not_retried(
    test_database_url: str, index: Index
) -> None:
    # A refusal that nevertheless carries a valid marker (c1), an unknown one (c9) and a claim.
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="insufficient_context",
                answer_markdown="The documentation does not cover Django. [c1][c9]",
                claims=[claim("Django is covered.", "c1")],
                follow_up_questions=["How do I add middleware in FastAPI?"],
            ),
            good_answer(),
        ]
    )
    response = await ask(test_database_url, provider)

    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    assert body.status == "insufficient_context"
    assert body.answer_markdown == "The documentation does not cover Django. "
    assert (body.claims, body.citations, body.min_confidence) == ([], [], None)
    assert body.follow_up_questions == ["How do I add middleware in FastAPI?"]
    assert len(provider.calls) == 1  # a refusal with no citation is not bad output
    assert provider.remaining == 1


async def test_the_http_api_has_no_no_rag_mode(test_database_url: str, index: Index) -> None:
    provider = FakeLLMProvider([good_answer()])
    async with app_client(
        make_settings(database_url=test_database_url),
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=provider,
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION, "mode": "no_rag"})

    assert response.status_code == 200
    body = AskResponse.model_validate(response.json())
    assert body.meta.prompt_version.startswith("answer_v1@")  # still the retrieval prompt
    [call] = provider.calls
    assert "<source" in call.user
