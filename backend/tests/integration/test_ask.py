"""``POST /v1/ask`` end to end: real pgvector index, fake embedder, scripted fake provider."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import NamedTuple

import psycopg
import pytest

from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import ProviderBadOutput, ProviderRateLimited
from grounded.ingest.embed import FakeEmbedder
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

    # Placeholders until 3.10: confidence is never the model's self-confidence.
    assert all(c.confidence == 0.0 and c.confidence_components == {} for c in body.claims)
    assert body.min_confidence == 0.0

    meta = body.meta
    assert (meta.provider, meta.model) == ("fake", "fake-model")
    assert (meta.fallback_used, meta.cache_hit, meta.rerank_used) == (False, False, False)
    assert re.fullmatch(r"answer_v1@[0-9a-f]{8}", meta.prompt_version)
    assert meta.index_version == f"0.0.1@{CONFIG_HASH[:8]}"
    expected_hash = RetrievalConfig.from_settings(settings, "hybrid").config_hash
    assert meta.retrieval_config_hash == expected_hash
    assert meta.tokens == {"input": 100, "output": 50}
    assert set(meta.latency_ms) >= {"total", "embed", "retrieval", "llm"}
    assert call.temperature == settings.llm_temperature
    assert call.max_output_tokens == settings.llm_max_output_tokens
    assert call.timeout_s == settings.llm_timeout_s


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
    provider = FakeLLMProvider([ProviderBadOutput("bad", raw="{", validation_error="oops")])
    settings = make_settings(database_url=test_database_url)
    async with app_client(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "validation_failed"
    assert "oops" not in response.text


async def test_rate_limited_provider_is_a_503_with_retry_after(
    test_database_url: str, index: Index
) -> None:
    error = ProviderRateLimited("slow down", retry_after_s=6.2, is_quota=False)
    settings = make_settings(database_url=test_database_url)
    async with app_client(
        settings,
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=FakeLLMProvider([error]),
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "7"
    error_body = response.json()["error"]
    assert (error_body["code"], error_body["retry_after_s"]) == ("provider_unavailable", 7)


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


async def test_an_answer_with_no_valid_citation_is_a_502_until_the_retry_exists(
    test_database_url: str, index: Index
) -> None:
    # 3.07 turns this into one retry; until then the semantic check fails the request.
    provider = FakeLLMProvider(
        [
            LLMAnswer(
                status="answered",
                answer_markdown="Quokkas sleep. [c9]",
                claims=[claim("Quokkas sleep.", "c9")],
            )
        ]
    )
    async with app_client(
        make_settings(database_url=test_database_url),
        embedder=FakeEmbedder(dim=EMBEDDING_DIM),
        provider=provider,
    ) as client:
        response = await client.post("/v1/ask", json={"question": QUESTION})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "validation_failed"


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
