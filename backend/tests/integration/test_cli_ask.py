"""`grounded ask` end to end: real pgvector index, fake embedder, stub provider."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner

import grounded.cli
import grounded.retrieval.query_embedding
import grounded.runtime
from grounded.cli import app
from grounded.ingest.embed import FakeEmbedder, TokenCounter
from grounded.schemas.api import AskResponse
from grounded.settings import Settings
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import (
    EMBEDDING_DIM,
    insert_index_version,
    make_settings,
    pricing_for_fake_embedder,
)

pytestmark = pytest.mark.integration

runner = CliRunner()


class StubGemini(FakeEmbedder):
    """Stands in for GeminiEmbedder, so no key and no network are needed."""

    @classmethod
    def from_settings(cls, settings: Settings, count_tokens: TokenCounter) -> StubGemini:
        return cls(model=settings.embedding_model, dim=settings.embedding_dim)


@pytest.fixture
def settings(
    test_database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    settings = make_settings(
        database_url=test_database_url,
        cache_dir=tmp_path / "cache",
        embedding_model="fake-embedding",
        embedding_dim=EMBEDDING_DIM,
    )
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.retrieval.query_embedding, "GeminiEmbedder", StubGemini)
    monkeypatch.setattr(grounded.runtime, "load_pricing", pricing_for_fake_embedder)
    yield settings
    wipe()


def test_ask_prints_a_valid_response_with_the_fake_provider(settings: Settings) -> None:
    result = runner.invoke(app, ["ask", "Where does the quokka sleep?", "--fake"])
    assert result.exit_code == 0, result.output
    body = AskResponse.model_validate(json.loads(result.stdout))
    assert body.status == "answered"
    assert body.meta.provider == "fake"
    assert [c.n for c in body.citations] == [1]
    assert body.answer_markdown.endswith("[1]")


def test_ask_without_a_gemini_key_fails_cleanly(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Whatever the environment (a CI secret, a developer .env) says, this run has no key.
    unset = settings.model_copy(update={"gemini_api_key": None, "gemini_model": None})
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: unset)
    result = runner.invoke(app, ["ask", "Where does the quokka sleep?"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY must be set" in result.output


def test_ask_rejects_a_question_that_is_too_short(settings: Settings) -> None:
    result = runner.invoke(app, ["ask", "hi", "--fake"])
    assert result.exit_code == 1
    assert "Invalid question" in result.output


def test_ask_no_rag_needs_no_embedding_and_cites_nothing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("no_rag must not build an embedder")

    monkeypatch.setattr(grounded.retrieval.query_embedding, "GeminiEmbedder", fail)
    result = runner.invoke(
        app, ["ask", "Where does the quokka sleep?", "--fake", "--mode", "no_rag"]
    )
    assert result.exit_code == 0, result.output
    body = AskResponse.model_validate(json.loads(result.stdout))
    assert body.status == "answered"
    assert body.citations == []
    assert all(claim.citations == [] for claim in body.claims)
    assert body.meta.prompt_version.startswith("answer_no_rag_v1@")
    assert body.meta.index_version == "none"
    assert "[c1]" not in body.answer_markdown


def test_ask_rejects_an_unknown_mode(settings: Settings) -> None:
    result = runner.invoke(
        app, ["ask", "Where does the quokka sleep?", "--fake", "--mode", "dense"]
    )
    assert result.exit_code != 0
