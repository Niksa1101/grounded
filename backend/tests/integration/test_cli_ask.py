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
from grounded.cli import app
from grounded.ingest.embed import FakeEmbedder, TokenCounter
from grounded.schemas.api import AskResponse
from grounded.settings import Settings
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import EMBEDDING_DIM, insert_index_version, make_settings

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


def test_ask_without_a_provider_adapter_fails_cleanly(settings: Settings) -> None:
    result = runner.invoke(app, ["ask", "Where does the quokka sleep?"])
    assert result.exit_code == 1
    assert "no adapter for provider 'gemini'" in result.output


def test_ask_rejects_a_question_that_is_too_short(settings: Settings) -> None:
    result = runner.invoke(app, ["ask", "hi", "--fake"])
    assert result.exit_code == 1
    assert "Invalid question" in result.output
