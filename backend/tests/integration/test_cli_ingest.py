"""`grounded ingest` and `grounded index list` end to end, with a fake provider and corpus_mini."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.infra.provider_errors import ProviderRateLimited
from grounded.ingest.embed import FakeEmbedder, TaskType, TokenCounter, Vector
from grounded.ingest.types import CorpusCheckout
from grounded.settings import Settings
from tests.support import make_settings

pytestmark = pytest.mark.integration

runner = CliRunner()
CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"


class StubGemini(FakeEmbedder):
    """Stands in for GeminiEmbedder in the CLI: fake vectors, counted calls."""

    quota_after: int | None = None  # texts allowed before a daily-quota 429

    def __init__(self, *, model: str, dim: int) -> None:
        super().__init__(model=model, dim=dim)
        self.api_calls = 0

    @classmethod
    def from_settings(cls, settings: Settings, count_tokens: TokenCounter) -> StubGemini:
        assert settings.embedding_model is not None
        return cls(model=settings.embedding_model, dim=settings.embedding_dim)

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        sent = sum(len(batch) for batch, _ in self.calls)
        if self.quota_after is not None and sent + len(texts) > self.quota_after:
            raise ProviderRateLimited("quota", retry_after_s=None, is_quota=True)
        self.api_calls += 1
        return await super().embed(texts, task_type)


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    yield test_database_url
    wipe()


@pytest.fixture
def cli_env(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fetch(ref: str, cache_dir: Path) -> CorpusCheckout:
        return CorpusCheckout(path=CORPUS_MINI, ref=ref, sha="a" * 40)

    settings = make_settings(
        database_url_direct=db,
        cache_dir=tmp_path,
        embedding_model="fake-embedding",
        embedding_batch_size=3,
        fastapi_ref="0.0.1",
    )
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.cli, "fetch_corpus", fake_fetch)
    monkeypatch.setattr(grounded.cli, "GeminiEmbedder", StubGemini)
    monkeypatch.setattr(StubGemini, "quota_after", None)


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


@pytest.mark.usefixtures("cli_env")
def test_ingest_builds_activates_and_then_reuses() -> None:
    first = invoke("ingest", "--activate")
    assert first.exit_code == 0, first.output
    assert "Stored index version 1 (ready)." in first.output
    assert "from cache" in first.output
    assert "Active: yes." in first.output

    again = invoke("ingest")
    assert again.exit_code == 0, again.output
    assert "Index version 1 with this config is already built." in again.output
    assert "Active: yes." in again.output

    listing = invoke("index", "list")
    assert listing.exit_code == 0, listing.output
    [header, row] = listing.output.strip().splitlines()
    assert header.split()[:3] == ["id", "ref", "sha"]
    assert row.split()[:6] == ["1", "0.0.1", "aaaaaaaa", row.split()[3], "ready", "*"]


@pytest.mark.usefixtures("cli_env")
def test_ingest_quota_stop_explains_how_to_resume(db: str) -> None:
    StubGemini.quota_after = 3
    result = invoke("ingest")
    assert result.exit_code == 1
    assert "daily embedding quota reached: 3 of " in result.output
    assert "Nothing was written to the database" in result.output
    with psycopg.connect(db) as conn:
        assert conn.execute("SELECT count(*) FROM index_versions").fetchone() == (0,)

    StubGemini.quota_after = None
    resumed = invoke("ingest")
    assert resumed.exit_code == 0, resumed.output
    assert "3 from cache" in resumed.output


@pytest.mark.usefixtures("cli_env")
def test_index_list_when_empty() -> None:
    result = invoke("index", "list")
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "No index versions."
