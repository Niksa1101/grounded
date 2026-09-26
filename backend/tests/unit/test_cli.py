"""CLI wiring: `grounded serve` runs uvicorn safely, `grounded migrate` fails cleanly."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.ingest.types import CorpusCheckout
from tests.support import make_settings

runner = CliRunner()

# Nothing listens on port 1, so the connection is refused locally (no external network). The timeout
# keeps Windows, where refused connects are retried slowly, from stalling the test.
_UNREACHABLE_DB = "postgresql://grounded@127.0.0.1:1/grounded_test?connect_timeout=2"


@pytest.fixture
def uvicorn_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(grounded.cli, "get_settings", make_settings)
    return calls


def test_serve_disables_access_log_and_uses_app_factory(
    uvicorn_calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    result = runner.invoke(app, ["serve", "--port", "7860"])
    assert result.exit_code == 0, result.output
    [(args, kwargs)] = uvicorn_calls
    assert args == ("grounded.main:create_app",)
    assert kwargs["factory"] is True
    assert kwargs["port"] == 7860
    # uvicorn's access log prints raw client IPs (AGENTS.md §6.13).
    assert kwargs["access_log"] is False
    # Our JSON logging stays in charge instead of uvicorn's dictConfig.
    assert kwargs["log_config"] is None


def test_serve_picks_a_psycopg_compatible_event_loop(
    uvicorn_calls: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> None:
    runner.invoke(app, ["serve"])
    [(_, kwargs)] = uvicorn_calls
    expected = "asyncio:SelectorEventLoop" if sys.platform == "win32" else "auto"
    assert kwargs["loop"] == expected


def test_migrate_reports_unreachable_database_without_traceback() -> None:
    result = runner.invoke(app, ["migrate", "--database-url", _UNREACHABLE_DB])
    assert result.exit_code == 1
    assert "Migration aborted: cannot connect" in result.output
    assert isinstance(result.exception, SystemExit)  # typer.Exit, not an unhandled psycopg error


# --- ingest (no database: dry run and configuration errors) ----------------------------------

CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"


@pytest.fixture
def mini_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """`fetch_corpus` returns corpus_mini instead of cloning."""

    def fake_fetch(ref: str, cache_dir: Path) -> CorpusCheckout:
        return CorpusCheckout(path=CORPUS_MINI, ref=ref, sha="a" * 40)

    monkeypatch.setattr(grounded.cli, "fetch_corpus", fake_fetch)


def use_settings(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: make_settings(**overrides))


@pytest.mark.usefixtures("mini_corpus")
def test_ingest_dry_run_counts_texts_to_embed_without_api_or_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_settings(
        monkeypatch,
        cache_dir=tmp_path,
        embedding_model="fake-embedding",
        database_url=_UNREACHABLE_DB,  # would fail if touched
    )
    monkeypatch.setattr(grounded.cli, "GeminiEmbedder", None)  # would fail if constructed
    result = runner.invoke(app, ["ingest", "--ref", "0.0.1", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Corpus 0.0.1 @ aaaaaaaa: 7 pages, " in result.output
    assert "Index 0.0.1@" in result.output
    assert re.search(r"Dry run: \d+ texts to embed", result.output)


def test_ingest_needs_a_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch, embedding_model="m", fastapi_ref=None)
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "pass --ref or set FASTAPI_REF" in result.output


def test_ingest_needs_an_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    use_settings(monkeypatch, embedding_model=None, fastapi_ref="0.0.1")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1
    assert "EMBEDDING_MODEL must be set" in result.output


@pytest.mark.usefixtures("mini_corpus")
def test_ingest_without_api_key_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_settings(monkeypatch, cache_dir=tmp_path, embedding_model="m", gemini_api_key=None)
    result = runner.invoke(app, ["ingest", "--ref", "0.0.1"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY and EMBEDDING_MODEL must be set" in result.output


@pytest.mark.usefixtures("mini_corpus")
def test_ingest_reports_unreachable_database_without_traceback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_settings(monkeypatch, cache_dir=tmp_path, embedding_model="m", gemini_api_key="k")
    result = runner.invoke(app, ["ingest", "--ref", "0.0.1", "--database-url", _UNREACHABLE_DB])
    assert result.exit_code == 1
    assert "Database error:" in result.output
    assert isinstance(result.exception, SystemExit)
