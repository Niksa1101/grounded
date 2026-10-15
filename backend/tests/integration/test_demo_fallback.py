"""``grounded demo-fallback`` end to end (7.04b): the real app over the local test database, with
the fakes the command builds itself. The pytest socket guard stands behind "no real API is called";
the settings below even name a Cohere reranker and keys the demo must ignore."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, LiteralString

import psycopg
import pytest
from psycopg.rows import dict_row
from typer.testing import CliRunner

import grounded.cli
import grounded.runtime
from grounded.cli import app
from grounded.settings import Settings
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import (
    EMBEDDING_DIM,
    EVAL_SETTINGS,
    insert_index_version,
    make_settings,
    pricing_for_fake_embedder,
)

pytestmark = pytest.mark.integration

runner = CliRunner()

GEMINI_MODEL = "gemini-3.5-flash-lite"  # priced in pricing.toml
GROQ_MODEL = "openai/gpt-oss-120b"


@pytest.fixture
def database(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    yield test_database_url
    wipe()


def use_settings(
    monkeypatch: pytest.MonkeyPatch, url: str, tmp_path: Path, **overrides: Any
) -> Settings:
    settings = make_settings(
        database_url=url,
        cache_dir=tmp_path / "cache",
        embedding_model="fake-embedding",
        embedding_dim=EMBEDDING_DIM,
        gemini_model=GEMINI_MODEL,
        groq_model=GROQ_MODEL,
        **overrides,
    )
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.runtime, "load_pricing", pricing_for_fake_embedder)
    return settings


def table(url: str, query: LiteralString) -> list[dict[str, Any]]:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        return cur.fetchall()


def rows(url: str) -> list[dict[str, Any]]:
    return table(url, "SELECT * FROM request_logs ORDER BY created_at, id")


def test_the_demo_shows_the_fallback_the_breaker_skip_and_the_trial(
    database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(monkeypatch, database, tmp_path)

    result = runner.invoke(app, ["demo-fallback"])

    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    first, second, third = (
        next(line for line in lines if line.startswith(f"request {n}:")) for n in (1, 2, 3)
    )
    # 1: Gemini's 429 is answered by Groq.
    assert "HTTP 200" in first
    assert f"provider=groq  model={GROQ_MODEL}  fallback_used=true" in first
    assert "calls so far: gemini=1 groq=1" in first
    # 2: the breaker is open, Gemini is not called again.
    assert "provider=groq" in second
    assert "fallback_used=true" in second
    assert "calls so far: gemini=1 groq=2" in second
    # 3: past the open time one trial reaches Gemini, which answers.
    assert f"provider=gemini  model={GEMINI_MODEL}  fallback_used=false" in third
    assert "calls so far: gemini=2 groq=2" in third
    assert lines[-1].startswith("PASS:")

    # The service's own log lines are printed: the breaker opening, the skip, the summary with
    # ``fallback_used``.
    log_lines = [line for line in lines if line.startswith("   log ")]
    assert any('"msg": "breaker opened"' in line and '"open_s": 30.0' in line for line in log_lines)
    assert any(
        '"msg": "provider skipped"' in line and '"breaker_open"' in line for line in log_lines
    )
    summaries = [line for line in log_lines if '"msg": "request completed"' in line]
    assert [('"fallback_used": true' in s) for s in summaries] == [True, True, False]

    # And the rows say the same.
    logged = rows(database)
    assert [(r["provider"], r["fallback_used"], r["outcome"]) for r in logged] == [
        ("groq", True, "answered"),
        ("groq", True, "answered"),
        ("gemini", False, "answered"),
    ]


def test_no_question_text_is_printed(
    database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(monkeypatch, database, tmp_path)

    result = runner.invoke(app, ["demo-fallback"])

    assert result.exit_code == 0, result.output
    assert "declare a" not in result.output  # the questions are asked, never printed or logged
    assert "declare a" not in result.stderr


def test_a_rerun_asks_new_questions_and_is_not_answered_from_the_cache(
    database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(monkeypatch, database, tmp_path)

    first = runner.invoke(app, ["demo-fallback"])
    second = runner.invoke(app, ["demo-fallback"])

    assert (first.exit_code, second.exit_code) == (0, 0), second.output
    assert "cache_hit=true" not in second.stdout
    assert len(rows(database)) == 6


def test_the_demo_ignores_a_cohere_reranker_and_a_budget_in_the_settings(
    database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A developer's .env may switch the real reranker on and set a budget; the demo runs on a copy
    # with rerank off and no budget, so it makes no Cohere call (the guard would fail one) and
    # spends no unit of the day.
    use_settings(
        monkeypatch,
        database,
        tmp_path,
        rerank_provider="cohere",
        rerank_model="rerank-v4.0-pro",
        cohere_api_key="k-not-used",
        daily_llm_budget=1,
        ip_hash_secret="s" * 32,
    )

    result = runner.invoke(app, ["demo-fallback"])

    assert result.exit_code == 0, result.output
    assert all(r["rerank_used"] is False for r in rows(database))
    assert table(database, "SELECT * FROM daily_usage") == []


def test_it_is_refused_in_prod_and_writes_nothing(
    database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(
        monkeypatch,
        database,
        tmp_path,
        app_env="prod",
        proxy_shared_secret="x" * 32,
        ip_hash_secret="y" * 32,
        daily_llm_budget=300,
    )

    result = runner.invoke(app, ["demo-fallback"])

    assert result.exit_code == 1
    assert "refused with APP_ENV=prod" in result.stderr
    assert rows(database) == []


def test_it_is_refused_in_eval_mode(
    database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(monkeypatch, database, tmp_path, **EVAL_SETTINGS)

    result = runner.invoke(app, ["demo-fallback"])

    assert result.exit_code == 1
    assert "refused with APP_ENV=eval" in result.stderr


@pytest.mark.parametrize(
    ("overrides", "missing"),
    [({"gemini_model": None}, "GEMINI_MODEL"), ({"groq_model": None}, "GROQ_MODEL")],
)
def test_a_missing_model_name_is_named(
    database: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, None],
    missing: str,
) -> None:
    settings = use_settings(monkeypatch, database, tmp_path)
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings.model_copy(update=overrides))

    result = runner.invoke(app, ["demo-fallback"])

    assert result.exit_code == 1
    assert missing in result.stderr
    assert "No API key is needed" in result.stderr


def test_without_an_active_index_it_fails_and_says_what_to_do(
    database: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_settings(monkeypatch, database, tmp_path)
    with psycopg.connect(database, autocommit=True) as conn:
        conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    result = runner.invoke(app, ["demo-fallback"])

    assert result.exit_code == 1
    assert "FAIL:" in result.stdout
    assert "grounded ingest --activate" in result.stdout
