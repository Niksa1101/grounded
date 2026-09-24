"""CLI wiring: `grounded serve` runs uvicorn safely, `grounded migrate` fails cleanly."""

from __future__ import annotations

import sys
from typing import Any

import pytest
import uvicorn
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
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
