"""Housekeeping without a database: the SQL cannot drift from DB.md §9, and the command is wired.

The deleting statements are the privacy contract (NFR-4), so the module and the document that
states the contract must say the same thing. The behavior itself is proven by
``tests/integration/test_housekeeping.py`` against the real schema.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.infra import housekeeping
from grounded.infra.housekeeping import (
    RULES,
    RuleResult,
    render_results,
)
from tests.support import make_settings

ROOT = Path(__file__).resolve().parents[3]
DB_MD = (ROOT / "docs" / "DB.md").read_text(encoding="utf-8")

runner = CliRunner()
_UNREACHABLE_DB = "postgresql://grounded@127.0.0.1:1/grounded_test?connect_timeout=2"


def normalized(sql: str) -> list[str]:
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    return [" ".join(part.split()) for part in code.split(";") if part.strip()]


def section_nine() -> str:
    start = DB_MD.index("## 9. Retention and housekeeping")
    return DB_MD[start : DB_MD.index("## 10. Migrations")]


def test_the_module_runs_exactly_the_statements_of_db_md_section_9() -> None:
    block = re.search(r"```sql\n(.*?)```", section_nine(), re.DOTALL)
    assert block is not None

    assert normalized(block.group(1)) == [normalized(rule.sql)[0] for rule in RULES]


def test_the_parameter_table_of_db_md_matches_the_constants() -> None:
    rows = dict(re.findall(r"(?m)^\| `(\w+)` \| (\d+) \|", section_nine()))
    values = {name: str(value) for rule in RULES for name, value in rule.params.items()}

    assert rows == values
    assert values == {
        "text_days": "30",
        "telemetry_days": "180",
        "usage_days": "90",
        "window_days": "2",
    }


def test_every_statement_binds_its_parameters_and_builds_nothing_from_input() -> None:
    for rule in RULES:
        assert set(re.findall(r"%\((\w+)\)s", rule.sql)) == set(rule.params), rule.name
        assert "{" not in rule.sql  # no f-string or format hole (AGENTS.md §6.11)
    assert len({rule.name for rule in RULES}) == len(RULES)


def test_the_output_has_one_aligned_line_per_rule_and_a_total() -> None:
    results = [RuleResult("short", 1), RuleResult("a_longer_name", 12)]

    assert render_results(results, dry_run=False) == [
        "  short          1",
        "  a_longer_name  12",
        "Done: 13 rows affected.",
    ]
    assert render_results(results, dry_run=True)[-1] == (
        "Dry run: 13 rows would be affected; nothing was changed."
    )


# --- the command ---------------------------------------------------------------------------------


def test_the_command_reports_an_unreachable_database_without_a_traceback() -> None:
    result = runner.invoke(app, ["housekeeping", "--database-url", _UNREACHABLE_DB])

    assert result.exit_code == 1
    assert "Housekeeping failed, nothing was changed" in result.output
    assert isinstance(result.exception, SystemExit)  # typer.Exit, not an unhandled psycopg error


def test_the_command_uses_the_owner_url_and_passes_dry_run_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_run(conninfo: str, **kwargs: Any) -> list[RuleResult]:
        calls.append((conninfo, kwargs))
        return [RuleResult("rule_a", 0)]

    settings = make_settings(
        database_url="postgresql://app@127.0.0.1:1/runtime",
        database_url_direct="postgresql://owner@127.0.0.1:1/owner",
    )
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.cli, "run_housekeeping", fake_run)

    plain = runner.invoke(app, ["housekeeping"])
    dry = runner.invoke(app, ["housekeeping", "--dry-run"])

    assert plain.exit_code == 0
    assert dry.exit_code == 0
    # The owner connection, never the runtime's `app` role (which cannot delete the logs).
    assert calls == [
        ("postgresql://owner@127.0.0.1:1/owner", {"dry_run": False}),
        ("postgresql://owner@127.0.0.1:1/owner", {"dry_run": True}),
    ]
    assert "Housekeeping on 127.0.0.1:1/owner" in plain.output
    assert "owner@" not in plain.output  # the target is host:port/db, no user or password


def test_the_periods_are_the_prd_contract() -> None:
    # NFR-4 / DB.md §9. Not Settings fields: a setting could only lengthen the privacy retention.
    assert housekeeping.USER_TEXT_RETENTION_DAYS == 30
    assert housekeeping.TELEMETRY_RETENTION_DAYS == 180
