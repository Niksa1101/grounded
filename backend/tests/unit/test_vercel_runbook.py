"""The Vercel deploy runbook (ticket 5.10) cannot drift from the code and config it describes.

The runbook is prose the Author follows by hand against production, so the names it tells them to
set must be the ones `Settings`, Tech §18 and the frontend actually read, the preview must hold
placeholders only, and the measurement command must have the flags it uses. No network, no Vercel.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from grounded.cli import app

ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = (ROOT / "docs" / "runbooks" / "vercel-deploy.md").read_text(encoding="utf-8")
TECH = (ROOT / "docs" / "Tech.md").read_text(encoding="utf-8")


def section(text: str, heading: str) -> str:
    """The text from ``heading`` to the next heading of the same or a higher level."""
    level = len(heading.split(" ", 1)[0])
    start = text.index(heading)
    rest = text[start + len(heading) :]
    end = re.search(rf"(?m)^#{{1,{level}}} ", rest)
    return rest[: end.start()] if end else rest


def table_names(text: str) -> list[str]:
    """The first column of the Markdown table rows that name a variable, in order."""
    return re.findall(r"(?m)^\| `([A-Z][A-Z0-9_]+)` \|", text)


def env_add_commands() -> list[str]:
    return re.findall(r"(?m)^(?:printf [^|]*\| )?vercel env add .*$", RUNBOOK)


def test_the_backend_production_table_is_the_production_env_line_of_tech_18() -> None:
    [line] = [ln for ln in TECH.splitlines() if ln.startswith("- Production env:")]
    documented = set(re.findall(r"`([A-Z][A-Z0-9_]+)(?:=[^`]*)?`", line))
    table = table_names(section(RUNBOOK, "### 3.2 Production variables"))
    assert len(table) == len(set(table)) == 9
    assert set(table) == documented


def test_the_frontend_variables_are_the_ones_the_route_and_the_env_example_use() -> None:
    table = table_names(section(RUNBOOK, "### 5.1 Production variables"))
    assert table == ["BACKEND_URL", "PROXY_SHARED_SECRET"]
    example = (ROOT / "frontend" / ".env.example").read_text(encoding="utf-8")
    route = (ROOT / "frontend" / "app" / "api" / "ask" / "route.ts").read_text(encoding="utf-8")
    for name in table:
        assert re.search(rf"(?m)^{name}=", example), name
        assert f"process.env.{name}" in route, name
    assert not re.search(r"(?m)^NEXT_PUBLIC_", example)


def test_every_env_command_names_its_environment_and_none_targets_development() -> None:
    commands = env_add_commands()
    assert commands
    for command in commands:
        assert re.search(r"vercel env add [A-Z_]+ (production|preview)\b", command), command
    assert not [c for c in commands if " development" in c]


def test_the_production_commands_match_the_production_tables() -> None:
    commands = env_add_commands()
    backend = table_names(section(RUNBOOK, "### 3.2 Production variables"))
    frontend = table_names(section(RUNBOOK, "### 5.1 Production variables"))
    in_production = re.findall(r"vercel env add ([A-Z_]+) production", "\n".join(commands))
    # DATABASE_URL and PROXY_SHARED_SECRET are in both projects' commands or tables.
    assert set(in_production) == set(backend) | set(frontend)


def test_preview_values_are_placeholders_and_the_set_starts_the_app() -> None:
    commands = [c for c in env_add_commands() if " preview " in c]
    names = {n for c in commands for n in re.findall(r"env add ([A-Z_]+) preview", c)}
    # What APP_ENV=prod demands, plus the Gemini pair the lifespan builds (Tech §18).
    assert names == {
        "APP_ENV",
        "DATABASE_URL",
        "PROXY_SHARED_SECRET",
        "IP_HASH_SECRET",
        "DAILY_LLM_BUDGET",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
    }
    for command in commands:
        assert "--no-sensitive" in command, command  # Config: readable, so the check can read it
        assert "--sensitive " not in command
    database = next(c for c in commands if "DATABASE_URL" in c)
    assert "@db.invalid:" in database  # a name that never resolves


def test_no_secret_looking_value_is_written_in_the_runbook() -> None:
    assert not re.search(r"\b[0-9a-f]{32,}\b", RUNBOOK)
    assert not re.search(r"postgres(?:ql)?://[^\s:@]+:(?!preview@|<)[^\s@]+@", RUNBOOK)
    # The secrets come from the shell or a prompt, never from an argument.
    for command in env_add_commands():
        if "--sensitive" in command and "--no-sensitive" not in command:
            assert "--value" not in command, command


def test_both_projects_use_the_region_of_d53() -> None:
    backend = json.loads((ROOT / "backend" / "vercel.json").read_text(encoding="utf-8"))
    frontend = json.loads((ROOT / "frontend" / "vercel.json").read_text(encoding="utf-8"))
    assert backend["regions"] == frontend["regions"] == ["cle1"]
    assert "cle1" in RUNBOOK


def test_the_measurement_commands_use_flags_the_command_has() -> None:
    result = CliRunner().invoke(
        app, ["measure-cold-start", "--help"], env={"COLUMNS": "200", "NO_COLOR": "1"}
    )
    # CI forces colour, and rich then splits "--frontend-url" into styled pieces.
    help_text = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    used = re.findall(r"(?m)^.*grounded measure-cold-start (.*?)(?: \| tee .*)?$", RUNBOOK)
    assert len(used) == 2
    for arguments in used:
        for flag in re.findall(r"(--[a-z-]+)", arguments):
            assert flag in help_text, flag
    assert {"--first health", "--first ask"} <= {
        m for arguments in used for m in re.findall(r"--first \w+", arguments)
    }


def test_the_checklist_ids_run_without_gaps_and_the_ticket_items_are_covered() -> None:
    checklist = section(RUNBOOK, "## 8. Verify on the first deployment")
    ids = re.findall(r"(?m)^\| V(\d+) \|", checklist)
    assert ids == [str(n) for n in range(1, len(ids) + 1)]
    for needle in (
        "client IP",  # 5.03
        "X-Forwarded-For",  # 5.06
        "prepare_threshold",  # 5.08
        "CACHE_DIR",
        "bundle size",
        "maxDuration",
        "Fluid",
        "/readyz",
    ):
        assert needle in checklist, needle
    # The prose that sends the reader to the table says how many items there are.
    assert f"V1{chr(0x2013)}V{len(ids)}" in RUNBOOK  # an en dash, as the runbook writes it


def test_tech_18_points_to_the_runbook() -> None:
    assert "docs/runbooks/vercel-deploy.md" in section(TECH, "## 18. Deployment")
