"""The hybrid_rerank runbook (docs/runbooks/rerank-eval.md, 6.04) names commands that exist.

Like the Neon and Vercel runbook tests: the runbook is what the Author copies, so a renamed option
or file must fail here and not on the day. Nothing runs: each `grounded` command is only parsed
(with `--help` appended, click parses every option before it runs the help, so an unknown option
or subcommand is exit 2).
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grounded.cli import app

ROOT = Path(__file__).resolve().parents[3]
RUNBOOK = (ROOT / "docs" / "runbooks" / "rerank-eval.md").read_text(encoding="utf-8")
runner = CliRunner()


def command_blocks(text: str) -> list[str]:
    """The contents of the fenced `powershell` and `bash` blocks."""
    return re.findall(r"```(?:powershell|bash)\n(.*?)```", text, flags=re.DOTALL)


def grounded_commands() -> list[list[str]]:
    commands: list[list[str]] = []
    for block in command_blocks(RUNBOOK):
        for line in block.splitlines():
            if line.startswith("uv run grounded "):
                commands.append(shlex.split(line.removeprefix("uv run grounded ")))
    return commands


def test_the_runbook_has_the_commands_of_the_steps() -> None:
    commands = grounded_commands()

    assert {c[0] for c in commands} == {"migrate", "index", "ask", "eval"}
    assert {c[1] for c in commands if c[0] == "eval"} == {"retrieval", "diff", "baseline", "report"}
    configs = [c[i + 1] for c in commands for i, arg in enumerate(c) if arg == "--config"]
    assert (
        configs.count("hybrid_rerank") == 3
    )  # the comparison run, its diff and the baseline write


@pytest.mark.parametrize("args", grounded_commands(), ids=lambda args: " ".join(args)[:60])
def test_every_grounded_command_parses(args: list[str]) -> None:
    done = runner.invoke(app, [*args, "--help"])

    assert done.exit_code == 0, done.output


def test_no_command_has_a_placeholder_to_fill_in() -> None:
    for block in command_blocks(RUNBOOK):
        assert not re.search(r"<[A-Za-z][^>\n]*>", block), block


def test_the_files_and_settings_it_names_exist() -> None:
    for name in ("promptfooconfig.rerank.yaml", "provider.py"):
        assert (ROOT / "eval" / "promptfoo" / name).is_file()
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for variable in re.findall(r"\b([A-Z][A-Z0-9_]{6,})=", RUNBOOK):
        if variable.endswith("_KEY") or variable.startswith(("PROMPTFOO_", "EVAL_QUESTION")):
            continue  # secrets, and the shell variables of the harness that are not settings
        assert variable in env_example, variable


def test_the_budget_and_the_model_are_the_decided_ones() -> None:
    assert "150 real calls" in RUNBOOK
    assert "RERANK_MODEL=rerank-v4.0-pro" in RUNBOOK
    assert "rerank-v4.0-fast" in RUNBOOK
    assert "6 seconds" in RUNBOOK  # the default of EVAL_RERANK_MIN_INTERVAL_S


def test_the_promptfoo_command_is_the_pinned_one_with_the_second_config() -> None:
    commands = [
        line
        for block in command_blocks(RUNBOOK)
        for line in block.splitlines()
        if line.startswith("npx promptfoo")
    ]

    assert commands
    for command in commands:
        assert command.startswith("npx promptfoo@0.123.1 eval -c eval/promptfoo/")
        assert "promptfooconfig.rerank.yaml" in command
        assert "-j 1 --no-cache -o eval/results/" in command
