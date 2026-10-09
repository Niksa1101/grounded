"""The recorded judge-human agreement (ticket 4.11b, Tech.md §15.4): the Author's labels, the key
and the published output are one set of files.

These tests read the committed files and call no judge and no network: ``grounded eval agreement``
only reads the sheet and the key. What they pin is that nothing in the repository states a number
the command does not print for the committed labels and key (AGENTS.md §7: no number is typed by
hand). If a label or the key changes, they fail until the output is regenerated and pasted again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from grounded.cli import app
from grounded.evals.agreement import AGREEMENT_DIR, DEFAULT_SHEET

runner = CliRunner()
README = Path(__file__).resolve().parents[3] / "README.md"
KEY = AGREEMENT_DIR / "v1.key.json"
PUBLISHED = AGREEMENT_DIR / "v1.agreement.md"
START, END = "<!-- agreement-report:start -->", "<!-- agreement-report:end -->"


def read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


@pytest.fixture(scope="module")
def output() -> str:
    """What the command prints for the committed labels and the committed key."""
    result = runner.invoke(
        app, ["eval", "agreement", "--labels", str(DEFAULT_SHEET), "--key", str(KEY)]
    )
    assert result.exit_code == 0, result.output
    return result.output


def test_the_committed_labels_and_key_give_a_complete_report(output: str) -> None:
    """Exit 0 means the join found a label for every key item and a judge verdict for each of them:
    the command refuses a missing, invalid or unknown label instead of reporting a smaller n."""
    assert output.startswith("# Judge-human agreement\n")
    assert "| All items (labels of both kinds pooled) | 20 |" in output


def test_the_published_output_is_what_the_command_prints(output: str) -> None:
    """``v1.agreement.md`` is a one-line header and then the command's output, verbatim."""
    header, _, body = read_text(PUBLISHED).partition("\n\n")
    assert header.startswith("> Generated, do not edit")
    assert "\n" not in header
    assert body == output, (
        "eval/judge_agreement/v1.agreement.md is out of date: regenerate it with `uv run grounded "
        "eval agreement --labels ../eval/judge_agreement/v1.csv --key "
        "../eval/judge_agreement/v1.key.json` (in backend/) under its header"
    )


def test_the_readme_block_is_what_the_command_prints(output: str) -> None:
    readme = read_text(README)
    assert readme.count(START) == 1
    assert readme.count(END) == 1
    block = readme.split(START, 1)[1].split(END, 1)[0]
    assert block.strip() == output.strip(), (
        f"README.md is out of date: paste the output of `uv run grounded eval agreement` between "
        f"{START} and {END}"
    )
