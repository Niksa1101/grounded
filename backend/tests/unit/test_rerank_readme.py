"""The README never claims a rerank lift the committed baselines do not hold (ticket 6.05).

The section "Re-ranking: what it bought" was drafted while no `hybrid_rerank` row existed, so it
says "Lift: not yet measured" and holds no metric. Both statements are true only until the baseline
PR of 6.04b lands. This test pins the first half (no claim without a row) and makes the second half
fail loudly: once a row is committed, the draft line must be replaced by the Author's reading, or
the README would keep saying something stale. It reads files only.
"""

from __future__ import annotations

import re
from pathlib import Path

from grounded.evals.generation_results import GENERATION_BASELINE, read_generation_baseline
from grounded.evals.retrieval_runner import RETRIEVAL_BASELINE, read_baseline

README = Path(__file__).resolve().parents[3] / "README.md"
HEADING = "### Re-ranking: what it bought"
NOT_MEASURED = "**Lift: not yet measured.**"
EMPTY_BLOCK = "<!-- rerank-lift: NOT FILLED YET"


def section() -> str:
    text = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count(HEADING) == 1, f"the README needs exactly one {HEADING!r} section"
    return text.split(HEADING, 1)[1].split("\n### ", 1)[0]


def rerank_rows_are_committed() -> bool:
    retrieval = read_baseline(RETRIEVAL_BASELINE)
    generation = read_generation_baseline(GENERATION_BASELINE)
    return "hybrid_rerank" in retrieval or "hybrid_rerank" in generation


def test_the_draft_claims_no_lift_while_no_hybrid_rerank_row_is_committed() -> None:
    body = section()
    if rerank_rows_are_committed():
        return  # covered by the test below
    assert NOT_MEASURED in body
    assert EMPTY_BLOCK in body
    # A metric is printed with three decimals (`grounded eval report`, `grounded eval diff`): none
    # may appear in a section whose numbers do not exist yet.
    assert not re.findall(r"(?<![\d.])[01]\.\d{3}(?!\d)", body)


def test_the_draft_line_is_replaced_once_a_hybrid_rerank_row_is_committed() -> None:
    if not rerank_rows_are_committed():
        return
    body = section()
    assert NOT_MEASURED not in body, (
        "eval/baselines/ has a hybrid_rerank row, so the README must stop saying 'Lift: not yet "
        "measured': paste the output of `uv run grounded eval diff` and write the reading "
        "(ticket 6.05), then remove the draft line and the empty rerank-lift comment"
    )
    assert EMPTY_BLOCK not in body


def test_the_roadmap_does_not_call_phase_6_done_while_no_row_is_committed() -> None:
    text = README.read_text(encoding="utf-8")
    [row] = [line for line in text.splitlines() if line.startswith("| 6 | Re-ranking")]
    if rerank_rows_are_committed():
        return
    assert "✅" not in row
    assert "pending" in row
