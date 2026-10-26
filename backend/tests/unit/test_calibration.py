"""Confidence calibration (8.04, Tech §9.8): the bucket rule, the counts, the monotonicity flag,
reading a promptfoo results file and the `grounded eval calibration` command.

The counting is tested on outcomes built here; the reading and the command on the committed
promptfoo recording (scripted models: it checks the mechanics, not any result). The last tests pin
that the README block is the committed `eval/calibration/v1.md` (AGENTS.md §7: no number typed by
hand). No network, no database.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from grounded.cli import app
from grounded.evals.calibration import (
    CALIBRATION_DIR,
    CalibrationError,
    ClaimOutcome,
    Outcome,
    bucket_of,
    build_report,
    calibrate,
    published_header,
)

FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample_judge.json"
)
README = Path(__file__).resolve().parents[3] / "README.md"
PUBLISHED = CALIBRATION_DIR / "v1.md"
START, END = "<!-- calibration-report:start -->", "<!-- calibration-report:end -->"
runner = CliRunner()


def claim(
    confidence: float,
    outcome: Outcome = "supported",
    *,
    decided_locally: bool = False,
    cache_hit: bool = False,
) -> ClaimOutcome:
    return ClaimOutcome(confidence, outcome, decided_locally=decided_locally, cache_hit=cache_hit)


# --- Buckets -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("confidence", "bucket"),
    [
        (0.0, "low"),
        (0.4999, "low"),
        (0.5, "mid"),  # the lower edge belongs to mid
        (0.65, "mid"),
        (0.8, "mid"),  # so does the upper edge
        (0.8001, "high"),
        (1.0, "high"),
    ],
)
def test_bucket_edges(confidence: float, bucket: str) -> None:
    assert bucket_of(confidence) == bucket


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
def test_a_confidence_outside_0_1_is_refused(confidence: float) -> None:
    with pytest.raises(CalibrationError, match="not in"):
        bucket_of(confidence)


# --- Counting ------------------------------------------------------------------------------------


def test_counts_and_rates_per_bucket() -> None:
    cal = calibrate(
        [
            claim(0.1, "not_supported", decided_locally=True),
            claim(0.3, "supported"),
            claim(0.5, "supported", cache_hit=True),
            claim(0.8, "not_supported"),
            claim(0.9, "supported"),
            claim(0.95, "supported"),
        ]
    )
    low, mid, high = cal.buckets
    assert (low.n, low.supported, low.decided_locally, low.rate) == (2, 1, 1, 0.5)
    assert (mid.n, mid.supported, mid.rate) == (2, 1, 0.5)
    assert (high.n, high.supported, high.rate) == (2, 2, 1.0)
    assert (cal.overall.name, cal.overall.n, cal.overall.supported) == ("all", 6, 4)
    assert cal.cache_hits == 1
    assert cal.monotonic is True  # 0.5, 0.5, 1.0: equal rates do not break it


def test_answers_counts_the_distinct_answers_behind_n() -> None:
    cal = calibrate(
        [
            ClaimOutcome(0.9, "supported", answer="q1"),
            ClaimOutcome(0.95, "supported", answer="q1"),
            ClaimOutcome(0.85, "not_supported", answer="q2"),
            ClaimOutcome(0.9, "judge_failed", answer="q3"),  # no verdict: not behind n
            ClaimOutcome(0.6, "supported", answer="q1"),
        ]
    )
    assert (cal.buckets[2].n, cal.buckets[2].answers) == (3, 2)
    assert (cal.buckets[1].n, cal.buckets[1].answers) == (1, 1)
    assert (cal.overall.n, cal.overall.answers) == (4, 2)


def test_an_empty_bucket_has_no_rate_and_is_skipped_by_the_monotonic_check() -> None:
    cal = calibrate([claim(0.6, "not_supported"), claim(0.9, "supported")])
    assert cal.buckets[0].n == 0
    assert cal.buckets[0].rate is None
    assert cal.monotonic is True


def test_judge_failures_and_unjudged_claims_are_left_out_of_n_and_counted() -> None:
    cal = calibrate(
        [
            claim(0.9, "supported"),
            claim(0.9, "judge_failed", cache_hit=True),
            claim(0.6, "judge_failed"),
            claim(0.15, "not_judged"),
            claim(0.15, "not_judged"),
        ]
    )
    low, mid, high = cal.buckets
    assert (low.n, low.not_judged, low.judge_failed) == (0, 2, 0)
    assert (mid.n, mid.judge_failed) == (0, 1)
    assert (high.n, high.supported, high.judge_failed) == (1, 1, 1)
    assert (cal.overall.n, cal.overall.judge_failed, cal.overall.not_judged) == (1, 2, 2)
    assert cal.overall.rate == 1.0
    assert cal.cache_hits == 0  # a failed call is not a replayed verdict


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([], None),
        ([claim(0.9)], None),  # one bucket with claims: nothing to compare
        ([claim(0.1, "not_supported"), claim(0.9)], True),
        ([claim(0.1), claim(0.9, "not_supported")], False),
        ([claim(0.1), claim(0.6, "not_supported"), claim(0.9)], False),  # a dip in the middle
        # mid 2/3, high 0/1: a fall even though low is empty
        ([claim(0.6), claim(0.6), claim(0.6, "not_supported"), claim(0.9, "not_supported")], False),
    ],
)
def test_monotonic_flag(outcomes: list[ClaimOutcome], expected: bool | None) -> None:
    assert calibrate(outcomes).monotonic is expected


def test_the_result_does_not_depend_on_the_order_of_the_claims() -> None:
    outcomes: list[ClaimOutcome] = [
        claim(i / 20, ("supported", "not_supported", "judge_failed", "not_judged")[i % 4])
        for i in range(21)
    ]
    shuffled = outcomes[:]
    random.Random(804).shuffle(shuffled)
    assert calibrate(shuffled) == calibrate(outcomes)
    assert calibrate(outcomes) == calibrate(outcomes)


# --- Reading a results file ----------------------------------------------------------------------


def test_reads_the_recorded_run() -> None:
    """hybrid in the recording: q003 S 0.84 + NS 0.87, q007 S 0.84 + NS decided locally 0.12,
    q008 one errored claim + S 0.76, q015 S 0.79, q045 a refusal with no claim, q047 and q049 an
    errored faithfulness result with no per-claim record (one claim each, 0.76-ish)."""
    report = build_report(FIXTURE.read_bytes(), source_name=FIXTURE.name)
    assert [c.identity.config for c in report.configs] == ["hybrid", "no_rag"]  # hybrid first
    hybrid = report.configs[0]
    assert hybrid.answers == 7
    low, mid, high = hybrid.calibration.buckets
    assert (low.n, low.answers, low.supported, low.decided_locally) == (1, 1, 0, 1)
    assert (mid.n, mid.answers, mid.supported, mid.judge_failed) == (2, 2, 2, 3)
    assert (high.n, high.answers, high.supported) == (3, 2, 2)  # q003 gives two of the three
    assert (hybrid.calibration.overall.n, hybrid.calibration.overall.answers) == (6, 4)
    assert hybrid.calibration.monotonic is False
    no_rag = report.configs[1].calibration
    assert (no_rag.overall.n, no_rag.overall.not_judged, no_rag.monotonic) == (0, 7, None)


def test_configs_can_be_picked_and_ordered() -> None:
    report = build_report(FIXTURE.read_bytes(), source_name="x", configs=["no_rag", "no_rag"])
    assert [c.identity.config for c in report.configs] == ["no_rag"]


def _hybrid_q003(document: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The answer's claims and the judge's per-claim records of hybrid q003 (two claims)."""
    for row in document["results"]["results"]:
        if row["provider"]["label"] == "hybrid" and row["metadata"]["golden"]["id"] == "q003":
            for component in row["gradingResult"]["componentResults"]:
                if component["assertion"]["metric"] == "faithfulness":
                    return row["metadata"]["claims"], component["judge"]["claims"]
    raise AssertionError("no hybrid q003 in the fixture")


def _changed(tmp_path: Path, change: str) -> Path:
    document = json.loads(FIXTURE.read_bytes())
    answer, records = _hybrid_q003(document)
    if change == "verdict":
        records[0]["verdict"] = "PARTIALLY_SUPPORTED"
    elif change == "confidence":  # out of range on both sides, so only the range check trips
        records[0]["confidence"] = answer[0]["confidence"] = 1.5
    elif change == "mismatch":
        records[0]["confidence"] = 0.123
    elif change == "index":
        records[1]["claim_index"] = 0
    elif change == "count":
        records.pop()
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("verdict", "unknown verdict 'PARTIALLY_SUPPORTED'"),
        ("confidence", "hybrid q003 claim 0: confidence 1.5 is not in"),
        ("mismatch", r"claim 0: the judge record has confidence 0\.123, the answer 0\.8"),
        ("index", r"claim indices \[0, 0\] are not 0\.\.1, once each"),
        ("count", "the judge recorded 1 claim"),
    ],
)
def test_bad_input_is_refused(tmp_path: Path, change: str, message: str) -> None:
    with pytest.raises(CalibrationError, match=message):
        build_report(_changed(tmp_path, change).read_bytes(), source_name="x")


# --- The command ---------------------------------------------------------------------------------


def test_the_command_prints_the_tables_and_writes_them_with_out(tmp_path: Path) -> None:
    out = tmp_path / "calibration" / "v9.md"
    result = runner.invoke(
        app,
        [
            "eval",
            "calibration",
            "--results",
            str(FIXTURE),
            "--git-sha",
            "abc123",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "git abc123" in result.output
    assert "| mid | 0.5 to 0.8 | 2 | 2 | 2 | 1.000 | 0 | 3 | 0 |" in result.output
    assert "| low | < 0.5 | 0 | 0 | 0 | — | 0 | 0 | 7 |" in result.output
    assert "Monotonic: no (over low, mid, high)." in result.output
    header, _, body = out.read_bytes().decode("utf-8").partition("\n\n")
    assert header.startswith("> Generated, do not edit: the output of `uv run grounded eval")
    expected = f"--results ../eval/results/{FIXTURE.name} --git-sha abc123 --out <dir>/v9.md`"
    assert expected in header
    assert str(tmp_path) not in header  # no local path leaks into a committed file
    assert str(FIXTURE.parent) not in header
    assert body == result.output


def test_the_header_names_an_output_inside_the_repository_from_backend() -> None:
    header = published_header("run.json", CALIBRATION_DIR / "v1.md", configs=["hybrid"])
    assert (
        "`uv run grounded eval calibration --results ../eval/results/run.json --config hybrid "
        "--out ../eval/calibration/v1.md`"
    ) in header


@pytest.mark.parametrize(
    "args",
    [
        ["--results", "does-not-exist.json"],
        ["--results", str(FIXTURE), "--config", "hybrid_rerank"],
    ],
)
def test_the_command_refuses_with_exit_1(args: list[str]) -> None:
    result = runner.invoke(app, ["eval", "calibration", *args])
    assert result.exit_code == 1
    assert "Cannot calibrate" in result.output


def test_the_command_refuses_a_file_that_is_not_a_promptfoo_run(tmp_path: Path) -> None:
    path = tmp_path / "not-a-run.json"
    path.write_text('{"results": {}}', encoding="utf-8")
    result = runner.invoke(app, ["eval", "calibration", "--results", str(path)])
    assert result.exit_code == 1
    assert "not a promptfoo results file" in result.output


# --- The published report ------------------------------------------------------------------------


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def test_the_published_report_is_a_header_and_the_output() -> None:
    header, _, body = _text(PUBLISHED).partition("\n\n")
    assert header.startswith("> Generated, do not edit")
    assert "\n" not in header
    assert body.startswith("Confidence calibration of `20261008T155539Z-generation.json`")


def test_the_readme_block_is_the_published_report() -> None:
    readme = _text(README)
    assert readme.count(START) == 1
    assert readme.count(END) == 1
    block = readme.split(START, 1)[1].split(END, 1)[0]
    body = _text(PUBLISHED).partition("\n\n")[2]
    assert block.strip() == body.strip(), (
        f"README.md is out of date: paste eval/calibration/v1.md (without its header) between "
        f"{START} and {END}"
    )
