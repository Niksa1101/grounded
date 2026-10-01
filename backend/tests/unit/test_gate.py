"""The retrieval gate (Tech.md §15.5, ticket 2.09).

The spec tests of ``evaluate_gate`` (ticket 2.10) state *behavior* (a table of baselines, runs and
verdicts), not the algorithm. The rest is boilerplate: the baseline schema, the Markdown report, the
exit code and the CLI wrapper, with ``evaluate_gate`` replaced by a scripted verdict.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.gate import GateReport, GateRow, evaluate_gate, exit_code, render_markdown
from grounded.evals.retrieval_runner import (
    RETRIEVAL_BASELINE,
    RetrievalEvalError,
    metric_names,
    read_baseline,
    update_baseline,
    write_run,
)
from grounded.retrieval.config import RetrievalConfig, RetrievalMode
from grounded.schemas.eval import (
    MetricThreshold,
    RetrievalBaselineEntry,
    RetrievalConfigResult,
    RetrievalRun,
    RetrievalRunInfo,
)

NOW = datetime(2026, 10, 1, 12, 0, 0, tzinfo=UTC)
GATED = ("recall@5", "mrr", "ndcg@5")
TOLERANCE = 0.04

runner = CliRunner()
HEADLINE = {"pass": "✅", "inconclusive": "⚠️", "fail": "❌"}

# --- Builders ------------------------------------------------------------------------------------


def retrieval_config(mode: RetrievalMode) -> RetrievalConfig:
    return RetrievalConfig.model_validate(
        {
            "mode": mode,
            "k_dense": 20,
            "k_fts": 20,
            "k_fused": 40,
            "k_context": 5,
            "rrf_k": 60,
        }
    )


def all_metrics(value: float = 0.7, **overrides: float) -> dict[str, float]:
    return dict.fromkeys(metric_names(), value) | overrides


def entry(
    metrics: dict[str, float] | None = None,
    *,
    gated: tuple[str, ...] = GATED,
    tolerance: float = TOLERANCE,
    **overrides: Any,
) -> RetrievalBaselineEntry:
    """A baseline row. ``gated`` names the thresholded metrics (none: a reported-only row)."""
    fields: dict[str, Any] = {
        "metrics": metrics if metrics is not None else all_metrics(),
        "thresholds": {name: MetricThreshold(tolerance=tolerance) for name in gated},
        "n": 25,
        "k": 40,
        "golden_set_version": "v1",
        "golden_set_sha256": "a" * 64,
        "index_config_hash": "f" * 64,
        "fastapi_ref": "0.141.1",
        "fastapi_sha": "9" * 40,
        "embedding_model": "gemini-embedding-001",
        "embedding_dim": 768,
        "git_sha": "c" * 40,
        "git_dirty": False,
        "date": NOW,
    }
    return RetrievalBaselineEntry.model_validate(fields | overrides)


def make_run(
    configs: dict[RetrievalMode, dict[str, float]],
    *,
    n: int = 30,
    overrides: dict[str, Any] | None = None,
) -> RetrievalRun:
    info = RetrievalRunInfo.model_validate(
        {
            "date": NOW,
            "git_sha": "d" * 40,
            "git_dirty": False,
            "golden_set_version": "v1",
            "golden_set_sha256": "a" * 64,
            "index_version_id": 3,
            "index_config_hash": "f" * 64,
            "fastapi_ref": "0.141.1",
            "fastapi_sha": "9" * 40,
            "embedding_model": "gemini-embedding-001",
            "embedding_dim": 768,
        }
        | (overrides or {})
    )
    results = {
        mode: RetrievalConfigResult(
            config=mode,
            retrieval_config=retrieval_config(mode),
            retrieval_config_hash=retrieval_config(mode).config_hash,
            k=40,
            n=n,
            skipped_unanswerable=5,
            metrics=metrics,
            questions=[],
        )
        for mode, metrics in configs.items()
    }
    return RetrievalRun(info=info, configs=results)


def rows_of(report: GateReport, config: str) -> dict[str, GateRow]:
    return {row.metric: row for row in report.rows if row.config == config}


def reason_mentioning(report: GateReport, word: str) -> bool:
    return any(word in reason for reason in report.reasons)


# --- Spec: evaluate_gate (ticket 2.10) --------------------------------------------


def test_a_run_equal_to_the_baseline_passes() -> None:
    report = evaluate_gate(make_run({"hybrid": all_metrics()}), {"hybrid": entry()})
    assert report.status == "pass"
    assert report.reasons == []
    gated = rows_of(report, "hybrid")
    assert {name for name, row in gated.items() if row.passed} == set(GATED)


def test_a_run_better_than_the_baseline_passes() -> None:
    report = evaluate_gate(
        make_run({"hybrid": all_metrics(0.9)}), {"hybrid": entry(all_metrics(0.7))}
    )
    assert report.status == "pass"
    assert rows_of(report, "hybrid")["mrr"].delta == pytest.approx(0.2)


# (baseline, tolerance, current). The decimal cases are float-hostile: baseline - tolerance is
# 0.5399999999999999 and 0.6599999999999999, yet a drop of exactly the tolerance must pass.
AT_TOLERANCE = [
    pytest.param(0.75, 0.25, 0.5, id="binary-exact"),
    pytest.param(0.58, 0.04, 0.54, id="decimal-0.58"),
    pytest.param(0.70, 0.04, 0.66, id="decimal-0.70"),
    pytest.param(0.74, 0.0, 0.74, id="zero-tolerance-equal"),
]
JUST_PAST = [
    pytest.param(0.75, 0.25, 0.49, id="binary"),
    pytest.param(0.58, 0.04, 0.53, id="decimal-0.58"),
    pytest.param(0.70, 0.04, 0.65, id="decimal-0.70"),
    pytest.param(0.74, 0.0, 0.73, id="zero-tolerance-drop"),
]


@pytest.mark.parametrize(("baseline", "tolerance", "current"), AT_TOLERANCE)
def test_a_drop_of_exactly_the_tolerance_passes(
    baseline: float, tolerance: float, current: float
) -> None:
    row = entry(all_metrics(baseline), tolerance=tolerance)
    report = evaluate_gate(make_run({"hybrid": all_metrics(current)}), {"hybrid": row})
    assert report.status == "pass"
    assert all(row.passed for name, row in rows_of(report, "hybrid").items() if name in GATED)


@pytest.mark.parametrize(("baseline", "tolerance", "current"), JUST_PAST)
def test_a_drop_just_past_the_tolerance_fails(
    baseline: float, tolerance: float, current: float
) -> None:
    row = entry(all_metrics(baseline), tolerance=tolerance)
    report = evaluate_gate(make_run({"hybrid": all_metrics(current)}), {"hybrid": row})
    assert report.status == "fail"
    assert not any(row.passed for name, row in rows_of(report, "hybrid").items() if name in GATED)


@pytest.mark.parametrize("metric", GATED)
def test_one_failing_metric_fails_the_suite(metric: str) -> None:
    current = all_metrics(0.7, **{metric: 0.5})
    report = evaluate_gate(make_run({"hybrid": current}), {"hybrid": entry(all_metrics(0.7))})
    assert report.status == "fail"
    rows = rows_of(report, "hybrid")
    assert [name for name, row in rows.items() if row.passed is False] == [metric]


def test_rows_carry_baseline_current_delta_threshold_and_n() -> None:
    report = evaluate_gate(
        make_run({"hybrid": all_metrics(0.5)}, n=30), {"hybrid": entry(all_metrics(0.75))}
    )
    row = rows_of(report, "hybrid")["recall@5"]
    assert row.baseline == pytest.approx(0.75)
    assert row.current == pytest.approx(0.5)
    assert row.delta == pytest.approx(-0.25)
    assert row.threshold == pytest.approx(0.75 - TOLERANCE)
    assert row.n == 30  # the run's n, not the baseline's 25


def test_only_metrics_with_a_threshold_are_gated() -> None:
    # recall@10 collapses, but the baseline only gates the three headline metrics.
    current = all_metrics(0.7, **{"recall@10": 0.0})
    report = evaluate_gate(make_run({"hybrid": current}), {"hybrid": entry(all_metrics(0.7))})
    assert report.status == "pass"
    unrated = rows_of(report, "hybrid").get("recall@10")
    assert unrated is None or unrated.passed is None


def test_a_gated_config_missing_from_the_results_fails_with_its_name() -> None:
    baseline = {"hybrid": entry(), "dense": entry(gated=())}
    report = evaluate_gate(make_run({"dense": all_metrics()}), baseline)
    assert report.status == "fail"
    assert reason_mentioning(report, "hybrid")


def test_a_gated_metric_missing_from_the_run_fails_with_the_config_name() -> None:
    current = {name: value for name, value in all_metrics().items() if name != "mrr"}
    report = evaluate_gate(make_run({"hybrid": current}), {"hybrid": entry()})
    assert report.status == "fail"
    assert reason_mentioning(report, "hybrid")


def test_configs_without_thresholds_are_reported_but_never_gated() -> None:
    baseline = {"hybrid": entry(), "dense": entry(all_metrics(0.76), gated=())}
    run = make_run({"hybrid": all_metrics(), "dense": all_metrics(0.1), "fts": all_metrics(0.1)})
    report = evaluate_gate(run, baseline)
    # dense collapsed and fts has no baseline row at all, but neither is gated.
    assert report.status == "pass"
    dense = rows_of(report, "dense")["mrr"]
    assert (dense.baseline, dense.passed, dense.threshold) == (pytest.approx(0.76), None, None)
    assert dense.current == pytest.approx(0.1)
    fts = rows_of(report, "fts")["mrr"]
    assert (fts.baseline, fts.passed, fts.threshold) == (None, None, None)


def test_a_baseline_that_gates_nothing_fails() -> None:
    report = evaluate_gate(
        make_run({"hybrid": all_metrics(), "dense": all_metrics()}),
        {"hybrid": entry(gated=()), "dense": entry(gated=())},
    )
    assert report.status == "fail"
    assert report.reasons


MISMATCHES = [
    pytest.param({"golden_set_version": "v2"}, id="golden-set-version"),
    pytest.param({"golden_set_sha256": "b" * 64}, id="golden-set-sha256"),
    pytest.param({"index_config_hash": "e" * 64}, id="index-config-hash"),
]


@pytest.mark.parametrize("differs", MISMATCHES)
def test_a_run_from_another_setup_than_the_baseline_fails_with_a_reason(
    differs: dict[str, str],
) -> None:
    # Numbers scored on another golden set or index aren't comparable, even if they look better.
    run = make_run({"hybrid": all_metrics(0.9)}, overrides=differs)
    report = evaluate_gate(run, {"hybrid": entry(all_metrics(0.7))})
    assert report.status == "fail"
    assert reason_mentioning(report, "hybrid")
    assert not any(row.passed for row in rows_of(report, "hybrid").values())


def test_the_same_inputs_give_the_same_report() -> None:
    run = make_run({"hybrid": all_metrics(0.5)})
    baseline = {"hybrid": entry(all_metrics(0.7))}
    assert evaluate_gate(run, baseline) == evaluate_gate(run, baseline)


# --- Baseline schema (boilerplate, green) --------------------------------------------------------


def test_thresholds_are_optional_and_default_to_none() -> None:
    assert entry(gated=()).thresholds == {}


def test_a_threshold_names_a_metric_the_row_has() -> None:
    with pytest.raises(ValidationError, match="the row doesn't have: p99"):
        entry(gated=("mrr", "p99"))


def test_a_tolerance_cannot_be_negative() -> None:
    with pytest.raises(ValidationError):
        MetricThreshold(tolerance=-0.01)


@pytest.mark.parametrize("field", ["golden_set_sha256", "git_dirty"])
def test_a_baseline_row_needs_the_golden_set_hash_and_the_git_state(field: str) -> None:
    fields = entry().model_dump(mode="json")
    del fields[field]
    with pytest.raises(ValidationError, match=field):
        RetrievalBaselineEntry.model_validate(fields)
    fields[field] = None
    with pytest.raises(ValidationError, match=field):
        RetrievalBaselineEntry.model_validate(fields)


def test_the_committed_baseline_still_loads() -> None:
    rows = read_baseline(RETRIEVAL_BASELINE)
    assert {"dense", "fts", "hybrid"} <= set(rows)


def test_refreshing_a_baseline_row_keeps_its_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    update_baseline(path, make_run({"hybrid": all_metrics(0.5)}))
    rows = json.loads(path.read_bytes())
    rows["hybrid"]["thresholds"] = {"mrr": {"tolerance": 0.04}}
    path.write_text(json.dumps(rows), encoding="utf-8")

    refreshed = update_baseline(path, make_run({"hybrid": all_metrics(0.6)}))
    assert refreshed["hybrid"].thresholds == {"mrr": MetricThreshold(tolerance=0.04)}
    assert refreshed["hybrid"].metrics["mrr"] == 0.6
    assert read_baseline(path)["hybrid"].thresholds == refreshed["hybrid"].thresholds


def test_a_run_without_git_state_cannot_become_a_baseline_row(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    run = make_run({"hybrid": all_metrics()}, overrides={"git_sha": None, "git_dirty": None})
    with pytest.raises(RetrievalEvalError, match="no git state"):
        update_baseline(path, run)
    assert not path.exists()


# --- Report: Markdown and exit code (boilerplate, green) -----------------------------------------


def row(config: str, metric: str, **fields: Any) -> GateRow:
    values: dict[str, Any] = {
        "baseline": 0.74,
        "current": 0.7,
        "delta": -0.04,
        "threshold": 0.7,
        "passed": True,
        "n": 25,
    }
    return GateRow(config=config, metric=metric, **(values | fields))


def test_markdown_shows_every_column_the_reasons_and_n() -> None:
    report = GateReport(
        status="fail",
        rows=[
            row("hybrid", "recall@5"),
            row("hybrid", "mrr", current=0.6, delta=-0.14, passed=False),
            row("dense", "mrr", threshold=None, passed=None),
            row("fts", "mrr", baseline=None, delta=None, threshold=None, passed=None),
        ],
        reasons=["hybrid: missing from the results"],
    )
    assert render_markdown(report) == (
        "### Retrieval gate: ❌ fail\n"
        "\n"
        "| config | metric | baseline | current | Δ | threshold | n | |\n"
        "|---|---|---:|---:|---:|---:|---:|:-:|\n"
        "| hybrid | recall@5 | 0.740 | 0.700 | -0.040 | ≥ 0.700 | 25 | ✅ |\n"
        "| hybrid | mrr | 0.740 | 0.600 | -0.140 | ≥ 0.700 | 25 | ❌ |\n"
        "| dense | mrr | 0.740 | 0.700 | -0.040 | not gated | 25 | · |\n"
        "| fts | mrr | — | 0.700 | — | not gated | 25 | · |\n"
        "\n"
        "· = not gated: reported only.\n"
        "\n"
        "- hybrid: missing from the results\n"
    )


def test_markdown_without_rows_is_the_headline_and_the_reasons() -> None:
    report = GateReport(status="fail", rows=[], reasons=["nothing is gated"])
    assert render_markdown(report) == "### Retrieval gate: ❌ fail\n\n- nothing is gated\n"


def test_markdown_for_missing_current_and_the_other_statuses() -> None:
    missing = row("hybrid", "mrr", current=None, delta=None, n=None, passed=False)
    text = render_markdown(GateReport(status="pass", rows=[missing], reasons=[]))
    assert text.startswith("### Retrieval gate: ✅ pass\n")
    assert "| hybrid | mrr | 0.740 | — | — | ≥ 0.700 | — | ❌ |" in text
    assert "inconclusive" in render_markdown(GateReport(status="inconclusive", rows=[], reasons=[]))


@pytest.mark.parametrize(
    ("status", "code"), [("pass", 0), ("inconclusive", 0), ("fail", 1)], ids=str
)
def test_exit_code_is_zero_unless_the_gate_fails(status: Any, code: int) -> None:
    assert exit_code(GateReport(status=status, rows=[], reasons=[])) == code


# --- CLI (boilerplate, green): evaluate_gate is scripted -----------------------------------------


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    """A results file and a baseline file that both validate."""
    results = tmp_path / "run.json"
    baseline = tmp_path / "retrieval.json"
    run = make_run({"hybrid": all_metrics(0.6)})
    write_run(results, run)
    update_baseline(baseline, run)
    return results, baseline


def script_gate(monkeypatch: pytest.MonkeyPatch, status: Any) -> list[tuple[Any, Any]]:
    calls: list[tuple[Any, Any]] = []

    def fake(run: RetrievalRun, baseline: Any) -> GateReport:
        calls.append((run, baseline))
        return GateReport(status=status, rows=[row("hybrid", "mrr")], reasons=[])

    monkeypatch.setattr(grounded.cli, "evaluate_gate", fake)
    return calls


def gate(results: Path, baseline: Path | None, *extra: str) -> Any:
    args = ["eval", "gate", "--suite", "retrieval", "--results", str(results), *extra]
    if baseline is not None:
        args += ["--baseline", str(baseline)]
    return runner.invoke(app, args)


@pytest.mark.parametrize(("status", "code"), [("pass", 0), ("inconclusive", 0), ("fail", 1)])
def test_gate_prints_the_report_and_exits_with_its_status(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, status: str, code: int
) -> None:
    results, baseline = files
    calls = script_gate(monkeypatch, status)
    result = gate(results, baseline)
    assert result.exit_code == code, result.output
    assert result.output.startswith(f"### Retrieval gate: {HEADLINE[status]} {status}\n")
    assert "| hybrid | mrr |" in result.output
    [(run, rows)] = calls
    assert isinstance(run, RetrievalRun)
    assert set(rows) == {"hybrid"}


def test_gate_defaults_to_the_committed_baseline_file(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    results, baseline = files
    monkeypatch.setattr(grounded.cli, "RETRIEVAL_BASELINE", baseline)
    calls = script_gate(monkeypatch, "pass")
    assert gate(results, None).exit_code == 0
    assert set(calls[0][1]) == {"hybrid"}


def test_gate_exits_2_when_the_results_file_is_missing(
    files: tuple[Path, Path], tmp_path: Path
) -> None:
    result = gate(tmp_path / "nope.json", files[1])
    assert result.exit_code == 2
    assert "Cannot read the results file" in result.output


def test_gate_exits_2_when_the_results_file_is_not_a_run(
    files: tuple[Path, Path], tmp_path: Path
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"suite": "retrieval"}', encoding="utf-8")
    result = gate(bad, files[1])
    assert result.exit_code == 2
    assert "Cannot read the results file" in result.output


def test_gate_rejects_a_baseline_row_without_the_golden_set_hash(
    files: tuple[Path, Path],
) -> None:
    results, baseline = files
    rows = json.loads(baseline.read_bytes())
    del rows["hybrid"]["golden_set_sha256"]
    baseline.write_text(json.dumps(rows), encoding="utf-8")
    result = gate(results, baseline)
    assert result.exit_code == 2
    assert "golden_set_sha256" in result.output
    assert "baseline PR" in result.output


def test_gate_exits_2_when_the_baseline_file_is_missing(
    files: tuple[Path, Path], tmp_path: Path
) -> None:
    result = gate(files[0], tmp_path / "nope.json")
    assert result.exit_code == 2
    assert "Cannot read the baseline" in result.output


def test_the_generation_gate_is_not_there_yet(files: tuple[Path, Path]) -> None:
    results, baseline = files
    args = ["eval", "gate", "--suite", "generation", "--results", str(results)]
    result = runner.invoke(app, [*args, "--baseline", str(baseline)])
    assert result.exit_code == 2
    assert "Phase 4" in result.output


def test_gate_help_names_the_options() -> None:
    result = runner.invoke(app, ["eval", "gate", "--help"])
    assert result.exit_code == 0
    # CI sets FORCE_COLOR, so the help is styled: compare the text without the escape codes.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ("--suite", "--results", "--baseline"):
        assert option in plain
