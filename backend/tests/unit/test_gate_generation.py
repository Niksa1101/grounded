"""The generation gate (Tech.md §15.5, ticket 4.07).

The spec tests of ``evaluate_generation_gate`` (ticket 4.08, ``xfail(strict=True)`` until then)
state *behavior*: hand-built runs and baselines, and the verdicts and numbers worked out on paper in
the comments. They do not re-implement the aggregation. The rest is boilerplate and already green:
the Markdown report, the exit code and the CLI wrapper with the gate function scripted.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.gate import (
    ConfigSummary,
    ErrorCounts,
    GateCannotRunError,
    GateRow,
    GenerationGateReport,
    evaluate_generation_gate,
    exit_code,
    render_generation_markdown,
)
from grounded.evals.generation_results import parse_results
from grounded.schemas.generation_eval import (
    ErrorInfo,
    ErrorStage,
    GenerationBaselineEntry,
    GenerationCase,
    GenerationConfigResult,
    GenerationRun,
    GenerationRunInfo,
    GenerationThreshold,
    MetricResult,
)

spec = pytest.mark.xfail(strict=True, reason="Author implements in 4.08")

NOW = datetime(2026, 10, 8, 12, 0, 0, tzinfo=UTC)
SHA = "8f0848177a5cb5a8ef649ee211ab9afb7f9bc07c73d37b37cbf3ef554e1abcb2"  # the recorded samples'
INDEX = "0.141.1@4949e8a3"
JUDGED = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample_judge.json"
)
PLAIN = JUDGED.with_name("results_sample.json")
HEADLINE = {"pass": "✅", "inconclusive": "⚠️", "fail": "❌"}

# The rules of Tech §15.5, as the baseline file carries them.
GATED = {
    "faithfulness": GenerationThreshold(floor=0.85, tolerance=0.05),
    "correctness": GenerationThreshold(tolerance=0.08),
    "refusal_correctness": GenerationThreshold(tolerance=0.07),
    "schema_first_try": GenerationThreshold(floor=0.95),
}
PERFECT = dict.fromkeys(GATED, 1.0)
GATED_METRICS = set(GATED)

runner = CliRunner()

# --- Builders ------------------------------------------------------------------------------------


def scored(value: float, claims: int | None = None, supported: int | None = None) -> MetricResult:
    return MetricResult(state="scored", value=value, claims=claims, supported=supported)


NA = MetricResult(state="na")


def generator_failed(
    kind: str, *, is_quota: bool = False, bases: tuple[str, ...] = (), skipped: bool = False
) -> ErrorInfo:
    return ErrorInfo(
        stage="generator", kind=kind, bases=bases, is_quota=is_quota, skipped=skipped, detail="x"
    )


def metric_failed(
    kind: str,
    *,
    stage: ErrorStage = "judge",
    provider_side: bool | None = None,
    is_quota: bool = False,
) -> MetricResult:
    error = ErrorInfo(
        stage=stage, kind=kind, provider_side=provider_side, is_quota=is_quota, detail="x"
    )
    return MetricResult(state="errored", error=error)


def judge_down(kind: str = "ProviderTimeout", *, is_quota: bool = False) -> MetricResult:
    """A judge metric that errored for a provider reason."""
    return metric_failed(kind, provider_side=True, is_quota=is_quota)


def build(
    index: int,
    metrics: dict[str, float | MetricResult],
    *,
    error: ErrorInfo | None = None,
    latency_ms: float = 100.0,
    cold_start: bool = False,
    llm_cache_hits: int = 0,
    cost_usd: float = 0.001,
) -> GenerationCase:
    """Question ``q<index>``. The four gated metrics default to a perfect score; a float is a
    scored value, a ``MetricResult`` is taken as it is. With ``error`` the generator failed."""
    fields: dict[str, Any] = {"id": f"q{index:03d}", "type": "factual", "cost_usd": cost_usd}
    if error is not None:
        return GenerationCase(**fields, error=error)
    values = dict.fromkeys(GATED, 1.0) | metrics
    return GenerationCase(
        **fields,
        metrics={k: v if isinstance(v, MetricResult) else scored(v) for k, v in values.items()},
        latency_ms=latency_ms,
        cold_start=cold_start,
        llm_cache_hits=llm_cache_hits,
    )


def case(
    index: int,
    *,
    error: ErrorInfo | None = None,
    latency_ms: float = 100.0,
    cold_start: bool = False,
    llm_cache_hits: int = 0,
    cost_usd: float = 0.001,
    **metrics: float | MetricResult,
) -> GenerationCase:
    """``build`` with the metrics as keyword arguments: ``case(3, correctness=0.5)``."""
    return build(
        index,
        metrics,
        error=error,
        latency_ms=latency_ms,
        cold_start=cold_start,
        llm_cache_hits=llm_cache_hits,
        cost_usd=cost_usd,
    )


def both_judge_metrics_down(index: int) -> GenerationCase:
    return case(index, faithfulness=judge_down(), correctness=judge_down())


def faithfulness_bad_output(index: int) -> GenerationCase:
    return case(index, faithfulness=metric_failed("ProviderBadOutput", provider_side=False))


def precision_malformed(index: int) -> GenerationCase:
    return case(index, citation_precision=metric_failed("MalformedInput", stage="assertion"))


def ones(k: int, n: int) -> list[float]:
    """``n`` per-question values: ``k`` perfect ones, the rest zero."""
    return [1.0] * k + [0.0] * (n - k)


def series(metric: str, values: list[float]) -> list[GenerationCase]:
    """One case per value for ``metric``; the other metrics are perfect."""
    return [build(i + 1, {metric: value}) for i, value in enumerate(values)]


def some_failing(
    total: int, failing: int, make: Callable[[int], GenerationCase]
) -> list[GenerationCase]:
    """``total`` cases, the first ``failing`` of them built by ``make(index)``."""
    return [make(i) if i <= failing else case(i) for i in range(1, total + 1)]


def make_run(
    configs: dict[str, list[GenerationCase]], *, info: dict[str, Any] | None = None, **identity: Any
) -> GenerationRun:
    """A run. ``info`` overrides the run's fields, ``identity`` every config's answer identity."""
    run_info = GenerationRunInfo.model_validate(
        {
            "date": NOW,
            "promptfoo_version": "0.123.1",
            "golden_set_version": "v1",
            "golden_set_sha256": SHA,
            "judge_provider": "groq",
            "judge_model": "openai/gpt-oss-120b",
        }
        | (info or {})
    )
    who: dict[str, Any] = {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "answer_v1@08cc49e5",
        "index_version": INDEX,
        "retrieval_config_hash": None,
    } | identity
    results = {
        name: GenerationConfigResult(config=name, cases=cases, **who)
        for name, cases in configs.items()
    }
    return GenerationRun(info=run_info, configs=results)


def entry(
    metrics: dict[str, float] | None = None,
    *,
    thresholds: dict[str, GenerationThreshold] | None = None,
    **overrides: Any,
) -> GenerationBaselineEntry:
    """A baseline row. Default: all four gated metrics at 1.0 with the §15.5 rules. Pass
    ``thresholds={}`` for a reported-only row."""
    numbers = PERFECT if metrics is None else metrics
    fields: dict[str, Any] = {
        "metrics": numbers,
        "n": dict.fromkeys(numbers, 30),
        "thresholds": GATED if thresholds is None else thresholds,
        "cases": 30,
        "golden_set_version": "v1",
        "golden_set_sha256": SHA,
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "answer_v1@08cc49e5",
        "index_version": INDEX,
        "judge_provider": "groq",
        "judge_model": "openai/gpt-oss-120b",
        "judge_prompt_versions": {"faithfulness": "judge_faithfulness_v1@84103412"},
        "promptfoo_version": "0.123.1",
        "git_sha": "c" * 40,
        "git_dirty": False,
        "date": NOW,
    }
    return GenerationBaselineEntry.model_validate(fields | overrides)


def gate_hybrid(
    cases: list[GenerationCase], baseline: dict[str, float] | None = None
) -> GenerationGateReport:
    """One gated config, ``hybrid``, against a baseline row of ``baseline`` (default: all 1.0)."""
    return evaluate_generation_gate(
        make_run({"hybrid": cases}), {"hybrid": entry(None if baseline is None else baseline)}
    )


def rows_of(report: GenerationGateReport, config: str) -> dict[str, GateRow]:
    return {row.metric: row for row in report.rows if row.config == config}


def summary_of(report: GenerationGateReport, config: str) -> ConfigSummary:
    [found] = [s for s in report.configs if s.config == config]
    return found


def reason_mentioning(report: GenerationGateReport, *words: str) -> bool:
    return any(all(word in reason for word in words) for reason in report.reasons)


# --- Spec: thresholds (Tech §15.5) ---------------------------------------------------------------


@spec
def test_a_run_equal_to_the_baseline_passes() -> None:
    report = gate_hybrid([case(i) for i in range(1, 11)])
    assert report.status == "pass"
    assert report.reasons == []
    assert {name for name, row in rows_of(report, "hybrid").items() if row.passed} == GATED_METRICS


@spec
def test_a_run_better_than_the_baseline_passes() -> None:
    # correctness: baseline 0.7, the run scores 1.0 on all ten questions.
    report = gate_hybrid([case(i) for i in range(1, 11)], {**PERFECT, "correctness": 0.7})
    assert report.status == "pass"
    assert rows_of(report, "hybrid")["correctness"].delta == pytest.approx(0.3)


# (metric, baseline, per-question values, current, threshold, passes), worked out by hand. The
# decimal ones are float-hostile: 0.8 - 0.08 is 0.7200000000000001 and 0.9 - 0.07 is
# 0.8300000000000001, yet a run that scored exactly 0.72 (36/50) or 0.83 (83/100) is at the
# threshold.
BOUNDARIES = [
    # faithfulness: the floor 0.85 against baseline - 0.05, whichever is higher.
    ("faithfulness", 0.88, ones(17, 20), 0.85, 0.85, True),  # 0.88 - 0.05 = 0.83: the floor rules
    ("faithfulness", 0.88, ones(18, 20), 0.90, 0.85, True),
    ("faithfulness", 0.88, [1.0] * 16 + [0.99] + [0.0] * 3, 0.8495, 0.85, False),  # 16.99 / 20
    (
        "faithfulness",
        0.93,
        ones(88, 100),
        0.88,
        0.88,
        True,
    ),  # 0.93 - 0.05 = 0.88: the baseline rules
    ("faithfulness", 0.93, ones(89, 100), 0.89, 0.88, True),
    ("faithfulness", 0.93, ones(87, 100), 0.87, 0.88, False),  # above the floor, below the rule
    # correctness: baseline - 0.08.
    ("correctness", 0.80, ones(36, 50), 0.72, 0.72, True),
    ("correctness", 0.80, ones(37, 50), 0.74, 0.72, True),
    ("correctness", 0.80, [1.0] * 35 + [0.5] + [0.0] * 14, 0.71, 0.72, False),  # 35.5 / 50
    ("correctness", 0.80, [1.0] * 30 + [0.5] * 12 + [0.0] * 8, 0.72, 0.72, True),  # (30 + 6) / 50
    ("correctness", 0.78, ones(70, 100), 0.70, 0.70, True),
    ("correctness", 0.78, ones(69, 100), 0.69, 0.70, False),
    # refusal accuracy (the refusal_correctness assertion): baseline - 0.07.
    ("refusal_correctness", 0.90, ones(83, 100), 0.83, 0.83, True),
    ("refusal_correctness", 0.90, ones(84, 100), 0.84, 0.83, True),
    ("refusal_correctness", 0.90, ones(82, 100), 0.82, 0.83, False),
    # schema first-try validity: 0.95 absolute, whatever the baseline.
    ("schema_first_try", 1.00, ones(95, 100), 0.95, 0.95, True),
    ("schema_first_try", 1.00, ones(100, 100), 1.00, 0.95, True),
    ("schema_first_try", 1.00, ones(94, 100), 0.94, 0.95, False),
    ("schema_first_try", 0.97, ones(19, 20), 0.95, 0.95, True),
    ("schema_first_try", 0.80, ones(90, 100), 0.90, 0.95, False),  # above its baseline, below 0.95
]


@spec
@pytest.mark.parametrize(
    ("metric", "baseline", "values", "current", "threshold", "passes"), BOUNDARIES
)
def test_each_threshold_at_just_inside_and_just_outside_the_boundary(
    metric: str,
    baseline: float,
    values: list[float],
    current: float,
    threshold: float,
    passes: bool,
) -> None:
    report = gate_hybrid(series(metric, values), {**PERFECT, metric: baseline})

    row = rows_of(report, "hybrid")[metric]
    assert (row.current, row.threshold) == (pytest.approx(current), pytest.approx(threshold))
    assert (row.n, row.passed) == (len(values), passes)
    assert report.status == ("pass" if passes else "fail")
    assert [name for name, r in rows_of(report, "hybrid").items() if r.passed is False] == (
        [] if passes else [metric]
    )


@spec
@pytest.mark.parametrize("metric", sorted(GATED_METRICS))
def test_one_failing_metric_fails_the_suite(metric: str) -> None:
    report = gate_hybrid(series(metric, [0.0] * 10))
    assert report.status == "fail"
    assert [n for n, row in rows_of(report, "hybrid").items() if row.passed is False] == [metric]


@spec
def test_rows_carry_baseline_current_delta_threshold_and_the_runs_n() -> None:
    # correctness: 15 of 20 questions fully right = 0.75, baseline 0.80, so >= 0.72 and -0.05.
    report = gate_hybrid(series("correctness", ones(15, 20)), {**PERFECT, "correctness": 0.80})

    row = rows_of(report, "hybrid")["correctness"]
    assert row.baseline == pytest.approx(0.80)
    assert row.current == pytest.approx(0.75)
    assert row.delta == pytest.approx(-0.05)
    assert row.threshold == pytest.approx(0.72)
    assert (row.passed, row.n) == (True, 20)  # the run's n, not the baseline's 30


@spec
def test_only_metrics_with_a_threshold_are_gated() -> None:
    baseline = entry(thresholds={"faithfulness": GATED["faithfulness"]})
    report = evaluate_generation_gate(
        make_run({"hybrid": series("correctness", [0.0] * 10)}), {"hybrid": baseline}
    )
    assert report.status == "pass"  # correctness collapsed, but only faithfulness is gated
    unrated = rows_of(report, "hybrid")["correctness"]
    assert (unrated.current, unrated.threshold, unrated.passed) == (0.0, None, None)


# --- Spec: the inconclusive rule -----------------------------------------------------------------


@spec
@pytest.mark.parametrize(
    ("total", "errored", "inconclusive"),
    [
        pytest.param(25, 5, False, id="5-of-25-is-exactly-20%"),
        pytest.param(25, 6, True, id="6-of-25-is-24%"),
        pytest.param(30, 6, False, id="6-of-30-is-exactly-20%"),
        pytest.param(30, 7, True, id="7-of-30-is-23%"),
        pytest.param(35, 7, False, id="7-of-35-is-exactly-20%"),
        pytest.param(35, 8, True, id="8-of-35-is-23%"),
        pytest.param(10, 2, False, id="2-of-10-is-exactly-20%"),
        pytest.param(10, 3, True, id="3-of-10-is-30%"),
        pytest.param(5, 1, False, id="1-of-5-is-exactly-20%"),
        pytest.param(5, 2, True, id="2-of-5-is-40%"),
    ],
)
def test_exactly_20_percent_errored_is_not_inconclusive_just_above_is(
    total: int, errored: int, inconclusive: bool
) -> None:
    cases = some_failing(
        total, errored, lambda i: case(i, error=generator_failed("ProviderTimeout"))
    )
    report = gate_hybrid(cases)

    summary = summary_of(report, "hybrid")
    assert (summary.cases, summary.errors.provider, summary.n) == (total, errored, total - errored)
    assert summary.inconclusive is inconclusive
    assert report.status == ("inconclusive" if inconclusive else "pass")


PROVIDER_REASONS: dict[str, Callable[[int], GenerationCase]] = {
    "generator-per-minute-limit": lambda i: case(i, error=generator_failed("ProviderRateLimited")),
    "generator-daily-quota": lambda i: case(
        i, error=generator_failed("ProviderRateLimited", is_quota=True)
    ),
    "generator-skipped-after-a-quota": lambda i: case(
        i, error=generator_failed("ProviderRateLimited", is_quota=True, skipped=True)
    ),
    "generator-backoff-exhausted": lambda i: case(
        i,
        error=generator_failed(
            "BackoffExhaustedError", bases=("ProviderRateLimited", "ProviderError")
        ),
    ),
    "generator-5xx": lambda i: case(i, error=generator_failed("ProviderUnavailable")),
    "generator-timeout": lambda i: case(i, error=generator_failed("ProviderTimeout")),
    "generator-embedder-down": lambda i: case(  # proposed: the embedding API failed
        i, error=generator_failed("EmbedderUnavailableError")
    ),
    "judge-daily-quota": lambda i: case(
        i, correctness=judge_down("ProviderRateLimited", is_quota=True)
    ),
    "judge-backoff-exhausted": lambda i: case(i, correctness=judge_down("BackoffExhaustedError")),
    "judge-5xx": lambda i: case(i, faithfulness=judge_down("ProviderUnavailable")),
    "judge-timeout": lambda i: case(i, correctness=judge_down("ProviderTimeout")),
}


@spec
@pytest.mark.parametrize("make", PROVIDER_REASONS.values(), ids=PROVIDER_REASONS.keys())
def test_a_quota_a_5xx_or_a_timeout_of_the_generator_or_the_judge_counts(
    make: Callable[[int], GenerationCase],
) -> None:
    report = gate_hybrid(some_failing(10, 3, make))  # 30%
    assert summary_of(report, "hybrid").errors.provider == 3
    assert summary_of(report, "hybrid").inconclusive is True
    assert report.status == "inconclusive"


QUALITY_MISSES: dict[str, tuple[Callable[[int], GenerationCase], str]] = {
    "generator-bad-output": (
        lambda i: case(i, error=generator_failed("ProviderBadOutput")),
        "generator_bad_output",
    ),
    "judge-bad-output": (
        lambda i: case(i, faithfulness=metric_failed("ProviderBadOutput", provider_side=False)),
        "judge_bad_output",
    ),
    "malformed-input": (
        lambda i: case(i, citation_precision=metric_failed("MalformedInput", stage="assertion")),
        "malformed_input",
    ),
}


@spec
@pytest.mark.parametrize(("make", "category"), QUALITY_MISSES.values(), ids=QUALITY_MISSES.keys())
def test_a_bad_output_or_a_harness_bug_is_not_a_provider_reason(
    make: Callable[[int], GenerationCase], category: str
) -> None:
    report = gate_hybrid(some_failing(10, 3, make))  # 30% would be inconclusive if it counted

    summary = summary_of(report, "hybrid")
    assert summary.inconclusive is False
    assert summary.errors.provider == 0
    assert getattr(summary.errors, category) == 3
    assert report.status != "inconclusive"


@spec
def test_a_case_with_several_provider_errors_counts_once() -> None:
    report = gate_hybrid(some_failing(10, 2, both_judge_metrics_down))  # 2 of 10, not 4 of 10
    assert summary_of(report, "hybrid").errors.provider == 2
    assert summary_of(report, "hybrid").inconclusive is False


@spec
def test_inconclusive_shows_the_metrics_with_n_but_gates_nothing() -> None:
    # Ten cases, three lost to 5xx (30%); the seven that ran have collapsed correctness.
    cases = [case(i, error=generator_failed("ProviderUnavailable")) for i in (1, 2, 3)]
    cases += [case(i, correctness=0.0) for i in range(4, 11)]
    report = gate_hybrid(cases)

    assert report.status == "inconclusive"
    assert exit_code(report) == 0
    assert report.reasons == []
    rows = rows_of(report, "hybrid")
    assert {row.passed for row in rows.values()} == {None}  # nothing is gated
    assert (rows["correctness"].current, rows["correctness"].n) == (0.0, 7)
    assert (rows["schema_first_try"].current, rows["schema_first_try"].n) == (1.0, 7)
    assert summary_of(report, "hybrid").inconclusive is True


@spec
def test_a_run_where_every_case_errored_is_inconclusive_and_does_not_divide_by_zero() -> None:
    report = gate_hybrid(
        [case(i, error=generator_failed("ProviderRateLimited")) for i in range(1, 11)]
    )

    assert report.status == "inconclusive"
    for metric in GATED_METRICS:  # listed from the baseline row, as not available
        row = rows_of(report, "hybrid")[metric]
        assert (row.current, row.delta, row.n, row.passed) == (None, None, 0, None)
    summary = summary_of(report, "hybrid")
    assert (summary.cases, summary.n, summary.n_faithfulness) == (10, 0, 0)
    assert (summary.latency_p50_ms, summary.latency_p95_ms, summary.cost_per_1k_usd) == (
        None,
        None,
        None,
    )


@spec
def test_a_reported_only_config_never_changes_the_status_even_when_it_is_inconclusive() -> None:
    run = make_run(
        {
            "hybrid": [case(i) for i in range(1, 11)],
            "no_rag": some_failing(
                10, 5, lambda i: case(i, error=generator_failed("ProviderTimeout"))
            ),
        }
    )
    report = evaluate_generation_gate(
        run, {"hybrid": entry(), "no_rag": entry({"correctness": 0.6}, thresholds={})}
    )
    assert report.status == "pass"
    assert summary_of(report, "no_rag").inconclusive is True
    assert summary_of(report, "no_rag").gated is False
    assert summary_of(report, "hybrid").gated is True
    assert {row.passed for row in rows_of(report, "no_rag").values()} == {None}


@spec
def test_a_failing_gated_config_beats_an_inconclusive_one() -> None:
    run = make_run(
        {
            "hybrid": some_failing(
                10, 5, lambda i: case(i, error=generator_failed("ProviderTimeout"))
            ),
            "other": series("correctness", [0.0] * 10),
        }
    )
    report = evaluate_generation_gate(run, {"hybrid": entry(), "other": entry()})
    assert report.status == "fail"
    assert summary_of(report, "hybrid").inconclusive is True


# --- Spec: n, N/A, macro, errors -----------------------------------------------------------------


@spec
def test_n_excludes_the_cases_that_errored_for_that_metric() -> None:
    # Ten cases. q001: the generator hit a 5xx (no metric at all). q002: the judge timed out on
    # correctness only. 2 of 10 is exactly 20%: not inconclusive. Correctness is scored on
    # q003..q010: four 1.0, two 0.5, two 0.0 = 5.0 / 8 = 0.625 (the two lost cases as 0 would
    # give 0.5).
    cases = [
        case(1, error=generator_failed("ProviderUnavailable")),
        case(2, correctness=judge_down()),
        *[
            case(i, correctness=value)
            for i, value in zip(range(3, 11), [1, 1, 1, 1, 0.5, 0.5, 0, 0], strict=True)
        ],
    ]
    report = gate_hybrid(cases, {**PERFECT, "correctness": 0.6})

    rows = rows_of(report, "hybrid")
    assert (rows["correctness"].current, rows["correctness"].n) == (pytest.approx(0.625), 8)
    assert rows["schema_first_try"].n == rows["faithfulness"].n == 9  # q002's other metrics count
    summary = summary_of(report, "hybrid")
    assert (summary.cases, summary.n, summary.errors.provider) == (10, 8, 2)
    assert summary.inconclusive is False


@spec
def test_n_faithfulness_is_its_own_count_next_to_n_because_not_applicable_is_left_out() -> None:
    # Ten answers. q008..q010 are refusals: faithfulness N/A. The seven others: 1, 1, 1, 1, 0.5,
    # 0.5, 0 = 5.0 / 7 = 0.714. As a 1 the three would give 8 / 10 = 0.8, as a 0, 5 / 10 = 0.5.
    scores = [1, 1, 1, 1, 0.5, 0.5, 0]
    cases = [case(i + 1, faithfulness=value) for i, value in enumerate(scores)]
    cases += [case(i, faithfulness=NA) for i in (8, 9, 10)]
    report = gate_hybrid(cases)

    row = rows_of(report, "hybrid")["faithfulness"]
    assert (row.current, row.n) == (pytest.approx(5 / 7), 7)
    summary = summary_of(report, "hybrid")
    assert (summary.n_faithfulness, summary.n) == (7, 10)


@spec
def test_a_question_weighs_the_same_whatever_its_claim_count_so_faithfulness_is_macro() -> None:
    # Per question: 1 of 1 claim, 1 of 4, 3 of 3, 0 of 2. Macro: (1 + 0.25 + 1 + 0) / 4 = 0.5625.
    # Micro (claims pooled) would be 5 / 10 = 0.5.
    per_question = [(1, 1), (1, 4), (3, 3), (0, 2)]
    cases = [
        case(i + 1, faithfulness=scored(supported / claims, claims=claims, supported=supported))
        for i, (supported, claims) in enumerate(per_question)
    ]
    report = gate_hybrid(cases, {**PERFECT, "faithfulness": 0.5})

    assert rows_of(report, "hybrid")["faithfulness"].current == pytest.approx(0.5625)


@spec
def test_not_applicable_is_neither_zero_nor_one_for_any_metric() -> None:
    # citation_precision: 1.0, 0.0 and two N/A = 0.5 over n = 2 (0.75 if N/A were 1, 0.25 if 0).
    cases = [
        case(1, citation_precision=1.0),
        case(2, citation_precision=0.0),
        case(3, citation_precision=NA),
        case(4, citation_precision=NA),
    ]
    row = rows_of(gate_hybrid(cases), "hybrid")["citation_precision"]
    assert (row.current, row.n, row.passed) == (pytest.approx(0.5), 2, None)  # reported only


@spec
def test_a_metric_that_is_not_applicable_to_every_case_has_no_value_and_n_zero() -> None:
    # no_rag: faithfulness is N/A everywhere. Reported, never 0 and never 1.
    run = make_run(
        {
            "hybrid": [case(i) for i in range(1, 6)],
            "no_rag": [case(i, faithfulness=NA, correctness=0.5) for i in range(1, 6)],
        }
    )
    report = evaluate_generation_gate(
        run, {"hybrid": entry(), "no_rag": entry({"correctness": 0.5}, thresholds={})}
    )
    rows = rows_of(report, "no_rag")
    assert (rows["faithfulness"].current, rows["faithfulness"].n) == (None, 0)
    assert (rows["correctness"].current, rows["correctness"].n) == (0.5, 5)
    assert summary_of(report, "no_rag").n_faithfulness == 0


@spec
def test_a_generator_bad_output_stays_in_n_and_counts_against_schema_validity() -> None:
    # Ten questions, three of them ProviderBadOutput: schema_first_try is 7 / 10 = 0.7 over n = 10
    # (< 0.95: fail); the other metrics are scored on the seven that produced an answer.
    cases = some_failing(10, 3, lambda i: case(i, error=generator_failed("ProviderBadOutput")))
    report = gate_hybrid(cases)

    rows = rows_of(report, "hybrid")
    assert (rows["schema_first_try"].current, rows["schema_first_try"].n) == (0.7, 10)
    assert rows["schema_first_try"].passed is False
    assert (rows["correctness"].current, rows["correctness"].n) == (1.0, 7)
    assert (rows["refusal_correctness"].n, rows["faithfulness"].n) == (7, 7)
    assert report.status == "fail"
    summary = summary_of(report, "hybrid")
    assert (summary.n, summary.errors.generator_bad_output, summary.errors.provider) == (10, 3, 0)


@spec
@pytest.mark.parametrize(("bad", "passes"), [(1, True), (2, False)], ids=["1-of-20", "2-of-20"])
def test_one_generator_bad_output_in_twenty_is_exactly_at_the_schema_floor(
    bad: int, passes: bool
) -> None:
    # 19 / 20 = 0.95 passes; 18 / 20 = 0.90 does not.
    cases = some_failing(20, bad, lambda i: case(i, error=generator_failed("ProviderBadOutput")))
    row = rows_of(gate_hybrid(cases), "hybrid")["schema_first_try"]
    assert (row.n, row.passed) == (20, passes)


@spec
def test_a_judge_bad_output_leaves_that_metric_unscored_and_is_not_counted() -> None:
    # Four of ten answers could not be judged for faithfulness (bad output after the retry): 40%
    # would be inconclusive if it counted. Faithfulness is scored on the other six.
    report = gate_hybrid(some_failing(10, 4, faithfulness_bad_output))

    row = rows_of(report, "hybrid")["faithfulness"]
    assert (row.current, row.n) == (1.0, 6)
    assert rows_of(report, "hybrid")["correctness"].n == 10
    summary = summary_of(report, "hybrid")
    assert (summary.inconclusive, summary.errors.provider, summary.errors.judge_bad_output) == (
        False,
        0,
        4,
    )
    assert report.status == "pass"


@spec
def test_a_harness_bug_is_unscored_and_reported() -> None:
    cases = [
        precision_malformed(i) if i <= 2 else case(i, citation_precision=1.0) for i in range(1, 11)
    ]
    report = gate_hybrid(cases)

    assert rows_of(report, "hybrid")["citation_precision"].n == 8
    assert summary_of(report, "hybrid").errors.malformed_input == 2
    assert summary_of(report, "hybrid").inconclusive is False


# --- Spec: reported-only configs and the other fail-closed cases ---------------------------------


@spec
def test_a_config_without_thresholds_is_reported_but_never_gated() -> None:
    run = make_run(
        {
            "hybrid": [case(i) for i in range(1, 11)],
            "no_rag": [case(i, correctness=0.0, refusal_correctness=0.0) for i in range(1, 11)],
            "extra": [case(i, correctness=0.0) for i in range(1, 11)],
        }
    )
    baseline = {"hybrid": entry(), "no_rag": entry({"correctness": 0.6}, thresholds={})}
    report = evaluate_generation_gate(run, baseline)

    # no_rag collapsed and "extra" has no baseline row at all, but neither is gated.
    assert report.status == "pass"
    no_rag = rows_of(report, "no_rag")["correctness"]
    assert (no_rag.baseline, no_rag.current, no_rag.threshold, no_rag.passed) == (
        0.6,
        0.0,
        None,
        None,
    )
    extra = rows_of(report, "extra")["correctness"]
    assert (extra.baseline, extra.delta, extra.threshold, extra.passed) == (None, None, None, None)


@spec
def test_a_gated_config_missing_from_the_results_fails_with_its_name() -> None:
    report = evaluate_generation_gate(
        make_run({"no_rag": [case(1)]}),
        {"hybrid": entry(), "no_rag": entry(thresholds={})},
    )
    assert report.status == "fail"
    assert reason_mentioning(report, "hybrid")


@spec
def test_a_gated_metric_the_run_never_scored_fails_with_the_config_and_metric_name() -> None:
    # A run without judge assertions: faithfulness is gated by the baseline but nobody scored it.
    cases = [
        GenerationCase(
            id=f"q{i:03d}",
            type="factual",
            metrics={m: scored(1.0) for m in GATED_METRICS - {"faithfulness"}},
        )
        for i in range(1, 11)
    ]
    report = gate_hybrid(cases)
    assert report.status == "fail"
    assert reason_mentioning(report, "hybrid", "faithfulness")
    row = rows_of(report, "hybrid")["faithfulness"]
    assert (row.current, row.n, row.passed) == (None, 0, False)
    assert rows_of(report, "hybrid")["correctness"].passed is True  # the others are still checked


@spec
def test_a_gated_metric_that_is_not_applicable_everywhere_fails_instead_of_passing() -> None:
    report = gate_hybrid([case(i, faithfulness=NA) for i in range(1, 11)])
    assert report.status == "fail"
    assert reason_mentioning(report, "hybrid", "faithfulness")


@spec
def test_a_baseline_that_gates_nothing_fails() -> None:
    report = evaluate_generation_gate(
        make_run({"hybrid": [case(1)], "no_rag": [case(1)]}),
        {"hybrid": entry(thresholds={}), "no_rag": entry(thresholds={})},
    )
    assert report.status == "fail"
    assert report.reasons


MISMATCHES = [
    pytest.param({"info": {"golden_set_version": "v2"}}, id="golden-set-version"),
    pytest.param({"info": {"golden_set_sha256": "b" * 64}}, id="golden-set-sha256"),
    pytest.param({"index_version": "0.141.1@deadbeef"}, id="index-version"),
]


@spec
@pytest.mark.parametrize("differs", MISMATCHES)
def test_a_run_from_another_golden_set_or_index_fails_with_a_reason(
    differs: dict[str, Any],
) -> None:
    # Numbers scored on other data aren't comparable, even if they look better.
    run = make_run({"hybrid": [case(i) for i in range(1, 11)]}, **differs)
    report = evaluate_generation_gate(run, {"hybrid": entry({**PERFECT, "correctness": 0.5})})
    assert report.status == "fail"
    assert reason_mentioning(report, "hybrid")
    assert rows_of(report, "hybrid") == {}


@spec
def test_a_changed_prompt_model_or_retrieval_config_is_compared_not_refused() -> None:
    # Catching what a prompt or model change did is the gate's job.
    run = make_run(
        {"hybrid": series("correctness", [0.0] * 10)},
        prompt_version="answer_v2@00000000",
        model="another-model",
        retrieval_config_hash="f" * 64,
    )
    report = evaluate_generation_gate(run, {"hybrid": entry()})
    assert report.status == "fail"
    assert report.reasons == []
    assert rows_of(report, "hybrid")["correctness"].passed is False


@spec
@pytest.mark.parametrize(
    ("make", "kind"),
    [
        pytest.param(
            lambda i: case(i, error=generator_failed("ProviderRequestRejected")),
            "ProviderRequestRejected",
            id="generator-key-refused",
        ),
        pytest.param(
            lambda i: case(
                i, faithfulness=metric_failed("ProviderRequestRejected", provider_side=False)
            ),
            "ProviderRequestRejected",
            id="judge-key-refused",
        ),
        pytest.param(
            lambda i: case(
                i, correctness=metric_failed("ProviderConfigError", provider_side=False)
            ),
            "ProviderConfigError",
            id="judge-not-configured",
        ),
        pytest.param(  # proposed: a kind the contract does not name is the setup's fault
            lambda i: case(i, error=generator_failed("NoActiveIndexError")),
            "NoActiveIndexError",
            id="no-active-index",
        ),
    ],
)
def test_when_the_results_show_the_gate_could_not_run_it_raises_with_the_reason(
    make: Callable[[int], GenerationCase], kind: str
) -> None:
    with pytest.raises(GateCannotRunError) as info:
        gate_hybrid(some_failing(10, 1, make))
    assert any(kind in reason for reason in info.value.reasons)


# --- Spec: latency and cost are reported, never gated --------------------------------------------


def latency_cases() -> list[GenerationCase]:
    clean = [case(i, latency_ms=i * 100.0) for i in range(1, 25)]  # 100 .. 2400 ms
    cached = [case(25 + j, latency_ms=float(j + 1), llm_cache_hits=1) for j in range(3)]  # 1, 2, 3
    cold = [case(28, latency_ms=99999.0, cold_start=True)]
    lost = [case(29, error=generator_failed("ProviderUnavailable"), cost_usd=0.1)]
    return [*clean, *cached, *cold, *lost]


@spec
def test_latency_percentiles_leave_out_cold_starts_cache_hits_and_failures() -> None:
    # 24 warm latencies 100..2400, nearest rank: p50 = rank ceil(0.5 * 24) = 12 -> 1200 and
    # p95 = rank ceil(0.95 * 24) = 23 -> 2300. Pooling the 3 cache hits and the cold start (28
    # values) would give 1100 and 2400.
    summary = summary_of(gate_hybrid(latency_cases()), "hybrid")
    assert (summary.n_latency, summary.latency_p50_ms, summary.latency_p95_ms) == (
        24,
        1200.0,
        2300.0,
    )
    assert (summary.cases, summary.n, summary.errors.provider) == (29, 28, 1)  # one 5xx


@spec
def test_cost_per_1k_is_the_mean_over_answered_cases_cache_hits_included() -> None:
    # 28 answered cases at $0.001 each (the cache hits and the cold start stay in) = $1.00 per 1k.
    # The lost case spent $0.1 before it failed: not part of it (it would give about $4.41).
    summary = summary_of(gate_hybrid(latency_cases()), "hybrid")
    assert summary.cost_per_1k_usd == pytest.approx(1.0)


@spec
def test_a_run_with_no_warm_case_has_no_latency_but_still_a_cost() -> None:
    cases = [case(i, cold_start=True) for i in range(1, 4)] + [
        case(i, llm_cache_hits=2) for i in range(4, 7)
    ]
    summary = summary_of(gate_hybrid(cases), "hybrid")
    assert (summary.n_latency, summary.latency_p50_ms, summary.latency_p95_ms) == (0, None, None)
    assert summary.cost_per_1k_usd == pytest.approx(1.0)


@spec
def test_latency_and_cost_are_never_gated() -> None:
    slow = [case(i, latency_ms=60000.0, cost_usd=5.0) for i in range(1, 11)]
    report = gate_hybrid(slow)
    assert report.status == "pass"
    assert not any("latency" in row.metric or "cost" in row.metric for row in report.rows)
    assert summary_of(report, "hybrid").cost_per_1k_usd == pytest.approx(5000.0)


# --- Spec: determinism and the recorded runs -----------------------------------------------------


@spec
def test_the_same_inputs_give_the_same_report_with_rows_sorted_by_config_then_metric() -> None:
    run = make_run({"no_rag": [case(1)], "hybrid": [case(i) for i in range(1, 11)]})
    baseline = {"hybrid": entry(), "no_rag": entry(thresholds={})}
    report = evaluate_generation_gate(run, baseline)

    assert report == evaluate_generation_gate(run, baseline)
    keys = [(row.config, row.metric) for row in report.rows]
    assert keys == sorted(keys)
    assert [s.config for s in report.configs] == ["hybrid", "no_rag"]


def recorded_baseline(run: GenerationRun) -> dict[str, GenerationBaselineEntry]:
    index = run.configs["hybrid"].index_version
    return {
        "hybrid": entry(golden_set_sha256=run.info.golden_set_sha256, index_version=index),
        "no_rag": entry({"correctness": 0.5}, thresholds={}, index_version="none"),
    }


@spec
def test_the_recorded_run_with_the_judge_is_inconclusive_for_hybrid() -> None:
    # hybrid, 7 questions: provider errors on q015 (judge timeout), q047 (quota, then skipped) and
    # q049 (skipped) = 3 of 7 > 20%. q008's faithfulness was a judge bad output (not counted).
    run = parse_results(JUDGED.read_bytes())
    report = evaluate_generation_gate(run, recorded_baseline(run))

    assert report.status == "inconclusive"
    summary = summary_of(report, "hybrid")
    assert (summary.cases, summary.n, summary.n_faithfulness) == (7, 4, 3)
    assert (summary.errors.provider, summary.errors.judge_bad_output) == (3, 1)
    assert (summary.latency_p50_ms, summary.latency_p95_ms, summary.n_latency) == (14.0, 18.0, 6)
    assert summary.cost_per_1k_usd == pytest.approx(1.16858, rel=1e-4)
    rows = rows_of(report, "hybrid")
    assert (rows["faithfulness"].current, rows["faithfulness"].n) == (pytest.approx(2 / 3), 3)
    assert (rows["correctness"].current, rows["correctness"].n) == (pytest.approx(0.875), 4)
    assert (rows["refusal_correctness"].current, rows["refusal_correctness"].n) == (
        pytest.approx(5 / 7),
        7,
    )
    assert {row.passed for row in rows.values()} == {None}
    # no_rag: q015 and q049 errored for the judge's provider = 2 of 7; correctness 0.5 / 5.
    no_rag = rows_of(report, "no_rag")
    assert (no_rag["faithfulness"].current, no_rag["faithfulness"].n) == (None, 0)
    assert (no_rag["correctness"].current, no_rag["correctness"].n) == (pytest.approx(0.1), 5)


@spec
def test_the_recorded_run_without_a_judge_keeps_the_generator_bad_output_in_n() -> None:
    # hybrid: q008 is a generator bad output, q015 a backoff, q047 a quota, q049 skipped: 3 of 7
    # provider errors (inconclusive). schema_first_try: q003, q007, q045 = 1.0 and q008 = 0 -> 0.75.
    run = parse_results(PLAIN.read_bytes())
    report = evaluate_generation_gate(run, recorded_baseline(run))

    assert report.status == "inconclusive"
    assert (
        report.reasons == []
    )  # faithfulness has no case at all, but an inconclusive run gates nothing
    summary = summary_of(report, "hybrid")
    assert (summary.cases, summary.n, summary.errors.generator_bad_output) == (7, 4, 1)
    rows = rows_of(report, "hybrid")
    assert (rows["schema_first_try"].current, rows["schema_first_try"].n) == (0.75, 4)
    assert (rows["refusal_correctness"].current, rows["refusal_correctness"].n) == (1.0, 3)


@spec
def test_the_cli_gates_a_recorded_run_end_to_end(tmp_path: Path) -> None:
    run = parse_results(JUDGED.read_bytes())
    baseline = tmp_path / "generation.json"
    rows = {name: row.model_dump(mode="json") for name, row in recorded_baseline(run).items()}
    baseline.write_text(json.dumps(rows), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "eval",
            "gate",
            "--suite",
            "generation",
            "--results",
            str(JUDGED),
            "--baseline",
            str(baseline),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("### Generation gate: ⚠️ inconclusive\n")
    assert "hybrid: inconclusive." in result.output


# --- Report: Markdown and exit code (boilerplate, green) -----------------------------------------


NOT_GATED: dict[str, Any] = {"threshold": None, "passed": None}


def row(config: str, metric: str, **fields: Any) -> GateRow:
    values: dict[str, Any] = {
        "baseline": 0.9,
        "current": 0.88,
        "delta": -0.02,
        "threshold": 0.85,
        "passed": True,
        "n": 22,
    }
    return GateRow(config=config, metric=metric, **(values | fields))


def summary(config: str, **fields: Any) -> ConfigSummary:
    values: dict[str, Any] = {
        "gated": True,
        "inconclusive": False,
        "cases": 30,
        "n": 29,
        "n_faithfulness": 22,
        "errors": ErrorCounts(provider=1, generator_bad_output=1),
        "n_latency": 27,
        "latency_p50_ms": 812.4,
        "latency_p95_ms": 1533.6,
        "cost_per_1k_usd": 0.421,
    }
    return ConfigSummary(config=config, **(values | fields))


def test_markdown_shows_every_column_the_notice_the_summary_and_the_reasons() -> None:
    report = GenerationGateReport(
        status="fail",
        rows=[
            row("hybrid", "faithfulness"),
            row(
                "hybrid",
                "correctness",
                baseline=0.8,
                current=0.6,
                delta=-0.2,
                threshold=0.72,
                passed=False,
                n=28,
            ),
            row("no_rag", "correctness", baseline=0.5, current=0.4, delta=-0.1, **NOT_GATED, n=30),
            row(
                "no_rag", "faithfulness", baseline=None, current=None, delta=None, **NOT_GATED, n=0
            ),
        ],
        reasons=["hybrid: gated metric schema_first_try has no scored case"],
        configs=[
            summary("hybrid"),
            summary(
                "no_rag",
                gated=False,
                inconclusive=True,
                n=23,
                n_faithfulness=0,
                errors=ErrorCounts(provider=7),
                n_latency=0,
                latency_p50_ms=None,
                latency_p95_ms=None,
                cost_per_1k_usd=None,
            ),
        ],
    )
    assert render_generation_markdown(report) == (
        "### Generation gate: ❌ fail\n"
        "\n"
        "> ⚠️ **no_rag: inconclusive.** 7 of 30 cases (23.3%) errored for provider reasons "
        "(limit 20%: a quota, a 5xx or a timeout). Its metrics are shown with n but not gated, "
        "and an inconclusive run is not a pass: re-run it.\n"
        "\n"
        "| config | metric | baseline | current | Δ | threshold | n | |\n"
        "|---|---|---:|---:|---:|---:|---:|:-:|\n"
        "| hybrid | faithfulness | 0.900 | 0.880 | -0.020 | ≥ 0.850 | 22 | ✅ |\n"
        "| hybrid | correctness | 0.800 | 0.600 | -0.200 | ≥ 0.720 | 28 | ❌ |\n"
        "| no_rag | correctness | 0.500 | 0.400 | -0.100 | not gated | 30 | · |\n"
        "| no_rag | faithfulness | — | — | — | not gated | 0 | · |\n"
        "\n"
        "· = not gated: reported only (no thresholds, or the config is inconclusive).\n"
        "\n"
        "| config | cases | n | n faithfulness | provider errors | generator bad output "
        "| judge bad output | malformed | latency p50 / p95 (ms) | cost / 1k |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        "| hybrid | 30 | 29 | 22 | 1 | 1 | 0 | 0 | 812 / 1534 (n=27) | $0.4210 |\n"
        "| no_rag | 30 | 23 | 0 | 7 | 0 | 0 | 0 | — | — |\n"
        "\n"
        "- hybrid: gated metric schema_first_try has no scored case\n"
    )


def test_markdown_without_rows_or_configs_is_the_headline_and_the_reasons() -> None:
    report = GenerationGateReport(status="fail", rows=[], reasons=["nothing is gated"], configs=[])
    assert (
        render_generation_markdown(report) == "### Generation gate: ❌ fail\n\n- nothing is gated\n"
    )
    empty = GenerationGateReport(status="pass", rows=[], reasons=[], configs=[])
    assert render_generation_markdown(empty) == "### Generation gate: ✅ pass\n"


def test_markdown_has_no_notice_and_no_footnote_when_everything_was_gated_and_conclusive() -> None:
    report = GenerationGateReport(
        status="pass", rows=[row("hybrid", "faithfulness")], reasons=[], configs=[summary("hybrid")]
    )
    text = render_generation_markdown(report)
    assert "inconclusive" not in text
    assert "not gated" not in text


def test_the_inconclusive_notice_of_a_run_with_no_cases_does_not_divide_by_zero() -> None:
    empty = summary("hybrid", inconclusive=True, cases=0, n=0, errors=ErrorCounts())
    report = GenerationGateReport(status="inconclusive", rows=[], reasons=[], configs=[empty])
    assert "hybrid: inconclusive." in render_generation_markdown(report)


@pytest.mark.parametrize(
    ("status", "code"), [("pass", 0), ("inconclusive", 0), ("fail", 1)], ids=str
)
def test_exit_code_is_zero_unless_the_gate_fails(status: Any, code: int) -> None:
    report = GenerationGateReport(status=status, rows=[], reasons=[], configs=[])
    assert exit_code(report) == code
    assert render_generation_markdown(report).startswith(
        f"### Generation gate: {HEADLINE[status]} {status}\n"
    )


def test_a_gate_cannot_run_error_carries_its_reasons() -> None:
    error = GateCannotRunError(["ProviderRequestRejected: the key was refused", "second"])
    assert error.reasons == ["ProviderRequestRejected: the key was refused", "second"]
    assert "the key was refused" in str(error)


# --- CLI (boilerplate, green): evaluate_generation_gate is scripted ------------------------------


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    """A promptfoo results file and a baseline file that both validate."""
    results = tmp_path / "promptfoo.json"
    shutil.copyfile(JUDGED, results)
    baseline = tmp_path / "generation.json"
    baseline.write_text(json.dumps({"hybrid": entry().model_dump(mode="json")}), encoding="utf-8")
    return results, baseline


def script_gate(monkeypatch: pytest.MonkeyPatch, status: Any) -> list[tuple[Any, Any]]:
    calls: list[tuple[Any, Any]] = []

    def fake(run: GenerationRun, baseline: Any) -> GenerationGateReport:
        calls.append((run, baseline))
        return GenerationGateReport(
            status=status,
            rows=[row("hybrid", "faithfulness")],
            reasons=[],
            configs=[summary("hybrid")],
        )

    monkeypatch.setattr(grounded.cli, "evaluate_generation_gate", fake)
    return calls


def gate(results: Path, baseline: Path | None) -> Any:
    args = ["eval", "gate", "--suite", "generation", "--results", str(results)]
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
    assert result.output.startswith(f"### Generation gate: {HEADLINE[status]} {status}\n")
    assert "| hybrid | faithfulness |" in result.output
    [(run, rows)] = calls
    assert isinstance(run, GenerationRun)
    assert set(run.configs) == {"no_rag", "hybrid"}  # promptfoo's own pass/fail is not consulted
    assert set(rows) == {"hybrid"}


def test_gate_defaults_to_the_committed_generation_baseline(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    results, baseline = files
    monkeypatch.setattr(grounded.cli, "GENERATION_BASELINE", baseline)
    calls = script_gate(monkeypatch, "pass")
    assert gate(results, None).exit_code == 0
    assert set(calls[0][1]) == {"hybrid"}


def test_gate_exits_2_with_the_reasons_when_the_results_show_it_could_not_run(
    files: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(run: GenerationRun, baseline: Any) -> GenerationGateReport:
        raise GateCannotRunError(["ProviderRequestRejected: the judge key was refused"])

    monkeypatch.setattr(grounded.cli, "evaluate_generation_gate", refuse)
    result = gate(*files)
    assert result.exit_code == 2
    assert "The gate could not run" in result.output
    assert "- ProviderRequestRejected: the judge key was refused" in result.output


def test_gate_exits_2_when_the_results_file_is_missing(
    files: tuple[Path, Path], tmp_path: Path
) -> None:
    result = gate(tmp_path / "nope.json", files[1])
    assert result.exit_code == 2
    assert "Cannot read the results file" in result.output


@pytest.mark.parametrize("text", ["", "not json", '{"suite": "retrieval"}', '{"results": {}}'])
def test_gate_exits_2_when_the_results_file_is_not_promptfoos_output(
    files: tuple[Path, Path], tmp_path: Path, text: str
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(text, encoding="utf-8")
    result = gate(bad, files[1])
    assert result.exit_code == 2
    assert "Cannot read the results file" in result.output


def test_gate_rejects_a_baseline_row_without_a_required_field(files: tuple[Path, Path]) -> None:
    results, baseline = files
    rows = json.loads(baseline.read_bytes())
    del rows["hybrid"]["index_version"]
    baseline.write_text(json.dumps(rows), encoding="utf-8")
    result = gate(results, baseline)
    assert result.exit_code == 2
    assert "index_version" in result.output
    assert "baseline PR" in result.output


def test_gate_exits_2_when_the_baseline_file_is_missing(
    files: tuple[Path, Path], tmp_path: Path
) -> None:
    result = gate(files[0], tmp_path / "nope.json")
    assert result.exit_code == 2
    assert "Cannot read the baseline" in result.output


def test_gate_help_names_the_options_and_both_suites() -> None:
    result = runner.invoke(app, ["eval", "gate", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ("--suite", "--results", "--baseline", "generation"):
        assert option in plain
