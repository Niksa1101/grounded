"""The generation results schema and the promptfoo parser (Tech.md §15.3, ticket 4.07), against the
two recorded real outputs in ``tests/fixtures/promptfoo/`` (what ran and how: the README there).

The parser records and decides nothing about gating, so these tests pin what it keeps of every kind
of row (pass, failed assertion, not applicable, errored judge, provider error, skipped after a daily
quota) and that malformed input is a clear ``PromptfooResultsError``, not a crash.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from grounded.evals.generation_results import (
    GENERATION_BASELINE,
    PromptfooResultsError,
    parse_results,
    read_generation_baseline,
    read_generation_results,
)
from grounded.schemas.generation_eval import (
    ErrorInfo,
    GenerationBaselineEntry,
    GenerationCase,
    GenerationRun,
    GenerationThreshold,
    MetricResult,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo"
PLAIN = FIXTURES / "results_sample.json"  # 4.05: the deterministic assertions
JUDGED = FIXTURES / "results_sample_judge.json"  # 4.06: plus the two judge assertions
SHA = "8f0848177a5cb5a8ef649ee211ab9afb7f9bc07c73d37b37cbf3ef554e1abcb2"
QUESTIONS = ["q003", "q007", "q008", "q015", "q045", "q047", "q049"]


@pytest.fixture(scope="module")
def plain() -> GenerationRun:
    return parse_results(PLAIN.read_bytes())


@pytest.fixture(scope="module")
def judged() -> GenerationRun:
    return parse_results(JUDGED.read_bytes())


def case_of(run: GenerationRun, config: str, question: str) -> GenerationCase:
    [found] = [c for c in run.configs[config].cases if c.id == question]
    return found


def metric_of(run: GenerationRun, config: str, question: str, metric: str) -> MetricResult:
    return case_of(run, config, question).metrics[metric]


def error_of(result: MetricResult | GenerationCase) -> ErrorInfo:
    error = result.error
    assert error is not None
    return error


# --- The run -----------------------------------------------------------------------------------


def test_the_run_info_comes_from_the_file_and_the_unredacted_row_metadata(
    plain: GenerationRun, judged: GenerationRun
) -> None:
    for run in (plain, judged):
        assert run.suite == "generation"
        assert run.info.promptfoo_version == "0.123.1"
        assert run.info.golden_set_version == "v1"
        assert run.info.golden_set_sha256 == SHA  # testCase has "[REDACTED]" in its place
        assert (run.info.git_sha, run.info.git_dirty) == (None, None)  # not in promptfoo's file
    assert plain.info.date.isoformat() == "2026-10-08T10:58:25.127000+00:00"


def test_the_judge_identity_is_recorded_only_when_a_judge_ran(
    plain: GenerationRun, judged: GenerationRun
) -> None:
    assert (plain.info.judge_provider, plain.info.judge_model) == (None, None)
    assert plain.info.judge_prompt_versions == {}
    assert (judged.info.judge_provider, judged.info.judge_model) == ("fake-judge", "scripted-judge")
    assert judged.info.judge_prompt_versions == {
        "faithfulness": "judge_faithfulness_v1@84103412",
        "correctness": "judge_correctness_v1@1bde5fe4",
    }


def test_each_config_keeps_the_identity_of_its_answers(plain: GenerationRun) -> None:
    assert list(plain.configs) == ["no_rag", "hybrid"]
    no_rag, hybrid = plain.configs["no_rag"], plain.configs["hybrid"]
    assert (no_rag.provider, no_rag.model) == ("gemini", "gemini-3.5-flash-lite")
    assert no_rag.prompt_version == "answer_no_rag_v1@5f725a9d"
    assert (no_rag.index_version, no_rag.retrieval_config_hash) == ("none", "no_rag")
    assert hybrid.prompt_version == "answer_v1@08cc49e5"
    assert hybrid.index_version == "0.141.1@4949e8a3"
    assert hybrid.retrieval_config_hash is not None
    assert len(hybrid.retrieval_config_hash) == 64


def test_every_config_has_one_case_per_question_in_id_order(plain: GenerationRun) -> None:
    for config in plain.configs.values():
        assert [c.id for c in config.cases] == QUESTIONS
    assert case_of(plain, "hybrid", "q045").type == "unanswerable"
    assert case_of(plain, "hybrid", "q003").type == "factual"


def test_a_run_round_trips_through_json(plain: GenerationRun, judged: GenerationRun) -> None:
    for run in (plain, judged):
        assert GenerationRun.model_validate_json(run.model_dump_json()) == run


# --- Graded rows ---------------------------------------------------------------------------------


def test_a_passing_row_keeps_its_scores_latency_tokens_and_cost(plain: GenerationRun) -> None:
    case = case_of(plain, "hybrid", "q003")

    assert case.error is None
    assert {name: (m.state, m.value) for name, m in case.metrics.items()} == {
        "schema_first_try": ("scored", 1.0),
        "citation_validity": ("scored", 1.0),
        "refusal_correctness": ("scored", 1.0),
        "citation_precision": ("scored", 1.0),
    }
    assert (case.latency_ms, case.cold_start, case.llm_cache_hits) == (66.0, True, 0)
    assert (case.input_tokens, case.output_tokens) == (1200, 140)
    assert case.cost_usd == pytest.approx(0.00072485)
    assert case_of(plain, "hybrid", "q007").cold_start is False


def test_a_failed_assertion_is_a_score_not_an_error(plain: GenerationRun) -> None:
    # q007 cites an invented label: citation_validity is 0, and that is a measurement.
    assert metric_of(plain, "hybrid", "q007", "citation_validity") == MetricResult(
        state="scored", value=0.0
    )
    assert case_of(plain, "hybrid", "q007").error is None


def test_a_not_applicable_metric_has_no_value_whatever_promptfoo_stored(
    plain: GenerationRun, judged: GenerationRun
) -> None:
    for metric in ("citation_validity", "citation_precision"):
        assert metric_of(plain, "no_rag", "q003", metric) == MetricResult(state="na")
    assert metric_of(plain, "hybrid", "q045", "citation_precision").state == "na"
    # promptfoo's namedScores say faithfulness is 1 here (the placeholder); the parser drops it.
    row = json.loads(JUDGED.read_text(encoding="utf-8"))["results"]["results"][0]
    assert row["gradingResult"]["namedScores"]["faithfulness"] == 1
    assert metric_of(judged, "no_rag", "q003", "faithfulness") == MetricResult(state="na")
    assert metric_of(judged, "hybrid", "q045", "faithfulness") == MetricResult(state="na")


def test_a_graded_row_is_a_case_even_when_promptfoo_failed_it(judged: GenerationRun) -> None:
    # failureReason 1: a failed assertion, here correctness INCORRECT.
    case = case_of(judged, "no_rag", "q003")
    assert case.error is None
    assert metric_of(judged, "no_rag", "q003", "correctness") == MetricResult(
        state="scored", value=0.0
    )


def test_judge_scores_keep_the_claim_counts_for_faithfulness_only(judged: GenerationRun) -> None:
    faithfulness = metric_of(judged, "hybrid", "q003", "faithfulness")
    assert (faithfulness.state, faithfulness.value) == ("scored", 0.5)
    assert (faithfulness.claims, faithfulness.supported) == (2, 1)
    assert metric_of(judged, "hybrid", "q015", "faithfulness").value == 1.0
    correctness = metric_of(judged, "hybrid", "q003", "correctness")
    assert (correctness.value, correctness.claims, correctness.supported) == (0.5, None, None)


# --- Errors --------------------------------------------------------------------------------------


def test_a_judge_error_errors_that_metric_and_drops_the_placeholder_score(
    judged: GenerationRun,
) -> None:
    faithfulness = metric_of(judged, "hybrid", "q008", "faithfulness")

    assert faithfulness.state == "errored"
    assert faithfulness.value is None  # promptfoo's 0 is a placeholder
    assert (faithfulness.claims, faithfulness.supported) == (None, None)  # partly judged
    error = error_of(faithfulness)
    assert (error.stage, error.kind, error.provider_side, error.is_quota) == (
        "judge",
        "ProviderBadOutput",
        False,
        False,
    )
    assert "Invalid JSON" in error.detail
    # The case itself is graded: its other metrics are scored.
    assert case_of(judged, "hybrid", "q008").error is None
    assert metric_of(judged, "hybrid", "q008", "correctness").value == 1.0


def test_a_provider_side_judge_error_says_so(judged: GenerationRun) -> None:
    for config in ("no_rag", "hybrid"):
        error = error_of(metric_of(judged, config, "q015", "correctness"))
        assert (error.kind, error.provider_side, error.is_quota) == ("ProviderTimeout", True, False)
        assert error.skipped is False


def test_a_daily_quota_and_the_judge_assertions_skipped_after_it(judged: GenerationRun) -> None:
    quota = error_of(metric_of(judged, "hybrid", "q047", "correctness"))
    assert (quota.kind, quota.is_quota, quota.provider_side, quota.skipped) == (
        "ProviderRateLimited",
        True,
        True,
        False,
    )
    for config, question, metric in [
        ("hybrid", "q047", "faithfulness"),
        ("no_rag", "q049", "correctness"),
        ("hybrid", "q049", "faithfulness"),
        ("hybrid", "q049", "correctness"),
    ]:
        skipped = error_of(metric_of(judged, config, question, metric))
        assert (skipped.kind, skipped.is_quota, skipped.skipped) == (
            "ProviderRateLimited",
            True,
            True,
        )
    # N/A stays N/A after the stop: nothing was to be asked.
    assert metric_of(judged, "no_rag", "q049", "faithfulness").state == "na"


def test_a_failed_generator_call_is_an_error_case_with_no_metric(plain: GenerationRun) -> None:
    case = case_of(plain, "hybrid", "q008")

    error = error_of(case)
    assert (error.stage, error.kind, error.is_quota, error.skipped) == (
        "generator",
        "ProviderBadOutput",
        False,
        False,
    )
    assert "LLMAnswer" in error.detail  # the tag is stripped, the message kept
    # The parser records a quality miss, it does not score it: no schema_first_try = 0 here.
    assert case.metrics == {}
    assert case.latency_ms is None
    assert case.cost_usd == pytest.approx(0.0007637)  # what the failed attempts spent
    assert case.input_tokens > 0


def test_the_backoff_error_keeps_its_bases_and_a_skipped_row_says_it_was_not_asked(
    plain: GenerationRun,
) -> None:
    backoff = error_of(case_of(plain, "hybrid", "q015"))
    assert backoff.kind == "BackoffExhaustedError"
    assert backoff.bases == ("ProviderRateLimited", "ProviderError")
    assert (backoff.is_quota, backoff.provider_side) == (False, None)

    quota = error_of(case_of(plain, "hybrid", "q047"))
    skipped = error_of(case_of(plain, "hybrid", "q049"))
    assert (quota.kind, quota.is_quota, quota.skipped) == ("ProviderRateLimited", True, False)
    assert (skipped.kind, skipped.is_quota, skipped.skipped) == ("ProviderRateLimited", True, True)
    assert "skipped" in skipped.detail
    assert case_of(plain, "hybrid", "q049").cost_usd == 0


# --- Reading a row that is not quite what we write -----------------------------------------------


def edited(source: Path, change: Callable[[dict[str, Any]], None]) -> bytes:
    document = json.loads(source.read_text(encoding="utf-8"))
    change(document)
    return json.dumps(document).encode()


def row_of(document: dict[str, Any], config: str, question: str) -> dict[str, Any]:
    [found] = [
        r
        for r in document["results"]["results"]
        if r["provider"]["label"] == config and r["metadata"]["golden"]["id"] == question
    ]
    return found


def component_of(row: dict[str, Any], metric: str) -> dict[str, Any]:
    [found] = [
        c for c in row["gradingResult"]["componentResults"] if c["assertion"]["metric"] == metric
    ]
    return found


def test_an_error_row_without_the_metadata_falls_back_to_the_tag_in_its_text() -> None:
    def untag(document: dict[str, Any]) -> None:
        row = row_of(document, "hybrid", "q047")
        for key in ("error_kind", "error_bases", "is_quota"):
            del row["metadata"][key]

    error = error_of(case_of(parse_results(edited(PLAIN, untag)), "hybrid", "q047"))
    assert (error.kind, error.is_quota, error.bases) == ("ProviderRateLimited", True, ())
    assert error.detail == "daily quota exhausted"


def test_an_error_row_with_no_tag_at_all_is_an_unknown_error() -> None:
    def untag(document: dict[str, Any]) -> None:
        row = row_of(document, "hybrid", "q047")
        del row["metadata"]["error_kind"]
        row["error"] = "Python worker timed out"

    error = error_of(case_of(parse_results(edited(PLAIN, untag)), "hybrid", "q047"))
    assert (error.kind, error.detail) == ("UnknownError", "Python worker timed out")


def test_malformed_input_reported_by_a_deterministic_assertion_is_an_error_not_a_zero() -> None:
    def malform(document: dict[str, Any]) -> None:
        component = component_of(row_of(document, "hybrid", "q003"), "citation_precision")
        component.update(
            {"pass": False, "score": 0, "reason": "malformed input: cited chunk is not in context"}
        )

    metric = metric_of(
        parse_results(edited(PLAIN, malform)), "hybrid", "q003", "citation_precision"
    )
    assert metric.state == "errored"
    assert metric.value is None
    assert (error_of(metric).stage, error_of(metric).kind) == ("assertion", "MalformedInput")


def test_an_assertion_that_raised_has_none_of_our_keys_and_is_an_error() -> None:
    def crash(document: dict[str, Any]) -> None:
        component = component_of(row_of(document, "hybrid", "q003"), "schema_first_try")
        del component["not_applicable"]
        component.update(
            {"pass": False, "score": 0, "reason": "Python code execution failed: boom"}
        )

    metric = metric_of(parse_results(edited(PLAIN, crash)), "hybrid", "q003", "schema_first_try")
    assert metric.state == "errored"
    assert (error_of(metric).stage, error_of(metric).kind) == ("assertion", "AssertionFailed")
    assert "boom" in error_of(metric).detail


def test_the_reader_supplies_the_repo_state_that_promptfoos_file_lacks() -> None:
    run = parse_results(PLAIN.read_bytes(), git_sha="c" * 40, git_dirty=False)
    assert (run.info.git_sha, run.info.git_dirty) == ("c" * 40, False)


def test_a_results_file_is_read_from_disk_as_bytes(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_bytes(PLAIN.read_bytes())
    assert read_generation_results(path) == parse_results(PLAIN.read_bytes())
    with pytest.raises(FileNotFoundError):
        read_generation_results(tmp_path / "nope.json")


# --- Malformed input: a clear error, never a crash -----------------------------------------------


def no_rows(document: dict[str, Any]) -> None:
    document["results"]["results"] = []


def no_label(document: dict[str, Any]) -> None:
    del document["results"]["results"][0]["provider"]["label"]


def unknown_reason(document: dict[str, Any]) -> None:
    document["results"]["results"][0]["failureReason"] = 7


def no_grading(document: dict[str, Any]) -> None:
    row_of(document, "hybrid", "q003")["gradingResult"] = None


def duplicate_row(document: dict[str, Any]) -> None:
    rows = document["results"]["results"]
    rows.append(json.loads(json.dumps(rows[0])))


def other_golden_set(document: dict[str, Any]) -> None:
    row_of(document, "hybrid", "q003")["metadata"]["golden"]["golden_set_version"] = "v2"


def redacted_digest(document: dict[str, Any]) -> None:
    for row in document["results"]["results"]:
        row["metadata"]["golden"]["golden_set_sha256"] = "[REDACTED]"


def two_prompt_versions(document: dict[str, Any]) -> None:
    row_of(document, "hybrid", "q003")["metadata"]["prompt_version"] = "answer_v1@00000000"


def metric_twice(document: dict[str, Any]) -> None:
    components = row_of(document, "hybrid", "q003")["gradingResult"]["componentResults"]
    components.append(json.loads(json.dumps(components[0])))


def score_out_of_range(document: dict[str, Any]) -> None:
    component_of(row_of(document, "hybrid", "q003"), "schema_first_try")["score"] = 1.5


def no_version(document: dict[str, Any]) -> None:
    del document["metadata"]["promptfooVersion"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param(no_rows, "no result rows", id="no-rows"),
        pytest.param(no_label, "provider.label", id="row-without-a-config"),
        pytest.param(unknown_reason, "unknown failureReason", id="unknown-failure-reason"),
        pytest.param(no_grading, "no assertion results", id="graded-row-without-assertions"),
        pytest.param(duplicate_row, "appears twice", id="question-asked-twice"),
        pytest.param(other_golden_set, "different golden sets", id="two-golden-sets"),
        pytest.param(redacted_digest, "golden_set_sha256", id="redacted-digest"),
        pytest.param(two_prompt_versions, "differs between rows", id="two-prompt-versions"),
        pytest.param(metric_twice, "appears twice", id="metric-twice"),
        pytest.param(score_out_of_range, "less than or equal to 1", id="score-out-of-range"),
        pytest.param(no_version, "promptfooVersion", id="no-promptfoo-version"),
    ],
)
def test_a_file_that_is_not_one_coherent_run_is_a_parse_error(
    change: Callable[[dict[str, Any]], None], message: str
) -> None:
    with pytest.raises(PromptfooResultsError, match=message):
        parse_results(edited(PLAIN, change))


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not json", id="not-json"),
        pytest.param(b"\xff\xfe", id="not-utf8"),
        pytest.param(b"[]", id="an-array"),
        pytest.param(b"{}", id="an-empty-object"),
        pytest.param(b'{"suite": "retrieval", "info": {}, "configs": {}}', id="a-retrieval-run"),
        pytest.param(b'{"results": {"timestamp": "2026-10-08", "results": [{}]}}', id="empty-row"),
    ],
)
def test_input_that_is_not_a_promptfoo_file_is_a_clear_parse_error(raw: bytes) -> None:
    with pytest.raises(PromptfooResultsError, match="not a promptfoo results file"):
        parse_results(raw)


def test_a_parse_error_is_a_value_error_so_the_cli_can_catch_it() -> None:
    assert issubclass(PromptfooResultsError, ValueError)


# --- The model's own rules -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"state": "scored"}, id="scored-without-a-value"),
        pytest.param({"state": "na", "value": 1.0}, id="na-with-a-value"),
        pytest.param({"state": "errored", "value": 0.0}, id="errored-with-a-value"),
        pytest.param({"state": "errored"}, id="errored-without-an-error"),
        pytest.param(
            {"state": "scored", "value": 0.5, "error": ErrorInfo(stage="judge", kind="X")}
        ),
        pytest.param({"state": "scored", "value": 1.2}, id="value-above-one"),
    ],
)
def test_a_metric_result_has_a_value_exactly_when_scored_and_an_error_exactly_when_errored(
    fields: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        MetricResult.model_validate(fields)


def test_a_generator_error_means_no_assertion_ran() -> None:
    error = ErrorInfo(stage="generator", kind="ProviderTimeout")
    with pytest.raises(ValidationError, match="no assertion ran"):
        GenerationCase(
            id="q001",
            type="factual",
            error=error,
            metrics={"correctness": MetricResult(state="na")},
        )
    with pytest.raises(ValidationError, match="no assertion ran"):
        GenerationCase(id="q001", type="factual", error=ErrorInfo(stage="judge", kind="X"))


# --- The baseline file ---------------------------------------------------------------------------


def baseline_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "metrics": {"faithfulness": 0.9, "correctness": 0.8},
        "n": {"faithfulness": 22, "correctness": 28},
        "thresholds": {
            "faithfulness": {"floor": 0.85, "tolerance": 0.05},
            "correctness": {"tolerance": 0.08},
        },
        "cases": 30,
        "golden_set_version": "v1",
        "golden_set_sha256": SHA,
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "answer_v1@08cc49e5",
        "index_version": "0.141.1@4949e8a3",
        "judge_provider": "groq",
        "judge_model": "openai/gpt-oss-120b",
        "judge_prompt_versions": {"faithfulness": "judge_faithfulness_v1@84103412"},
        "promptfoo_version": "0.123.1",
        "git_sha": "c" * 40,
        "git_dirty": False,
        "date": "2026-10-08T12:00:00Z",
    }
    return row | overrides


def test_a_baseline_row_loads_and_keeps_a_floor_a_tolerance_or_both() -> None:
    row = GenerationBaselineEntry.model_validate(baseline_row())
    assert row.thresholds["faithfulness"] == GenerationThreshold(floor=0.85, tolerance=0.05)
    assert row.thresholds["correctness"] == GenerationThreshold(tolerance=0.08)
    assert row.retrieval_config_hash is None  # optional: no gate rule uses it


@pytest.mark.parametrize("rule", [{}, {"tolerance": -0.01}, {"floor": 1.5}, {"floor": -0.1}])
def test_a_threshold_needs_a_valid_tolerance_or_floor(rule: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        GenerationThreshold.model_validate(rule)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("thresholds", {"mrr": {"tolerance": 0.04}}, id="threshold-for-no-metric"),
        pytest.param("n", {"faithfulness": 22}, id="n-for-fewer-metrics"),
        pytest.param("n", {"faithfulness": 22, "correctness": 0}, id="n-of-zero"),
        pytest.param("golden_set_sha256", "[REDACTED]", id="redacted-digest"),
    ],
)
def test_a_baseline_row_is_checked_against_itself(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        GenerationBaselineEntry.model_validate(baseline_row(**{field: value}))


@pytest.mark.parametrize("field", ["golden_set_sha256", "index_version", "git_dirty", "n", "model"])
def test_a_baseline_row_needs_what_the_gate_compares_and_the_git_state(field: str) -> None:
    row = baseline_row()
    del row[field]
    with pytest.raises(ValidationError, match=field):
        GenerationBaselineEntry.model_validate(row)


def test_the_baseline_file_is_read_as_rows_by_config(tmp_path: Path) -> None:
    path = tmp_path / "generation.json"
    path.write_text(json.dumps({"hybrid": baseline_row()}), encoding="utf-8")
    assert set(read_generation_baseline(path)) == {"hybrid"}

    path.write_text(json.dumps({"hybrid": {**baseline_row(), "git_dirty": None}}), encoding="utf-8")
    with pytest.raises(ValueError, match="git_dirty"):
        read_generation_baseline(path)


def test_the_baseline_lives_next_to_the_retrieval_one() -> None:
    assert GENERATION_BASELINE.parts[-3:] == ("eval", "baselines", "generation.json")
