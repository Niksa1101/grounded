"""The ``eval_runs`` rows of a generation run (``evals/eval_record.py``, ticket 4.10c): what a row
holds, the verdict it carries, and what it must never hold.

Runs are hand-built with the expected numbers worked out in the comments. The database side
(``insert_rows``, ``read_active_index``, the command) is in
``tests/integration/test_eval_record.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from grounded.evals.eval_record import (
    NO_INDEX,
    ActiveIndex,
    EvalRecordError,
    EvalRunRow,
    build_rows,
)
from grounded.evals.gate import GateCannotRunError
from grounded.evals.generation_baseline import update_generation_baseline
from grounded.evals.generation_results import read_generation_baseline
from grounded.schemas.generation_eval import (
    ErrorInfo,
    GenerationBaselineEntry,
    GenerationCase,
    GenerationConfigResult,
    GenerationRun,
    GenerationRunInfo,
    MetricResult,
)

SHA = "8f0848177a5cb5a8ef649ee211ab9afb7f9bc07c73d37b37cbf3ef554e1abcb2"
HEAD = "c" * 40
CONFIG_HASH = "4949e8a3" + "0" * 56
ACTIVE = ActiveIndex(git_ref="0.141.1", config_hash=CONFIG_HASH)
LABEL = "0.141.1@4949e8a3"
URL = "https://github.com/o/r/actions/runs/1"

NA = MetricResult(state="na")


def scored(value: float) -> MetricResult:
    return MetricResult(state="scored", value=value)


def case(index: int, metrics: Mapping[str, float | MetricResult], **fields: Any) -> GenerationCase:
    return GenerationCase(
        id=f"q{index:03d}",
        type="factual",
        metrics={k: v if isinstance(v, MetricResult) else scored(v) for k, v in metrics.items()},
        latency_ms=100.0 + index,
        cost_usd=0.001,
        **fields,
    )


def provider_error(index: int) -> GenerationCase:
    error = ErrorInfo(stage="generator", kind="ProviderRateLimited", is_quota=True, detail="x")
    return GenerationCase(id=f"q{index:03d}", type="factual", error=error, cost_usd=0.0)


def hybrid_cases(correctness: float = 1.0, n: int = 4) -> list[GenerationCase]:
    return [
        case(
            i,
            {
                "schema_first_try": 1.0,
                "citation_validity": 1.0,
                "refusal_correctness": 1.0,
                "citation_precision": 1.0,
                "faithfulness": 1.0,
                "correctness": correctness,
            },
        )
        for i in range(1, n + 1)
    ]


def no_rag_cases(n: int = 4) -> list[GenerationCase]:
    # No retrieval: faithfulness and the citation metrics are never applicable.
    return [
        case(
            i,
            {
                "schema_first_try": 1.0,
                "citation_validity": NA,
                "refusal_correctness": 1.0,
                "citation_precision": NA,
                "faithfulness": NA,
                "correctness": 0.5,
            },
        )
        for i in range(1, n + 1)
    ]


IDENTITY: dict[str, dict[str, Any]] = {
    "hybrid": {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "answer_v1@08cc49e5",
        "index_version": LABEL,
        "retrieval_config_hash": "2" * 64,
    },
    "no_rag": {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "answer_no_rag_v1@5f725a9d",
        "index_version": "none",
        "retrieval_config_hash": "no_rag",
    },
}


def make_run(
    hybrid: list[GenerationCase] | None = None,
    no_rag: list[GenerationCase] | None = None,
    *,
    info: Mapping[str, Any] | None = None,
    identity: Mapping[str, Mapping[str, Any]] | None = None,
) -> GenerationRun:
    chosen = {"no_rag": no_rag or no_rag_cases(), "hybrid": hybrid or hybrid_cases()}
    run_info = GenerationRunInfo.model_validate(
        {
            "date": datetime(2026, 10, 9, 8, 0, tzinfo=UTC),
            "promptfoo_version": "0.123.1",
            "git_sha": HEAD,
            "git_dirty": False,
            "golden_set_version": "v1",
            "golden_set_sha256": SHA,
            "judge_provider": "groq",
            "judge_model": "openai/gpt-oss-120b",
            "judge_prompt_versions": {
                "faithfulness": "judge_faithfulness_v1@84103412",
                "correctness": "judge_correctness_v1@1bde5fe4",
            },
        }
        | dict(info or {})
    )
    return GenerationRun(
        info=run_info,
        configs={
            name: GenerationConfigResult.model_validate(
                {"config": name, "cases": cases}
                | IDENTITY[name]
                | dict((identity or {}).get(name, {}))
            )
            for name, cases in chosen.items()
        },
    )


@pytest.fixture
def baseline(tmp_path: Path) -> dict[str, GenerationBaselineEntry]:
    """The baseline of the default run: hybrid correctness 1.0 (gated, floor/tolerance from the
    initial thresholds), no_rag reported only."""
    path = tmp_path / "generation.json"
    update_generation_baseline(path, make_run())
    return read_generation_baseline(path)


def rows_of(
    run: GenerationRun,
    baseline: Mapping[str, GenerationBaselineEntry],
    *,
    active: ActiveIndex | None = ACTIVE,
    report_url: str | None = URL,
) -> dict[str, EvalRunRow]:
    rows = build_rows(
        run, baseline, git_sha=HEAD, branch="main", report_url=report_url, active_index=active
    )
    return {row.config_name: row for row in rows}


# --- What a row holds ----------------------------------------------------------------------------


def test_one_row_per_config_with_the_identity_of_the_run(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    rows = rows_of(make_run(), baseline)

    assert list(rows) == ["hybrid", "no_rag"]  # sorted: the same rows whatever the dict order
    hybrid = rows["hybrid"]
    assert (hybrid.suite, hybrid.git_sha, hybrid.branch) == ("generation", HEAD, "main")
    assert hybrid.golden_set_version == "v1"
    assert hybrid.prompt_version == "answer_v1@08cc49e5"
    assert hybrid.generator_model == "gemini-3.5-flash-lite"
    assert hybrid.judge_model == "openai/gpt-oss-120b"
    assert hybrid.report_url == URL
    assert rows["no_rag"].prompt_version == "answer_no_rag_v1@5f725a9d"


def test_the_index_is_the_full_hash_of_the_active_one_and_none_without_retrieval(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    rows = rows_of(make_run(), baseline)

    assert rows["hybrid"].index_config_hash == CONFIG_HASH  # 64 characters, not the 8 of the label
    assert rows["no_rag"].index_config_hash == NO_INDEX == "none"


def test_metrics_are_flat_with_the_n_of_each_metric_and_leave_out_what_was_never_scored(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    rows = rows_of(make_run(), baseline)

    # Hybrid: four perfect answers. Cost is 0.001 per answer = 1.0 per 1k; latency 101..104.
    assert rows["hybrid"].metrics == pytest.approx(
        {
            "citation_precision": 1.0,
            "n_citation_precision": 4,
            "citation_validity": 1.0,
            "n_citation_validity": 4,
            "correctness": 1.0,
            "n_correctness": 4,
            "faithfulness": 1.0,
            "n_faithfulness": 4,
            "refusal_correctness": 1.0,
            "n_refusal_correctness": 4,
            "schema_first_try": 1.0,
            "n_schema_first_try": 4,
            "n": 4,
            "latency_p50_ms": 102.0,
            "latency_p95_ms": 104.0,
            "n_latency": 4,
            "cost_per_1k_usd": 1.0,
        }
    )
    # no_rag: faithfulness and the citation metrics are never applicable, so they have no key at
    # all (not 0, not 1); correctness is 0.5 on all four.
    assert rows["no_rag"].metrics["correctness"] == pytest.approx(0.5)
    assert rows["no_rag"].metrics["n_correctness"] == 4
    assert (
        not {"faithfulness", "n_faithfulness", "citation_precision"} & rows["no_rag"].metrics.keys()
    )


def test_case_counts_are_the_cases_asked_and_the_ones_that_errored_for_the_provider(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    run = make_run(
        hybrid=[*hybrid_cases(n=9), provider_error(10)]
    )  # 1 of 10 = 10%: not inconclusive
    rows = rows_of(run, baseline)

    assert (rows["hybrid"].case_count, rows["hybrid"].errored_case_count) == (10, 1)
    assert rows["hybrid"].metrics["n"] == 9  # the errored case is out of n
    assert rows["hybrid"].status == "pass"
    assert (rows["no_rag"].case_count, rows["no_rag"].errored_case_count) == (4, 0)


# --- The verdict ---------------------------------------------------------------------------------


def test_a_run_that_matches_its_baseline_is_a_pass_on_every_row(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    assert {row.status for row in rows_of(make_run(), baseline).values()} == {"pass"}


def test_a_regression_is_a_fail_on_every_row_and_the_numbers_are_still_recorded(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    rows = rows_of(make_run(hybrid=hybrid_cases(correctness=0.5)), baseline)  # 1.0 -> 0.5

    assert {row.status for row in rows.values()} == {"fail"}  # the run's verdict, not per config
    assert rows["hybrid"].metrics["correctness"] == pytest.approx(0.5)


def test_a_quota_dominated_config_is_inconclusive_and_keeps_its_numbers_and_error_count(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    # 3 of 5 hybrid cases errored (60% > 20%): inconclusive, metrics shown but not gated.
    run = make_run(
        hybrid=[*hybrid_cases(n=2), provider_error(3), provider_error(4), provider_error(5)]
    )
    rows = rows_of(run, baseline)

    assert {row.status for row in rows.values()} == {"inconclusive"}
    assert (rows["hybrid"].case_count, rows["hybrid"].errored_case_count) == (5, 3)
    assert rows["hybrid"].metrics["n"] == 2
    assert rows["hybrid"].metrics["correctness"] == pytest.approx(1.0)


def test_a_run_on_another_golden_set_is_a_fail_but_still_has_its_own_numbers(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    rows = rows_of(
        make_run(info={"golden_set_sha256": "a" * 64, "golden_set_version": "v2"}), baseline
    )

    assert {row.status for row in rows.values()} == {"fail"}  # not comparable with the baseline
    assert rows["hybrid"].golden_set_version == "v2"  # and the row says which set it was
    assert rows["hybrid"].metrics["correctness"] == pytest.approx(1.0)


def test_a_run_the_gate_cannot_judge_raises_instead_of_recording_it(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    rejected = ErrorInfo(stage="generator", kind="ProviderRequestRejected", detail="bad key")
    run = make_run(
        hybrid=[GenerationCase(id="q001", type="factual", error=rejected), *hybrid_cases(n=3)[1:]]
    )

    with pytest.raises(GateCannotRunError):
        rows_of(run, baseline)


# --- The index -----------------------------------------------------------------------------------


def test_a_run_made_on_another_index_than_the_active_one_is_refused(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    other = ActiveIndex(git_ref="0.141.1", config_hash="beef" + "0" * 60)

    with pytest.raises(EvalRecordError, match=r"0\.141\.1@4949e8a3.*0\.141\.1@beef0000"):
        rows_of(make_run(), baseline, active=other)


def test_without_an_active_index_only_a_run_with_no_retrieval_can_be_recorded(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    with pytest.raises(EvalRecordError, match="hybrid: the active index"):
        rows_of(make_run(), baseline, active=None)

    only_no_rag = make_run()
    only_no_rag = only_no_rag.model_copy(
        update={"configs": {"no_rag": only_no_rag.configs["no_rag"]}}
    )
    assert rows_of(only_no_rag, baseline, active=None)["no_rag"].index_config_hash == "none"


def test_a_config_whose_calls_all_failed_takes_the_only_active_index(
    baseline: dict[str, GenerationBaselineEntry],
) -> None:
    # Every hybrid call failed, so no answer named its index: the CI database's active one is the
    # only index the run could have used. (The status is the gate's business and is not asserted.)
    unnamed = {"hybrid": {"index_version": None, "prompt_version": None, "model": None}}
    run = make_run(hybrid=[provider_error(i) for i in range(1, 5)], identity=unnamed)
    rows = rows_of(run, baseline)

    assert rows["hybrid"].index_config_hash == CONFIG_HASH
    assert rows["hybrid"].prompt_version is None
    assert rows["hybrid"].generator_model is None
    assert (rows["hybrid"].case_count, rows["hybrid"].errored_case_count) == (4, 4)


# --- What a row never holds ----------------------------------------------------------------------


def test_a_row_is_aggregates_only(baseline: dict[str, GenerationBaselineEntry]) -> None:
    dumped = json.dumps([row.model_dump() for row in rows_of(make_run(), baseline).values()])

    assert "q00" not in dumped  # no question id (the cases are q001..q004)
    for row in rows_of(make_run(), baseline).values():
        assert all(isinstance(v, int | float) for v in row.metrics.values())


def test_a_row_is_validated_like_the_columns_it_goes_into() -> None:
    good: dict[str, Any] = {
        "suite": "generation",
        "config_name": "hybrid",
        "git_sha": HEAD,
        "branch": "main",
        "golden_set_version": "v1",
        "prompt_version": None,
        "index_config_hash": CONFIG_HASH,
        "generator_model": None,
        "judge_model": None,
        "status": "pass",
        "metrics": {"n": 1},
        "case_count": 1,
        "errored_case_count": 0,
        "report_url": None,
    }
    EvalRunRow.model_validate(good)
    for broken in (
        good | {"git_sha": "abc"},  # char(40): a full SHA
        good | {"git_sha": "C" * 40},  # git prints lowercase
        good | {"status": "ok"},  # the CHECK constraint
        good | {"suite": "retrieval"},  # this module records the generation suite
        good | {"branch": ""},
        good | {"case_count": -1},
        good | {"extra": 1},
    ):
        with pytest.raises(ValidationError):
            EvalRunRow.model_validate(broken)
