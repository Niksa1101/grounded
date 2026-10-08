"""The generation baseline writer (Tech.md §15.7, ticket 4.09a): ``update_generation_baseline`` and
``grounded eval baseline``.

Runs are hand-built, with the expected rows worked out on paper in the comments, or are the
committed promptfoo sample cut down to the questions that have no error. The samples are scripted
data: they test the mechanics (parse, write, read back, gate), never an eval result. Every test
writes into ``tmp_path``; none touches ``eval/baselines/``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.gate import evaluate_generation_gate
from grounded.evals.generation_baseline import (
    GATED_CONFIG,
    INITIAL_THRESHOLDS,
    BaselineUpdate,
    GenerationBaselineError,
    check_against_itself,
    update_generation_baseline,
)
from grounded.evals.generation_results import parse_results, read_generation_baseline
from grounded.schemas.generation_eval import (
    ErrorInfo,
    GenerationBaselineEntry,
    GenerationCase,
    GenerationConfigResult,
    GenerationRun,
    GenerationRunInfo,
    GenerationThreshold,
    MetricResult,
)

NOW = datetime(2026, 10, 9, 8, 0, 0, tzinfo=UTC)
SHA = "8f0848177a5cb5a8ef649ee211ab9afb7f9bc07c73d37b37cbf3ef554e1abcb2"  # the samples' golden set
OTHER_SHA = "a" * 64
HEAD = "c" * 40
INDEX = "0.141.1@4949e8a3"
SAMPLE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample_judge.json"
)

runner = CliRunner()

# Tech §15.5, "Initial thresholds", typed again here on purpose: the test pins the constant to it.
TECH_15_5 = {
    "faithfulness": GenerationThreshold(floor=0.85, tolerance=0.05),
    "correctness": GenerationThreshold(tolerance=0.08),
    "refusal_correctness": GenerationThreshold(tolerance=0.07),
    "schema_first_try": GenerationThreshold(floor=0.95),
}

# --- Builders ------------------------------------------------------------------------------------


def scored(value: float) -> MetricResult:
    return MetricResult(state="scored", value=value)


NA = MetricResult(state="na")


def failed(kind: str, *, stage: str = "judge", provider_side: bool | None = None) -> MetricResult:
    """A metric that errored (a judge or assertion error)."""
    error = ErrorInfo.model_validate(
        {"stage": stage, "kind": kind, "provider_side": provider_side, "detail": "x"}
    )
    return MetricResult(state="errored", error=error)


def generator_error(kind: str, *, is_quota: bool = False, skipped: bool = False) -> ErrorInfo:
    return ErrorInfo(stage="generator", kind=kind, is_quota=is_quota, skipped=skipped, detail="x")


def make_case(
    index: int,
    metrics: Mapping[str, float | MetricResult],
    *,
    latency_ms: float = 100.0,
    cold_start: bool = False,
    cost_usd: float = 0.001,
) -> GenerationCase:
    return GenerationCase(
        id=f"q{index:03d}",
        type="factual",
        metrics={k: v if isinstance(v, MetricResult) else scored(v) for k, v in metrics.items()},
        latency_ms=latency_ms,
        cold_start=cold_start,
        cost_usd=cost_usd,
    )


def errored_case(index: int, error: ErrorInfo) -> GenerationCase:
    return GenerationCase(id=f"q{index:03d}", type="factual", error=error, cost_usd=0.001)


def hybrid_case(index: int, **metrics: float | MetricResult) -> GenerationCase:
    """A perfect answer unless told otherwise. Latency is 100 + 10 * index; the first is cold."""
    perfect: dict[str, float | MetricResult] = {
        "schema_first_try": 1.0,
        "citation_validity": 1.0,
        "refusal_correctness": 1.0,
        "citation_precision": 1.0,
        "faithfulness": 1.0,
        "correctness": 1.0,
    }
    return make_case(index, perfect | metrics, latency_ms=100.0 + 10 * index, cold_start=index == 1)


def hybrid_cases(**replacing: GenerationCase) -> list[GenerationCase]:
    """Ten hybrid cases. Worked out by hand (the baseline of ``default_run``):

    - faithfulness: q001-q007 = 1, q008 = 0.5, q009 and q010 not applicable -> 7.5 / 8 = 0.9375, n 8
    - correctness: 7 x 1, 0.5, 1, 0 -> 8.5 / 10 = 0.85, n 10
    - refusal accuracy: q010 = 0 -> 0.9, n 10; schema first-try and citation validity: 1.0, n 10
    - citation precision: q001-q008 = 1, the two refusals not applicable -> 1.0, n 8
    - latency (the cold q001 is left out): 120, 130 ... 200 -> p50 = 5th = 160, p95 = 9th = 200, n 9
    - cost: 0.001 per answer -> 1.0 per 1k, n 10
    """
    cases = [hybrid_case(i) for i in range(1, 11)]
    cases[7] = hybrid_case(8, faithfulness=0.5, correctness=0.5)
    cases[8] = hybrid_case(9, faithfulness=NA, citation_precision=NA)
    cases[9] = hybrid_case(
        10, faithfulness=NA, citation_precision=NA, refusal_correctness=0.0, correctness=0.0
    )
    for key, case in replacing.items():  # hybrid_cases(q004=...)
        cases[int(key[1:]) - 1] = case
    return cases


def no_rag_case(index: int, **metrics: float | MetricResult) -> GenerationCase:
    base: dict[str, float | MetricResult] = {
        "schema_first_try": 1.0,
        "citation_validity": NA,
        "refusal_correctness": 1.0,
        "citation_precision": NA,
        "faithfulness": NA,
        "correctness": 1.0,
    }
    return make_case(index, base | metrics, latency_ms=50.0, cost_usd=0.0005)


def no_rag_cases() -> list[GenerationCase]:
    """Ten no_rag cases. By hand: correctness 5 x 1, 3 x 0.5, 0, 0 -> 6.5 / 10 = 0.65; refusal
    accuracy: q009 = 0 -> 0.9; faithfulness and the citation metrics are never applicable, so the
    row has no entry for them; latency 50 / 50 (n 10); cost 0.0005 -> 0.5 per 1k (n 10)."""
    cases = [no_rag_case(i) for i in range(1, 11)]
    for i in (6, 7, 8):
        cases[i - 1] = no_rag_case(i, correctness=0.5)
    cases[8] = no_rag_case(9, correctness=0.0, refusal_correctness=0.0)
    cases[9] = no_rag_case(10, correctness=0.0)
    return cases


IDENTITY: dict[str, dict[str, Any]] = {
    "hybrid": {
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "answer_v1@08cc49e5",
        "index_version": INDEX,
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
    configs: Mapping[str, list[GenerationCase]] | None = None,
    *,
    info: Mapping[str, Any] | None = None,
    identity: Mapping[str, Mapping[str, Any]] | None = None,
) -> GenerationRun:
    """The default run unless told otherwise: ``info`` overrides the run's fields and
    ``identity[name]`` the answers' identity of one config."""
    chosen = {"no_rag": no_rag_cases(), "hybrid": hybrid_cases()} if configs is None else configs
    run_info = GenerationRunInfo.model_validate(
        {
            "date": NOW,
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
                | IDENTITY.get(name, IDENTITY["hybrid"])
                | dict((identity or {}).get(name, {}))
            )
            for name, cases in chosen.items()
        },
    )


def write(tmp_path: Path, run: GenerationRun | None = None) -> tuple[Path, BaselineUpdate]:
    path = tmp_path / "generation.json"
    return path, update_generation_baseline(path, run or make_run())


def on_disk(path: Path) -> dict[str, Any]:
    return json.loads(path.read_bytes())


def clean_sample() -> bytes:
    """The 4.06 promptfoo recording cut to q003, q007 and q045: the questions of both configs with
    no provider or judge error (the others carry a timeout, a quota and a bad output on purpose)."""
    document = json.loads(SAMPLE.read_bytes())
    rows = document["results"]["results"]
    document["results"]["results"] = [
        row for row in rows if row["metadata"]["golden"]["id"] in {"q003", "q007", "q045"}
    ]
    return json.dumps(document).encode()


# --- Happy path ----------------------------------------------------------------------------------


def test_a_first_write_copies_the_numbers_of_the_run_with_their_n(tmp_path: Path) -> None:
    path, update = write(tmp_path)

    assert update.written == ("no_rag", "hybrid")
    assert list(on_disk(path)) == ["hybrid", "no_rag"]  # sorted: the same file whatever the order
    hybrid = read_generation_baseline(path)["hybrid"]
    assert hybrid.metrics == pytest.approx(
        {
            "citation_precision": 1.0,
            "citation_validity": 1.0,
            "correctness": 0.85,
            "faithfulness": 0.9375,
            "refusal_correctness": 0.9,
            "schema_first_try": 1.0,
        }
    )
    assert hybrid.n == {
        "citation_precision": 8,
        "citation_validity": 10,
        "correctness": 10,
        "faithfulness": 8,
        "refusal_correctness": 10,
        "schema_first_try": 10,
    }
    assert hybrid.cases == 10
    assert (hybrid.latency_p50_ms, hybrid.latency_p95_ms, hybrid.n_latency) == (160, 200, 9)
    assert (hybrid.cost_per_1k_usd, hybrid.n_cost) == (pytest.approx(1.0), 10)

    no_rag = read_generation_baseline(path)["no_rag"]
    assert no_rag.metrics == pytest.approx(
        {"correctness": 0.65, "refusal_correctness": 0.9, "schema_first_try": 1.0}
    )
    assert no_rag.n == {"correctness": 10, "refusal_correctness": 10, "schema_first_try": 10}
    assert (no_rag.latency_p50_ms, no_rag.latency_p95_ms, no_rag.n_latency) == (50, 50, 10)
    assert (no_rag.cost_per_1k_usd, no_rag.n_cost) == (pytest.approx(0.5), 10)


def test_a_row_records_where_the_numbers_came_from(tmp_path: Path) -> None:
    path, _ = write(tmp_path, make_run(info={"git_dirty": True}))

    rows = on_disk(path)
    hybrid, no_rag = rows["hybrid"], rows["no_rag"]
    assert hybrid["golden_set_version"] == "v1"
    assert hybrid["golden_set_sha256"] == SHA
    assert (hybrid["provider"], hybrid["model"]) == ("gemini", "gemini-3.5-flash-lite")
    assert hybrid["prompt_version"] == "answer_v1@08cc49e5"
    assert hybrid["index_version"] == INDEX
    assert hybrid["retrieval_config_hash"] == "2" * 64
    assert no_rag["prompt_version"] == "answer_no_rag_v1@5f725a9d"
    assert no_rag["index_version"] == "none"
    for row in (hybrid, no_rag):
        assert (row["judge_provider"], row["judge_model"]) == ("groq", "openai/gpt-oss-120b")
        assert row["judge_prompt_versions"] == {
            "faithfulness": "judge_faithfulness_v1@84103412",
            "correctness": "judge_correctness_v1@1bde5fe4",
        }
        assert row["promptfoo_version"] == "0.123.1"
        assert (row["git_sha"], row["git_dirty"]) == (HEAD, True)
        assert row["date"].startswith("2026-10-09T08:00:00")


def test_the_file_is_utf8_with_lf_endings_and_the_same_for_the_same_run(tmp_path: Path) -> None:
    first, _ = write(tmp_path / "a")
    second, _ = write(
        tmp_path / "b", make_run({"hybrid": hybrid_cases(), "no_rag": no_rag_cases()})
    )
    assert first.read_bytes() == second.read_bytes()  # config order in the run does not matter
    assert b"\r" not in first.read_bytes()
    assert first.read_bytes().endswith(b"}\n")


def test_the_numbers_are_the_gates_numbers(tmp_path: Path) -> None:
    run = make_run()
    path, _ = write(tmp_path, run)
    rows = read_generation_baseline(path)

    report = evaluate_generation_gate(run, {})
    for gate_row in report.rows:
        if gate_row.current is None:
            assert gate_row.metric not in rows[gate_row.config].metrics
        else:
            assert rows[gate_row.config].metrics[gate_row.metric] == gate_row.current
            assert rows[gate_row.config].n[gate_row.metric] == gate_row.n


# --- Thresholds: policy, from Tech §15.5, written once --------------------------------------------


def test_the_initial_thresholds_are_the_table_of_tech_15_5() -> None:
    assert dict(INITIAL_THRESHOLDS) == TECH_15_5
    assert GATED_CONFIG == "hybrid"


def test_a_new_hybrid_row_gets_the_initial_thresholds_and_no_rag_none(tmp_path: Path) -> None:
    path, update = write(tmp_path)

    assert update.seeded == ("hybrid",)
    rows = read_generation_baseline(path)
    assert rows["hybrid"].thresholds == TECH_15_5
    assert rows["no_rag"].thresholds == {}
    # Written as the rules read in Tech §15.7: only the parts that are set, no nulls.
    assert on_disk(path)["hybrid"]["thresholds"] == {
        "correctness": {"tolerance": 0.08},
        "faithfulness": {"tolerance": 0.05, "floor": 0.85},
        "refusal_correctness": {"tolerance": 0.07},
        "schema_first_try": {"floor": 0.95},
    }


def test_a_rewrite_keeps_the_thresholds_the_file_has_and_refreshes_the_numbers(
    tmp_path: Path,
) -> None:
    path, _ = write(tmp_path)
    rows = on_disk(path)
    rows["hybrid"]["thresholds"] = {"correctness": {"tolerance": 0.10}}  # edited by hand
    path.write_text(json.dumps(rows), encoding="utf-8")

    worse = [hybrid_case(i, correctness=0.5) for i in range(1, 11)]
    _, update = write(tmp_path, make_run({"hybrid": worse, "no_rag": no_rag_cases()}))

    assert update.seeded == ()
    hybrid = read_generation_baseline(path)["hybrid"]
    assert hybrid.thresholds == {"correctness": GenerationThreshold(tolerance=0.10)}
    assert hybrid.metrics["correctness"] == pytest.approx(0.5)


def test_a_row_the_author_made_reported_only_stays_reported_only(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    rows = on_disk(path)
    rows["hybrid"]["thresholds"] = {}
    path.write_text(json.dumps(rows), encoding="utf-8")

    _, update = write(tmp_path)

    assert update.seeded == ()
    assert read_generation_baseline(path)["hybrid"].thresholds == {}


def test_a_rewrite_of_one_config_keeps_the_others_byte_for_byte(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    before = on_disk(path)

    better = [hybrid_case(i) for i in range(1, 11)]
    write(tmp_path, make_run({"hybrid": better}))
    after = on_disk(path)
    assert after["no_rag"] == before["no_rag"]
    assert after["hybrid"]["metrics"]["correctness"] == 1.0

    write(tmp_path, make_run({"no_rag": [no_rag_case(i) for i in range(1, 11)]}))
    again = on_disk(path)
    assert again["hybrid"] == after["hybrid"]
    assert again["no_rag"]["metrics"]["correctness"] == 1.0


def test_a_new_config_next_to_the_kept_ones_gets_no_thresholds(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    rerank = make_run({"hybrid_rerank": hybrid_cases()})
    update_generation_baseline(path, rerank)

    rows = read_generation_baseline(path)
    assert list(rows) == ["hybrid", "hybrid_rerank", "no_rag"]
    assert rows["hybrid_rerank"].thresholds == {}  # only hybrid is seeded; Phase 6 decides the rest


@pytest.mark.parametrize("metric", ["faithfulness", "correctness"])
def test_a_threshold_on_a_metric_the_run_could_not_score_is_refused(
    tmp_path: Path, metric: str
) -> None:
    """A first hybrid row needs all four gated metrics; so does a kept rule on any metric."""
    nothing: dict[str, float | MetricResult] = {metric: NA}
    unscored = [hybrid_case(i, **nothing) for i in range(1, 11)]
    path = tmp_path / "generation.json"

    with pytest.raises(GenerationBaselineError, match=f"no scored case for {metric}"):
        update_generation_baseline(path, make_run({"hybrid": unscored}))

    assert not path.exists()


# --- Refusals: the file stays byte-identical ------------------------------------------------------


def timeouts(*indexes: int) -> list[GenerationCase]:
    return hybrid_cases(
        **{f"q{i:03d}": errored_case(i, generator_error("ProviderTimeout")) for i in indexes}
    )


def hybrid_with(**replacing: GenerationCase) -> GenerationRun:
    return make_run({"hybrid": hybrid_cases(**replacing), "no_rag": no_rag_cases()})


REFUSED: dict[str, tuple[Callable[[], GenerationRun], str]] = {
    "inconclusive": (
        lambda: make_run({"hybrid": timeouts(1, 2, 3)}),
        r"hybrid: inconclusive: 3 of 10 case\(s\) errored for a provider reason",
    ),
    "one-generator-timeout": (
        lambda: make_run({"hybrid": timeouts(4)}),
        r"hybrid: 1 of 10 case\(s\) errored for a provider reason",
    ),
    "one-generator-5xx": (
        lambda: hybrid_with(q004=errored_case(4, generator_error("ProviderUnavailable"))),
        r"hybrid: 1 of 10 case\(s\) errored for a provider reason",
    ),
    "skipped-after-a-daily-quota": (
        lambda: hybrid_with(
            q009=errored_case(
                9, generator_error("ProviderRateLimited", is_quota=True, skipped=True)
            )
        ),
        r"hybrid: 1 of 10 case\(s\) errored for a provider reason",
    ),
    "judge-timeout": (
        lambda: hybrid_with(
            q002=hybrid_case(2, correctness=failed("ProviderTimeout", provider_side=True))
        ),
        r"hybrid: 1 of 10 case\(s\) errored for a provider reason",
    ),
    "judge-quota": (
        lambda: hybrid_with(
            q003=hybrid_case(3, faithfulness=failed("ProviderRateLimited", provider_side=True))
        ),
        r"hybrid: 1 of 10 case\(s\) errored for a provider reason",
    ),
    "harness-bug": (
        lambda: hybrid_with(
            q005=hybrid_case(5, citation_precision=failed("MalformedInput", stage="assertion"))
        ),
        r"hybrid: 1 case\(s\) had an assertion that could not read its input",
    ),
    "judge-key-refused": (
        lambda: hybrid_with(
            q006=hybrid_case(6, correctness=failed("ProviderRequestRejected", provider_side=False))
        ),
        "the eval could not run",
    ),
    "no-judge-call": (
        lambda: make_run(info={"judge_provider": None, "judge_model": None}),
        "no judge call was made",
    ),
    "unknown-git-state": (
        lambda: make_run(info={"git_sha": None, "git_dirty": None}),
        "no git state",
    ),
    "no-config": (lambda: make_run({}), "no config to write"),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_a_run_that_cannot_be_a_baseline_is_refused_and_the_file_is_untouched(
    tmp_path: Path, name: str
) -> None:
    build, message = REFUSED[name]
    path, _ = write(tmp_path)
    before = path.read_bytes()

    with pytest.raises(GenerationBaselineError, match=message):
        update_generation_baseline(path, build())

    assert path.read_bytes() == before


@pytest.mark.parametrize("name", list(REFUSED))
def test_a_refused_run_creates_no_file(tmp_path: Path, name: str) -> None:
    build, message = REFUSED[name]
    path = tmp_path / "generation.json"

    with pytest.raises(GenerationBaselineError, match=message):
        update_generation_baseline(path, build())

    assert not path.exists()


def test_every_problem_is_listed_not_only_the_first(tmp_path: Path) -> None:
    run = make_run(
        {
            "hybrid": timeouts(4),
            "no_rag": [
                errored_case(1, generator_error("ProviderUnavailable")),
                *no_rag_cases()[1:],
            ],
        }
    )
    with pytest.raises(GenerationBaselineError) as caught:
        update_generation_baseline(tmp_path / "generation.json", run)

    assert "- hybrid: 1 of 10 case(s)" in str(caught.value)
    assert "- no_rag: 1 of 10 case(s)" in str(caught.value)


# --- A generator or judge bad output is a quality result, not a refusal ---------------------------


def test_a_generator_bad_output_is_written_as_a_scored_zero_for_schema_first_try(
    tmp_path: Path,
) -> None:
    bad = errored_case(4, generator_error("ProviderBadOutput"))
    path, _ = write(tmp_path, hybrid_with(q004=bad))

    hybrid = read_generation_baseline(path)["hybrid"]
    # schema first-try: nine 1s and the bad output's 0 -> 0.9 over all 10; the other metrics have
    # nine scored questions; the answered-case statistics leave the failed call out.
    assert (hybrid.metrics["schema_first_try"], hybrid.n["schema_first_try"]) == (0.9, 10)
    assert hybrid.n["correctness"] == 9
    assert (hybrid.cases, hybrid.n_cost) == (10, 9)


def test_a_judge_bad_output_leaves_one_metric_unscored_and_shows_in_n(tmp_path: Path) -> None:
    unjudged = hybrid_case(2, faithfulness=failed("ProviderBadOutput", provider_side=False))
    path, _ = write(tmp_path, hybrid_with(q002=unjudged))

    hybrid = read_generation_baseline(path)["hybrid"]
    assert hybrid.n["faithfulness"] == 7  # q002 is out, as are the two refusals
    assert hybrid.n["correctness"] == 10


# --- One setup per file ---------------------------------------------------------------------------

MISMATCHES = {
    "golden-set-bytes": ({"info": {"golden_set_sha256": OTHER_SHA}}, "golden_set_sha256"),
    "golden-set-version": ({"info": {"golden_set_version": "v2"}}, "golden_set_version"),
    "judge-model": ({"info": {"judge_model": "another-judge"}}, "judge_model"),
    "judge-provider": ({"info": {"judge_provider": "another"}}, "judge_provider"),
    "correctness-judge-prompt": (
        {"info": {"judge_prompt_versions": {"correctness": "judge_correctness_v2@aaaaaaaa"}}},
        "correctness judge prompt: row judge_correctness_v1@1bde5fe4",
    ),
    "generator-model": (
        {"identity": {"hybrid": {"model": "gemini-other"}}},
        "generator: row gemini/gemini-3.5-flash-lite, run gemini/gemini-other",
    ),
}


@pytest.mark.parametrize("name", list(MISMATCHES))
def test_a_run_from_another_setup_is_refused_next_to_the_kept_no_rag_row(
    tmp_path: Path, name: str
) -> None:
    options, message = MISMATCHES[name]
    path, _ = write(tmp_path)
    before = path.read_bytes()

    with pytest.raises(
        GenerationBaselineError, match=f"row 'no_rag' differs from the run in .*{message}"
    ):
        update_generation_baseline(path, make_run({"hybrid": hybrid_cases()}, **options))

    assert path.read_bytes() == before


def test_the_message_says_how_to_fix_a_mismatch(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    run = make_run({"hybrid": hybrid_cases()}, info={"golden_set_sha256": OTHER_SHA})
    with pytest.raises(GenerationBaselineError, match="write every config from one results file"):
        update_generation_baseline(path, run)


def test_a_joint_run_replaces_rows_from_another_setup(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    update_generation_baseline(path, make_run(info={"golden_set_sha256": OTHER_SHA}))
    assert {row.golden_set_sha256 for row in read_generation_baseline(path).values()} == {OTHER_SHA}


def test_the_index_is_compared_only_between_rows_that_use_one(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    before = path.read_bytes()
    other_index = {"hybrid_rerank": {"index_version": "0.141.1@deadbeef"}}

    with pytest.raises(GenerationBaselineError, match=r"row 'hybrid' differs .*index_version"):
        update_generation_baseline(
            path, make_run({"hybrid_rerank": hybrid_cases()}, identity=other_index)
        )
    assert path.read_bytes() == before

    # Same index: fine. A run of no_rag alone has no index to compare, and no_rag's "none" is not
    # an index: neither is a mismatch.
    update_generation_baseline(path, make_run({"hybrid_rerank": hybrid_cases()}))
    update_generation_baseline(path, make_run({"no_rag": no_rag_cases()}))
    assert set(on_disk(path)) == {"hybrid", "hybrid_rerank", "no_rag"}


@pytest.mark.parametrize("content", ["", "not json", "[]", '{"hybrid": {"metrics": {}}}'], ids=repr)
def test_a_baseline_file_that_does_not_validate_is_refused_and_left_alone(
    tmp_path: Path, content: str
) -> None:
    path = tmp_path / "generation.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(GenerationBaselineError, match="no longer validate"):
        update_generation_baseline(path, make_run({"no_rag": no_rag_cases()}))

    assert path.read_text(encoding="utf-8") == content


# --- A recorded promptfoo output, end to end ------------------------------------------------------


def test_a_recorded_run_becomes_baseline_rows(tmp_path: Path) -> None:
    run = parse_results(clean_sample(), git_sha=HEAD, git_dirty=False)
    path, update = write(tmp_path, run)

    rows = read_generation_baseline(path)
    assert update.written == ("no_rag", "hybrid")
    hybrid, no_rag = rows["hybrid"], rows["no_rag"]
    # Three questions, worked out from the recording: correctness 0.5, 1, 1 -> 5/6; faithfulness
    # 0.5, 0.5 and the refusal not applicable -> 0.5 (n 2); citation validity 1, 0, 1 -> 2/3.
    assert hybrid.metrics["correctness"] == pytest.approx(5 / 6)
    assert hybrid.metrics["faithfulness"] == 0.5
    assert (hybrid.n["correctness"], hybrid.n["faithfulness"], hybrid.n["citation_precision"]) == (
        3,
        2,
        2,
    )
    assert hybrid.metrics["citation_validity"] == pytest.approx(2 / 3)
    assert (hybrid.cases, hybrid.index_version) == (3, INDEX)
    assert hybrid.prompt_version == "answer_v1@08cc49e5"
    assert hybrid.thresholds == TECH_15_5
    # no_rag: correctness 0, 0, 0; refusal accuracy 1, 1, 0; no faithfulness, no citation metrics.
    assert no_rag.metrics == pytest.approx(
        {"correctness": 0.0, "refusal_correctness": 2 / 3, "schema_first_try": 1.0}
    )
    assert no_rag.index_version == "none"
    assert (no_rag.judge_provider, no_rag.judge_model) == ("fake-judge", "scripted-judge")
    assert no_rag.date == datetime(2026, 10, 8, 11, 33, 20, 796000, tzinfo=UTC)
    # Latency leaves the cold q003 out: hybrid q007 and q045 took 15 ms each, so p50 = p95 = 15.
    assert (hybrid.latency_p50_ms, hybrid.latency_p95_ms, hybrid.n_latency) == (15, 15, 2)
    # Cost is the mean over the three answers: (0.00116985 + 0.0011679 + 0.0011643) / 3 * 1000.
    assert hybrid.cost_per_1k_usd == pytest.approx(1.16735, rel=1e-5)


def test_the_whole_recording_is_refused_because_it_has_provider_errors(tmp_path: Path) -> None:
    run = parse_results(SAMPLE.read_bytes(), git_sha=HEAD, git_dirty=False)
    with pytest.raises(GenerationBaselineError, match="not fully scored"):
        update_generation_baseline(tmp_path / "generation.json", run)


# --- The gate on the new baseline against itself (4.09's check) -----------------------------------


def test_the_gate_on_a_new_baseline_against_the_run_it_came_from_passes(tmp_path: Path) -> None:
    run = make_run()
    path, update = write(tmp_path, run)

    for report in (
        check_against_itself(run, update),
        evaluate_generation_gate(run, read_generation_baseline(path)),
    ):
        assert report is not None
        assert report.status == "pass"
        assert report.reasons == []
        gated = [row for row in report.rows if row.config == "hybrid" and row.passed is not None]
        assert {row.metric for row in gated} == set(TECH_15_5)
        assert all(row.passed and row.delta == 0 for row in gated)


def test_the_baseline_does_gate_a_worse_run(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    worse = make_run({"hybrid": [hybrid_case(i, correctness=0.5) for i in range(1, 11)]})

    report = evaluate_generation_gate(worse, read_generation_baseline(path))

    assert report.status == "fail"  # correctness 0.5 < 0.85 - 0.08
    assert [row.metric for row in report.rows if row.passed is False] == ["correctness"]


def test_the_check_is_only_about_gated_rows_that_were_written(tmp_path: Path) -> None:
    run = make_run({"no_rag": no_rag_cases()})
    _, update = write(tmp_path, run)
    assert check_against_itself(run, update) is None  # no_rag has no thresholds


def test_a_seeded_floor_the_run_itself_misses_is_visible_in_the_check(tmp_path: Path) -> None:
    run = parse_results(clean_sample(), git_sha=HEAD, git_dirty=False)  # faithfulness 0.5
    _, update = write(tmp_path, run)

    report = check_against_itself(run, update)

    assert report is not None
    assert report.status == "fail"
    assert [row.metric for row in report.rows if row.passed is False] == ["faithfulness"]


# --- The row schema -------------------------------------------------------------------------------


def test_latency_and_cost_come_with_their_n_or_not_at_all(tmp_path: Path) -> None:
    path, _ = write(tmp_path)
    row = on_disk(path)["hybrid"]
    GenerationBaselineEntry.model_validate(row)  # as written
    without = {
        k: v
        for k, v in row.items()
        if k not in {"latency_p50_ms", "latency_p95_ms", "n_latency", "cost_per_1k_usd", "n_cost"}
    }
    GenerationBaselineEntry.model_validate(without)  # a row written before the fields existed

    for broken in (
        row | {"latency_p95_ms": None},
        row | {"n_latency": 0},
        without | {"n_latency": 3},
        row | {"cost_per_1k_usd": None},
        without | {"n_cost": 4},
    ):
        with pytest.raises(ValidationError):
            GenerationBaselineEntry.model_validate(broken)


# --- The command ----------------------------------------------------------------------------------


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str | None, bool | None]]:
    """``repo_state`` of the CLI, scripted: a clean checkout at ``HEAD``. Append to override."""
    states: list[tuple[str | None, bool | None]] = [(HEAD, False)]
    monkeypatch.setattr(grounded.cli, "repo_state", lambda: states[-1])
    return states


def baseline_command(results: Path, baseline: Path | None, *extra: str) -> Any:
    args = ["eval", "baseline", "--suite", "generation", "--results", str(results)]
    if baseline is not None:
        args += ["--baseline", str(baseline)]
    return runner.invoke(app, [*args, *extra])


@pytest.fixture
def sample(tmp_path: Path) -> Path:
    path = tmp_path / "promptfoo.json"
    path.write_bytes(clean_sample())
    return path


def test_the_command_writes_the_rows_prints_them_and_checks_the_gate(
    tmp_path: Path, sample: Path, git: list[Any]
) -> None:
    target = tmp_path / "out" / "generation.json"
    result = baseline_command(sample, target)

    assert result.exit_code == 0, result.output
    assert f"Baseline updated: {target} (no_rag, hybrid)" in result.output
    assert "hybrid: 3 cases;" in result.output
    assert "correctness 0.833 (n=3)" in result.output
    assert "faithfulness 0.500 (n=2)" in result.output
    assert "hybrid: thresholds are the initial table of Tech.md §15.5" in result.output
    # The sample's faithfulness (0.5) is under the 0.85 floor: the check says so.
    assert "### Generation gate: ❌ fail" in result.output
    assert "does not pass its own gate" in result.output
    assert set(on_disk(target)) == {"hybrid", "no_rag"}


def test_the_command_is_quiet_about_the_gate_when_the_new_baseline_passes_it(
    tmp_path: Path, git: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grounded.cli, "read_generation_results", lambda path, **_: make_run())
    target = tmp_path / "generation.json"

    result = baseline_command(tmp_path / "ignored.json", target)

    assert result.exit_code == 0, result.output
    assert "### Generation gate: ✅ pass" in result.output
    assert "does not pass" not in result.output


def test_the_default_target_is_the_committed_baseline_path(
    tmp_path: Path, sample: Path, git: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    default = tmp_path / "default.json"
    monkeypatch.setattr(grounded.cli, "GENERATION_BASELINE", default)
    # Typer read the default when the option was declared: the command falls back to the module's
    # name at call time, so the patched path is the one written.
    assert baseline_command(sample, None).exit_code == 0
    assert default.exists()


def test_the_command_refuses_a_run_with_errors_and_leaves_the_file_alone(
    tmp_path: Path, git: list[Any]
) -> None:
    whole = tmp_path / "whole.json"
    whole.write_bytes(SAMPLE.read_bytes())
    target = tmp_path / "generation.json"

    result = baseline_command(whole, target)

    assert result.exit_code == 1
    assert "Baseline not updated:" in result.output
    assert "not fully scored" in result.output
    assert "The baseline file was not changed." in result.output
    assert not target.exists()


def test_the_command_refuses_an_unknown_git_state(
    tmp_path: Path, sample: Path, git: list[Any]
) -> None:
    git.append((None, None))
    target = tmp_path / "generation.json"
    result = baseline_command(sample, target)
    assert result.exit_code == 1
    assert "no git state" in result.output
    assert not target.exists()


def test_the_command_warns_about_uncommitted_changes_and_records_them(
    tmp_path: Path, sample: Path, git: list[Any]
) -> None:
    git.append((HEAD, True))
    target = tmp_path / "generation.json"
    result = baseline_command(sample, target)
    assert result.exit_code == 0
    assert "Warning: uncommitted changes" in result.output
    assert on_disk(target)["hybrid"]["git_dirty"] is True


def test_the_command_refuses_a_row_from_another_setup(
    tmp_path: Path, sample: Path, git: list[Any]
) -> None:
    target, _ = write(tmp_path, make_run(info={"golden_set_sha256": OTHER_SHA}))
    before = target.read_bytes()
    # The recording has both configs, so both are replaced and nothing is kept: it works. A run of
    # one config over a kept row of another golden set is what is refused.
    only_hybrid = json.loads(sample.read_bytes())
    only_hybrid["results"]["results"] = [
        r for r in only_hybrid["results"]["results"] if r["provider"]["label"] == "hybrid"
    ]
    sample.write_text(json.dumps(only_hybrid), encoding="utf-8")

    result = baseline_command(sample, target)

    assert result.exit_code == 1
    assert "row 'no_rag' differs from the run in golden_set_sha256" in result.output
    assert target.read_bytes() == before


@pytest.mark.parametrize("text", ["", "not json", '{"results": {}}'])
def test_the_command_reports_a_results_file_it_cannot_read(
    tmp_path: Path, git: list[Any], text: str
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(text, encoding="utf-8")
    result = baseline_command(bad, tmp_path / "generation.json")
    assert result.exit_code == 1
    assert "Cannot read the results file" in result.output


def test_the_command_reports_a_missing_results_file(tmp_path: Path, git: list[Any]) -> None:
    result = baseline_command(tmp_path / "nope.json", tmp_path / "generation.json")
    assert result.exit_code == 1
    assert "Cannot read the results file" in result.output


def test_the_command_reports_a_baseline_path_it_cannot_use(
    tmp_path: Path, sample: Path, git: list[Any]
) -> None:
    result = baseline_command(sample, tmp_path)  # a directory
    assert result.exit_code == 1
    assert "Cannot read or write the baseline" in result.output


def test_retrieval_baselines_are_not_written_by_this_command(tmp_path: Path, sample: Path) -> None:
    result = runner.invoke(
        app, ["eval", "baseline", "--suite", "retrieval", "--results", str(sample)]
    )
    assert result.exit_code == 2  # a usage error: the only suite is generation


def test_help_names_the_options(tmp_path: Path) -> None:
    result = runner.invoke(app, ["eval", "baseline", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for word in ("--suite", "--results", "--baseline", "generation", "retrieval"):
        assert word in plain
