"""The baselines and the gates with ``hybrid_rerank`` rows in them (6.04, PRD D55).

The committed files are never written: every test reads ``eval/baselines/*.json`` and adds a
synthetic ``hybrid_rerank`` row, without thresholds, to a copy in ``tmp_path``. What is pinned is
that the two gates (Author-owned, untouched) report that row or skip it and keep their verdict, that
a rewrite of a baseline adds a row without touching the others, and that a reranked row cannot be
made of degraded cases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.gate import evaluate_gate, evaluate_generation_gate
from grounded.evals.generation_baseline import (
    GenerationBaselineError,
    update_generation_baseline,
)
from grounded.evals.generation_results import (
    GENERATION_BASELINE,
    parse_results,
    read_generation_baseline,
)
from grounded.evals.report import render_report
from grounded.evals.retrieval_runner import (
    RETRIEVAL_BASELINE,
    metric_names,
    read_baseline,
    update_baseline,
    write_json,
    write_run,
)
from grounded.retrieval.config import RetrievalConfig
from grounded.schemas.eval import (
    RerankRunStats,
    RetrievalBaselineEntry,
    RetrievalConfigResult,
    RetrievalRun,
    RetrievalRunInfo,
)
from grounded.schemas.generation_eval import (
    GenerationBaselineEntry,
    GenerationCase,
    GenerationConfigResult,
    GenerationRun,
    GenerationRunInfo,
    MetricResult,
)

NOW = datetime(2026, 10, 13, 9, 0, 0, tzinfo=UTC)
MODEL = "rerank-v4.0-pro"
runner = CliRunner()


# --- The committed files ------------------------------------------------------------------------


def test_the_committed_baselines_still_load() -> None:
    # `rerank` and the other 6.04 fields are optional: no committed row has them.
    assert set(read_baseline(RETRIEVAL_BASELINE)) >= {"dense", "fts", "hybrid"}
    assert set(read_generation_baseline(GENERATION_BASELINE)) >= {"hybrid", "no_rag"}


def retrieval_run(
    baseline: dict[str, RetrievalBaselineEntry], *extra: str, drop: float = 0.0
) -> RetrievalRun:
    """A run on exactly the setup of the committed rows, scoring what they score; the configs named
    in ``extra`` (``hybrid_rerank``) score ``drop`` less."""
    hybrid = baseline["hybrid"]
    info = RetrievalRunInfo(
        date=NOW,
        git_sha=hybrid.git_sha,
        git_dirty=False,
        golden_set_version=hybrid.golden_set_version,
        golden_set_sha256=hybrid.golden_set_sha256,
        index_version_id=1,
        index_config_hash=hybrid.index_config_hash,
        fastapi_ref=hybrid.fastapi_ref,
        fastapi_sha=hybrid.fastapi_sha,
        embedding_model=hybrid.embedding_model,
        embedding_dim=hybrid.embedding_dim,
    )
    results: dict[str, RetrievalConfigResult] = {}
    for name in ("dense", "fts", "hybrid", *extra):
        row = baseline["hybrid" if name == "hybrid_rerank" else name]
        fields: dict[str, Any] = {
            "mode": name,
            "k_dense": 20,
            "k_fts": 20,
            "k_fused": 40,
            "k_context": 5,
            "rrf_k": 60,
        }
        if name == "hybrid_rerank":
            fields |= {"rerank_provider": "cohere", "rerank_model": MODEL}
        config = RetrievalConfig.model_validate(fields)
        results[name] = RetrievalConfigResult(
            config=name,
            retrieval_config=config,
            retrieval_config_hash=config.config_hash,
            k=row.k,
            n=row.n,
            skipped_unanswerable=5,
            metrics={m: v - (drop if name in extra else 0.0) for m, v in row.metrics.items()},
            questions=[],
            rerank=RerankRunStats(model=MODEL, live_calls=25, cache_hits=0)
            if name == "hybrid_rerank"
            else None,
        )
    return RetrievalRun(info=info, configs=results)


def with_rerank_row(
    baseline: dict[str, RetrievalBaselineEntry],
) -> dict[str, RetrievalBaselineEntry]:
    row = baseline["hybrid"]
    return {
        **baseline,
        "hybrid_rerank": row.model_copy(
            update={"thresholds": {}, "metrics": {m: v + 0.1 for m, v in row.metrics.items()}}
        ),
    }


def test_the_retrieval_gate_skips_an_ungated_rerank_row_the_run_lacks(tmp_path: Path) -> None:
    real = read_baseline(RETRIEVAL_BASELINE)
    extended_file = tmp_path / "retrieval.json"
    write_json(extended_file, _dump(with_rerank_row(real)))
    extended = read_baseline(extended_file)  # the copy loads
    run = retrieval_run(real)

    before, after = evaluate_gate(run, real), evaluate_gate(run, extended)

    assert before.status == "pass"
    assert after == before  # same rows, same reasons, same verdict: the extra row is invisible


def test_the_retrieval_gate_only_reports_a_rerank_result_next_to_the_gated_hybrid() -> None:
    real = read_baseline(RETRIEVAL_BASELINE)
    run = retrieval_run(real, "hybrid_rerank", drop=0.5)  # a reranker that is much worse

    for baseline in (real, with_rerank_row(real)):
        report = evaluate_gate(run, baseline)

        assert report.status == "pass"  # an ungated config never changes the verdict
        assert report.reasons == []
        rerank_rows = [row for row in report.rows if row.config == "hybrid_rerank"]
        assert {row.metric for row in rerank_rows} == set(metric_names())
        assert all(row.passed is None and row.threshold is None for row in rerank_rows)


def test_the_retrieval_gate_still_fails_a_regression_of_hybrid_with_the_extra_row() -> None:
    real = read_baseline(RETRIEVAL_BASELINE)
    run = retrieval_run(real, "hybrid", drop=0.3)  # the gated config falls

    assert evaluate_gate(run, with_rerank_row(real)).status == "fail"


def test_the_retrieval_gate_command_runs_on_the_extended_file(tmp_path: Path) -> None:
    real = read_baseline(RETRIEVAL_BASELINE)
    baseline = tmp_path / "retrieval.json"
    results = tmp_path / "results.json"
    write_json(baseline, _dump(with_rerank_row(real)))
    write_run(results, retrieval_run(real))

    done = runner.invoke(
        app,
        [
            "eval",
            "gate",
            "--suite",
            "retrieval",
            "--results",
            str(results),
            "--baseline",
            str(baseline),
        ],
    )

    assert done.exit_code == 0, done.output
    assert "hybrid_rerank" not in done.output


def _dump(rows: dict[str, RetrievalBaselineEntry]) -> dict[str, Any]:
    from pydantic import TypeAdapter

    data: dict[str, Any] = TypeAdapter(dict[str, RetrievalBaselineEntry]).dump_python(
        rows, mode="json"
    )
    for row in data.values():
        if not row["thresholds"]:
            del row["thresholds"]
    return data


def test_an_old_results_file_without_the_rerank_key_still_validates() -> None:
    run = retrieval_run(read_baseline(RETRIEVAL_BASELINE), "hybrid_rerank")
    dumped = run.model_dump(mode="json")
    for config in dumped["configs"].values():
        del config["rerank"]

    assert all(c.rerank is None for c in RetrievalRun.model_validate(dumped).configs.values())


def test_adding_the_rerank_row_rewrites_no_other_byte_of_the_committed_file(
    tmp_path: Path,
) -> None:
    committed = RETRIEVAL_BASELINE.read_text(encoding="utf-8")
    copy = tmp_path / "retrieval.json"
    copy.write_text(committed, encoding="utf-8", newline="\n")
    real = read_baseline(RETRIEVAL_BASELINE)
    # A rerank-only run on the committed setup, as `--config hybrid_rerank --write-baseline` is.
    run = retrieval_run(real, "hybrid_rerank")
    run = run.model_copy(update={"configs": {"hybrid_rerank": run.configs["hybrid_rerank"]}})

    update_baseline(copy, run)

    rewritten = copy.read_text(encoding="utf-8")
    assert rewritten.startswith(committed[: committed.rindex("\n}")] + ",\n")
    added = json.loads(rewritten)["hybrid_rerank"]
    assert "thresholds" not in added  # ungated (D55)
    assert added["k"] == 40
    assert added["retrieval_config_hash"] == run.configs["hybrid_rerank"].retrieval_config_hash


# --- The generation baseline and its gate -------------------------------------------------------

FAITHFULNESS = MetricResult(state="scored", value=1.0, claims=3, supported=3)
SCORED = {
    "schema_first_try": MetricResult(state="scored", value=1.0),
    "citation_validity": MetricResult(state="scored", value=1.0),
    "refusal_correctness": MetricResult(state="scored", value=1.0),
    "citation_precision": MetricResult(state="scored", value=1.0),
    "faithfulness": FAITHFULNESS,
    "correctness": MetricResult(state="scored", value=1.0),
}


def case(index: int, **fields: Any) -> GenerationCase:
    return GenerationCase(
        id=f"q{index:03d}",
        type="factual",
        metrics=SCORED,
        latency_ms=100.0 + index,
        cost_usd=0.001,
        **fields,
    )


def generation_run(
    baseline: dict[str, GenerationBaselineEntry], configs: dict[str, list[GenerationCase]]
) -> GenerationRun:
    hybrid = baseline["hybrid"]
    info = GenerationRunInfo(
        date=NOW,
        promptfoo_version=hybrid.promptfoo_version,
        git_sha="c" * 40,
        git_dirty=False,
        golden_set_version=hybrid.golden_set_version,
        golden_set_sha256=hybrid.golden_set_sha256,
        judge_provider=hybrid.judge_provider,
        judge_model=hybrid.judge_model,
        judge_prompt_versions=hybrid.judge_prompt_versions,
    )
    return GenerationRun(
        info=info,
        configs={
            name: GenerationConfigResult(
                config=name,
                provider=hybrid.provider,
                model=hybrid.model,
                prompt_version=hybrid.prompt_version,
                index_version=hybrid.index_version,
                retrieval_config_hash=None,
                cases=cases,
            )
            for name, cases in configs.items()
        },
    )


def rerank_rows(baseline: dict[str, GenerationBaselineEntry]) -> dict[str, GenerationBaselineEntry]:
    return {**baseline, "hybrid_rerank": baseline["hybrid"].model_copy(update={"thresholds": {}})}


def reranked(count: int = 10) -> list[GenerationCase]:
    return [case(i, rerank_used=True) for i in range(1, count + 1)]


def test_the_generation_gate_skips_an_ungated_rerank_row_the_run_lacks() -> None:
    real = read_generation_baseline(GENERATION_BASELINE)
    run = generation_run(real, {"hybrid": [case(i) for i in range(1, 11)]})

    before = evaluate_generation_gate(run, real)
    after = evaluate_generation_gate(run, rerank_rows(real))

    assert before.status == "pass"
    assert after == before


def test_the_generation_gate_only_reports_a_rerank_config_next_to_the_gated_hybrid() -> None:
    real = read_generation_baseline(GENERATION_BASELINE)
    worse = [case(i, rerank_used=True) for i in range(1, 11)]
    worse[0] = worse[0].model_copy(
        update={"metrics": {**SCORED, "correctness": MetricResult(state="scored", value=0.0)}}
    )
    run = generation_run(real, {"hybrid": [case(i) for i in range(1, 11)], "hybrid_rerank": worse})

    for baseline in (real, rerank_rows(real)):
        report = evaluate_generation_gate(run, baseline)

        assert report.status == "pass"
        assert report.reasons == []
        assert {s.config for s in report.configs} == {"hybrid", "hybrid_rerank"}
        rows = [row for row in report.rows if row.config == "hybrid_rerank"]
        assert rows
        assert all(row.passed is None and row.threshold is None for row in rows)


# --- A reranked row is made of reranked cases ---------------------------------------------------


def test_a_clean_rerank_run_becomes_an_ungated_row_and_keeps_the_others_byte_for_byte(
    tmp_path: Path,
) -> None:
    real = read_generation_baseline(GENERATION_BASELINE)
    copy = tmp_path / "generation.json"
    copy.write_text(GENERATION_BASELINE.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    committed = json.loads(copy.read_text(encoding="utf-8"))

    update = update_generation_baseline(copy, generation_run(real, {"hybrid_rerank": reranked()}))

    assert update.written == ("hybrid_rerank",)
    assert update.seeded == ()  # only hybrid is ever seeded with the Tech §15.5 table
    after = json.loads(copy.read_text(encoding="utf-8"))
    assert after["hybrid_rerank"]["thresholds"] == {}
    assert {name: after[name] for name in committed} == committed  # every existing row, unchanged
    assert list(after) == ["hybrid", "hybrid_rerank", "no_rag"]


@pytest.mark.parametrize(
    ("fields", "named"),
    [
        ({"rerank_used": False, "rerank_error": "timeout"}, "timeout"),
        ({"rerank_used": False, "rerank_error": "rate_limited"}, "rate_limited"),
        ({"rerank_used": False}, "reranker not used"),
        ({"rerank_used": True, "rerank_error": "cap_reached"}, "cap_reached"),
    ],
)
def test_a_degraded_case_makes_the_whole_rerank_row_unwritable(
    tmp_path: Path, fields: dict[str, Any], named: str
) -> None:
    real = read_generation_baseline(GENERATION_BASELINE)
    copy = tmp_path / "generation.json"
    copy.write_text(GENERATION_BASELINE.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    before = copy.read_bytes()
    cases = reranked()
    cases[2] = case(3, **fields)

    with pytest.raises(GenerationBaselineError) as refused:
        update_generation_baseline(copy, generation_run(real, {"hybrid_rerank": cases}))

    message = str(refused.value)
    assert "a degraded case is not a result" in message
    assert f"hybrid_rerank: the reranker degraded in 1 of 10 case(s) ({named}; q003)" in message
    assert "answered from the RRF order" in message
    assert copy.read_bytes() == before  # nothing was written


def test_the_ids_of_many_degraded_cases_are_capped() -> None:
    real = read_generation_baseline(GENERATION_BASELINE)
    cases = [case(i, rerank_used=False, rerank_error="timeout") for i in range(1, 11)]

    with pytest.raises(GenerationBaselineError, match=r"q001, q002, q003, q004, q005 and 5 more"):
        update_generation_baseline(
            Path("unused.json"), generation_run(real, {"hybrid_rerank": cases})
        )


def test_a_hybrid_row_cannot_be_made_of_reranked_cases(tmp_path: Path) -> None:
    real = read_generation_baseline(GENERATION_BASELINE)
    cases = [case(i) for i in range(1, 11)]
    cases[4] = case(5, rerank_used=True)  # RERANK_PROVIDER=cohere was left in .env
    copy = tmp_path / "generation.json"

    with pytest.raises(GenerationBaselineError, match=r"hybrid: 1 case\(s\) were answered with a"):
        update_generation_baseline(copy, generation_run(real, {"hybrid": cases}))

    assert not copy.exists()


def test_no_rag_is_not_a_reranked_config_and_is_written_as_before(tmp_path: Path) -> None:
    real = read_generation_baseline(GENERATION_BASELINE)
    copy = tmp_path / "generation.json"
    no_rag = [case(i, rerank_used=False) for i in range(1, 11)]

    update = update_generation_baseline(copy, generation_run(real, {"no_rag": no_rag}))

    assert update.written == ("no_rag",)


def rerank_results(tmp_path: Path, *, degraded: str | None) -> Path:
    """The 4.06 recording cut to the three questions without an error (q003, q007, q045), its
    ``hybrid`` provider renamed ``hybrid_rerank`` with the trace facts of 6.03 in every row's
    metadata. ``degraded`` makes the first of them a case that stayed on the RRF order."""
    sample = Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo"
    document = json.loads((sample / "results_sample_judge.json").read_bytes())
    rows = [
        row
        for row in document["results"]["results"]
        if row["metadata"]["golden"]["id"] in {"q003", "q007", "q045"}
        and row["provider"]["label"] == "hybrid"
    ]
    for row in rows:
        row["provider"]["label"] = "hybrid_rerank"
        row["metadata"].update(rerank_used=True, rerank_error=None)
    if degraded is not None:
        rows[0]["metadata"].update(rerank_used=False, rerank_error=degraded)
    document["results"]["results"] = rows
    path = tmp_path / "promptfoo.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grounded.cli, "repo_state", lambda: ("c" * 40, False))


@pytest.mark.usefixtures("git")
def test_the_baseline_command_refuses_a_degraded_rerank_run(tmp_path: Path) -> None:
    results, baseline = rerank_results(tmp_path, degraded="rate_limited"), tmp_path / "g.json"

    done = runner.invoke(
        app,
        [
            "eval",
            "baseline",
            "--suite",
            "generation",
            "--results",
            str(results),
            "--baseline",
            str(baseline),
        ],
    )

    assert done.exit_code == 1
    assert "Baseline not updated:" in done.output
    assert (
        "hybrid_rerank: the reranker degraded in 1 of 3 case(s) (rate_limited; q003)" in done.output
    )
    assert "The baseline file was not changed." in done.output
    assert not baseline.exists()


@pytest.mark.usefixtures("git")
def test_the_baseline_command_writes_an_undegraded_rerank_run_without_thresholds(
    tmp_path: Path,
) -> None:
    results, baseline = rerank_results(tmp_path, degraded=None), tmp_path / "g.json"

    done = runner.invoke(
        app,
        [
            "eval",
            "baseline",
            "--suite",
            "generation",
            "--results",
            str(results),
            "--baseline",
            str(baseline),
        ],
    )

    assert done.exit_code == 0, done.output
    assert f"Baseline updated: {baseline} (hybrid_rerank)" in done.output
    assert "thresholds are the initial table" not in done.output  # only hybrid is seeded
    assert json.loads(baseline.read_bytes())["hybrid_rerank"]["thresholds"] == {}


# --- What the results file says -----------------------------------------------------------------


def test_the_parser_carries_rerank_used_and_error_into_every_case() -> None:
    sample = Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo"
    document = json.loads((sample / "results_sample_judge.json").read_bytes())
    rows = document["results"]["results"]
    hybrid_rows = [r for r in rows if r["provider"]["label"] == "hybrid"]
    for row in hybrid_rows:
        row["metadata"].update(rerank_used=True, rerank_error=None)
    hybrid_rows[0]["metadata"].update(rerank_used=False, rerank_error="timeout")

    run = parse_results(json.dumps(document))

    cases = run.configs["hybrid"].cases
    assert [(c.rerank_used, c.rerank_error) for c in cases].count((False, "timeout")) == 1
    assert {c.rerank_used for c in cases} == {True, False}
    # A file from before 6.03 has neither key: unknown, not "false".
    older = parse_results((sample / "results_sample_judge.json").read_bytes())
    assert {(c.rerank_used, c.rerank_error) for c in older.configs["hybrid"].cases} == {
        (None, None)
    }


# --- The report ---------------------------------------------------------------------------------


def test_the_report_puts_hybrid_rerank_right_after_hybrid_in_both_tables() -> None:
    retrieval = with_rerank_row(read_baseline(RETRIEVAL_BASELINE))
    generation = rerank_rows(read_generation_baseline(GENERATION_BASELINE))

    report = render_report(retrieval, generation)

    lines = report.splitlines()
    retrieval_names = [ln.split("|")[1].strip() for ln in lines if ln.startswith("| ")][1:6]
    assert retrieval_names == [
        "dense",
        "fts",
        "hybrid",
        "hybrid_rerank",
        "*PRD §8 target (hybrid)*",
    ]
    [header] = [ln for ln in lines if ln.startswith("| Metric |")]
    assert [c.strip() for c in header.split("|")[2:5]] == ["no_rag", "hybrid", "hybrid_rerank"]


def test_without_a_rerank_row_the_report_is_what_it_was() -> None:
    retrieval = read_baseline(RETRIEVAL_BASELINE)
    generation = read_generation_baseline(GENERATION_BASELINE)

    assert "hybrid_rerank" not in render_report(retrieval, generation)
    # The existing rows print the same with the extra one present (it only adds a row).
    plain = render_report(retrieval, None).splitlines()
    extended = render_report(with_rerank_row(retrieval), None).splitlines()
    for name in ("dense", "fts", "hybrid"):
        [row] = [ln for ln in plain if ln.startswith(f"| {name} |")]
        assert row in extended
