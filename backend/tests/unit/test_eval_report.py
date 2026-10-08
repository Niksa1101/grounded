"""``grounded eval report`` (Tech.md §15.7, ticket 4.09a): the README's eval tables, rendered from
the committed baselines only.

The output is pinned on small hand-built baselines: every number below is chosen so that the
verdicts sit exactly on a PRD §8 target or just beside it, and the expected text is written out by
hand. The one test that reads the real files checks that the README is not stale.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from grounded.cli import app
from grounded.evals.generation_results import GENERATION_BASELINE
from grounded.evals.report import (
    GENERATION_TARGETS,
    RAG_VALUE_TARGET,
    RETRIEVAL_TARGETS,
    Target,
    build_report,
    render_report,
)
from grounded.evals.retrieval_runner import RETRIEVAL_BASELINE
from grounded.schemas.eval import RetrievalBaselineEntry
from grounded.schemas.generation_eval import GenerationBaselineEntry

runner = CliRunner()
README = Path(__file__).resolve().parents[3] / "README.md"
SHA = "8f0848177a5cb5a8ef649ee211ab9afb7f9bc07c73d37b37cbf3ef554e1abcb2"
INDEX_HASH = "4949e8a3207c4294f1f39a763eba8c286fc4fc79499420b841321a66c4cf7e5a"

# --- Hand-built baselines -------------------------------------------------------------------------


def retrieval_row(metrics: dict[str, float], *, k: int = 20, **overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "metrics": metrics,
        "n": 25,
        "k": k,
        "golden_set_version": "v1",
        "golden_set_sha256": SHA,
        "index_config_hash": INDEX_HASH,
        "fastapi_ref": "0.141.1",
        "fastapi_sha": "95f8322ee1dcda7ceace7b1c4f6c9915b36d748f",
        "embedding_model": "gemini-embedding-001",
        "embedding_dim": 768,
        "git_sha": "18fd3996c5c9b23b0af50257f64a61c20cab83e3",
        "git_dirty": False,
        "date": datetime(2026, 10, 1, 11, 23, 53, tzinfo=UTC),
    }
    return RetrievalBaselineEntry.model_validate(fields | overrides)


def retrieval_rows() -> dict[str, RetrievalBaselineEntry]:
    """Hybrid sits exactly on Recall@5 0.80 and nDCG@5 0.65 (met: a target is "at least") and just
    under the MRR target of 0.60."""
    return {
        "dense": retrieval_row(
            {"recall@5": 0.76, "recall@10": 0.92, "mrr": 0.7, "ndcg@5": 0.7, "ndcg@10": 0.75}
        ),
        "fts": retrieval_row(
            {"recall@5": 0.58, "recall@10": 0.7, "mrr": 0.4, "ndcg@5": 0.45, "ndcg@10": 0.5}
        ),
        "hybrid": retrieval_row(
            {"recall@5": 0.8, "recall@10": 0.84, "mrr": 0.59, "ndcg@5": 0.65, "ndcg@10": 0.7},
            k=40,
        ),
    }


def generation_row(
    metrics: dict[str, float], n: dict[str, int] | int, **overrides: Any
) -> GenerationBaselineEntry:
    counts = dict.fromkeys(metrics, n) if isinstance(n, int) else n
    fields: dict[str, Any] = {
        "metrics": metrics,
        "n": counts,
        "cases": 30,
        "golden_set_version": "v1",
        "golden_set_sha256": SHA,
        "provider": "gemini",
        "model": "gemini-3.5-flash-lite",
        "prompt_version": "answer_v1@08cc49e5",
        "index_version": "0.141.1@4949e8a3",
        "judge_provider": "groq",
        "judge_model": "openai/gpt-oss-120b",
        "judge_prompt_versions": {
            "faithfulness": "judge_faithfulness_v1@84103412",
            "correctness": "judge_correctness_v1@1bde5fe4",
        },
        "promptfoo_version": "0.123.1",
        "git_sha": "c" * 40,
        "git_dirty": False,
        "date": datetime(2026, 10, 9, 8, 0, 0, tzinfo=UTC),
    }
    return GenerationBaselineEntry.model_validate(fields | overrides)


def generation_rows() -> dict[str, GenerationBaselineEntry]:
    """Hybrid is exactly on three targets (faithfulness 0.90, refusal 0.90, schema 0.97) and under
    one (correctness 0.74 < 0.75). The RAG value is 0.74 - 0.60 = 0.14."""
    return {
        "no_rag": generation_row(
            {"correctness": 0.6, "refusal_correctness": 0.87, "schema_first_try": 1.0},
            30,
            prompt_version="answer_no_rag_v1@5f725a9d",
            index_version="none",
            latency_p50_ms=400.0,
            latency_p95_ms=900.0,
            n_latency=30,
            cost_per_1k_usd=0.1,
            n_cost=30,
        ),
        "hybrid": generation_row(
            {
                "faithfulness": 0.9,
                "correctness": 0.74,
                "refusal_correctness": 0.9,
                "schema_first_try": 0.97,
                "citation_validity": 1.0,
                "citation_precision": 0.8,
            },
            {
                "faithfulness": 24,
                "correctness": 30,
                "refusal_correctness": 30,
                "schema_first_try": 30,
                "citation_validity": 30,
                "citation_precision": 22,
            },
            latency_p50_ms=812.4,
            latency_p95_ms=1533.6,
            n_latency=27,
            cost_per_1k_usd=0.421,
            n_cost=30,
        ),
    }


RETRIEVAL_TEXT = "\n".join(
    [
        "**Retrieval ablation** (`eval/baselines/retrieval.json`; retrieval only, no LLM): golden "
        "set v1 (sha256 8f084817), index 0.141.1@4949e8a3, embedding gemini-embedding-001 "
        "(768 dims), git 18fd399, 2026-10-01.",
        "",
        "| Config | n | k | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        "| dense | 25 | 20 | 0.760 | 0.920 | 0.700 | 0.700 | 0.750 |",
        "| fts | 25 | 20 | 0.580 | 0.700 | 0.400 | 0.450 | 0.500 |",
        "| hybrid | 25 | 40 | 0.800 | 0.840 | 0.590 | 0.650 | 0.700 |",
        "| *PRD §8 target (hybrid)* |  |  | ≥ 0.80 | — | ≥ 0.60 | ≥ 0.65 | — |",
        "| *hybrid vs target* |  |  | met | — | not met | met | — |",
    ]
)
GENERATION_TEXT = "\n".join(
    [
        "**Generation baseline** (`eval/baselines/generation.json`): golden set v1 (sha256 "
        "8f084817), index 0.141.1@4949e8a3, generator gemini / gemini-3.5-flash-lite (prompts: "
        "no_rag answer_no_rag_v1@5f725a9d, hybrid answer_v1@08cc49e5), judge groq / "
        "openai/gpt-oss-120b (judge_correctness_v1@1bde5fe4, judge_faithfulness_v1@84103412), "
        "promptfoo 0.123.1, git ccccccc, 2026-10-09.",
        "",
        "| Metric | no_rag | hybrid | Target (PRD §8) | hybrid vs target |",
        "|---|---:|---:|---:|---:|",
        "| Questions asked | 30 | 30 |  |  |",
        "| Faithfulness | — | 0.900 (n=24) | ≥ 0.90 | met |",
        "| Answer correctness | 0.600 (n=30) | 0.740 (n=30) | ≥ 0.75 | not met |",
        "| Refusal accuracy | 0.870 (n=30) | 0.900 (n=30) | ≥ 0.90 | met |",
        "| Schema first-try validity | 1.000 (n=30) | 0.970 (n=30) | ≥ 0.97 | met |",
        "| Citation validity | — | 1.000 (n=30) | — | — |",
        "| Citation precision | — | 0.800 (n=22) | — | — |",
        "| RAG value: correctness(hybrid) - correctness(no_rag) | — | +0.140 (n=30 vs 30) "
        "| > 0 | met |",
        "| Latency p50 / p95, ms (warm) | 400 / 900 (n=30) | 812 / 1534 (n=27) |  |  |",
        "| Shadow cost per 1k questions, USD | $0.1000 (n=30) | $0.4210 (n=30) |  |  |",
    ]
)
LEGEND = (
    "`—`: not applicable or not scored. `n` is the number of questions a metric was scored on "
    "(not-applicable and unscored questions are left out). Met / not met compares the unrounded "
    "value with the target of PRD §8."
)
K_NOTE = (
    "`k` is the length of the list a retrieval mode returns (`K_DENSE`, `K_FTS` or `K_FUSED`); "
    "MRR runs over it."
)
NO_GENERATION = "**Generation baseline:** not committed yet (`eval/baselines/generation.json`)."
NO_RETRIEVAL = "**Retrieval baseline:** not committed yet (`eval/baselines/retrieval.json`)."

# --- The whole output -----------------------------------------------------------------------------


def test_both_baselines_render_to_exactly_this_text() -> None:
    text = render_report(retrieval_rows(), generation_rows())
    assert text == f"{RETRIEVAL_TEXT}\n\n{GENERATION_TEXT}\n\n{LEGEND} {K_NOTE}\n"


def test_without_a_generation_baseline_the_retrieval_part_stands_alone() -> None:
    expected = f"{RETRIEVAL_TEXT}\n\n{NO_GENERATION}\n\n{LEGEND} {K_NOTE}\n"
    assert render_report(retrieval_rows(), None) == expected
    assert render_report(retrieval_rows(), {}) == expected  # an empty file is "not committed yet"


def test_without_a_retrieval_baseline_the_generation_part_stands_alone() -> None:
    assert render_report(None, generation_rows()) == (
        f"{NO_RETRIEVAL}\n\n{GENERATION_TEXT}\n\n{LEGEND}\n"
    )


def test_with_no_baseline_at_all_it_says_so_twice_and_has_no_legend() -> None:
    assert render_report(None, None) == f"{NO_RETRIEVAL}\n\n{NO_GENERATION}\n"


def test_the_output_does_not_depend_on_the_order_of_the_rows() -> None:
    reversed_retrieval = dict(reversed(retrieval_rows().items()))
    reversed_generation = dict(reversed(generation_rows().items()))
    assert render_report(reversed_retrieval, reversed_generation) == render_report(
        retrieval_rows(), generation_rows()
    )


def test_a_generation_metric_is_never_printed_without_its_n() -> None:
    lines = render_report(None, generation_rows()).splitlines()
    table = [line for line in lines if line.startswith("| ")][1:]  # without the header
    labels = ("Faithfulness", "Answer correctness", "Refusal accuracy", "Schema", "Citation", "RAG")
    metric_rows = [line for line in table if line.startswith(tuple(f"| {x}" for x in labels))]
    assert len(metric_rows) == 7
    for line in metric_rows:
        for cell in (c.strip() for c in line.split("|")[2:4]):  # the no_rag and hybrid columns
            assert cell == "—" or "(n=" in cell, line


# --- Targets: at least the number, strictly more for the RAG value --------------------------------


@pytest.mark.parametrize(
    ("target", "measured", "met"),
    [
        (Target(0.80), 0.80, True),  # exactly on the target
        (Target(0.80), 0.7999, False),  # a hair under
        (Target(0.80), 0.81, True),
        (Target(0.90), 0.8999999999999999, True),  # float noise of a mean is not a miss
        (Target(0.90), 27 / 30, True),  # 27 of 30 is 0.9
        (Target(0.97), 0.969, False),
        (Target(0.65), 0.649998, False),  # 2e-6 under: far outside the 1e-9 absorption
        (Target(0.65), 0.65 - 1e-12, True),
        (RAG_VALUE_TARGET, 0.0, False),  # "> 0": no lift is not a lift
        (RAG_VALUE_TARGET, 1e-12, False),  # float noise of two equal means
        (RAG_VALUE_TARGET, 0.001, True),
        (RAG_VALUE_TARGET, -0.1, False),
    ],
)
def test_met_and_not_met_at_the_boundaries(target: Target, measured: float, met: bool) -> None:
    assert target.met(measured) is met


def test_the_targets_are_the_numbers_of_prd_section_8() -> None:
    assert {m: t.label for m, t in RETRIEVAL_TARGETS.items()} == {
        "recall@5": "≥ 0.80",
        "mrr": "≥ 0.60",
        "ndcg@5": "≥ 0.65",
    }
    assert {m: t.label for m, t in GENERATION_TARGETS.items()} == {
        "faithfulness": "≥ 0.90",
        "correctness": "≥ 0.75",
        "refusal_correctness": "≥ 0.90",
        "schema_first_try": "≥ 0.97",
    }
    assert RAG_VALUE_TARGET.label == "> 0"


def test_the_verdict_uses_the_unrounded_value() -> None:
    rows = generation_rows()
    rows["hybrid"] = generation_row(
        {"correctness": 0.7499999, "faithfulness": 0.9, "refusal_correctness": 0.9},
        30,
    )
    text = render_report(None, rows)
    assert "| Answer correctness | 0.600 (n=30) | 0.750 (n=30) | ≥ 0.75 | not met |" in text


# --- The RAG value row ----------------------------------------------------------------------------


def rag_line(hybrid: float, no_rag: float, *, n: tuple[int, int] = (30, 30)) -> str:
    rows = {
        "no_rag": generation_row({"correctness": no_rag}, n[1]),
        "hybrid": generation_row({"correctness": hybrid}, n[0]),
    }
    [line] = [
        line for line in render_report(None, rows).splitlines() if line.startswith("| RAG value")
    ]
    return line


@pytest.mark.parametrize(
    ("hybrid", "no_rag", "cells"),
    [
        (0.74, 0.6, "+0.140 (n=30 vs 30) | > 0 | met"),  # 0.74 - 0.6 is 0.14000000000000001
        (0.6, 0.6, "+0.000 (n=30 vs 30) | > 0 | not met"),  # equal: no lift
        (0.55, 0.6, "-0.050 (n=30 vs 30) | > 0 | not met"),  # RAG hurts: reported as it is
        (0.1 + 0.2, 0.3, "+0.000 (n=30 vs 30) | > 0 | not met"),  # 5.5e-17 of noise is not a lift
    ],
)
def test_the_rag_value_is_hybrid_minus_no_rag_with_both_ns(
    hybrid: float, no_rag: float, cells: str
) -> None:
    assert rag_line(hybrid, no_rag).endswith(f"| {cells} |")


def test_the_rag_value_shows_the_n_of_each_side() -> None:
    assert "(n=28 vs 30)" in rag_line(0.8, 0.6, n=(28, 30))


def test_the_rag_value_needs_both_rows_with_a_correctness() -> None:
    rows = generation_rows()
    nothing = "| RAG value: correctness(hybrid) - correctness(no_rag) | — | > 0 | — |"
    assert nothing in render_report(None, {"hybrid": rows["hybrid"]})
    assert nothing in render_report(None, {"no_rag": rows["no_rag"]})


# --- Missing and extra pieces ---------------------------------------------------------------------


def test_a_missing_latency_or_cost_is_a_dash_not_a_zero() -> None:
    rows = generation_rows()
    rows["hybrid"] = generation_row({"correctness": 0.8}, 30)  # no latency, no cost
    text = render_report(None, rows)
    assert "| Latency p50 / p95, ms (warm) | 400 / 900 (n=30) | — |" in text
    assert "| Shadow cost per 1k questions, USD | $0.1000 (n=30) | — |" in text


def test_a_metric_the_report_does_not_know_is_listed_by_its_name() -> None:
    rows = {"hybrid": generation_row({"correctness": 0.8, "something_new": 0.5}, 30)}
    assert "| something_new | 0.500 (n=30) | — | — |" in render_report(None, rows)


def test_without_a_hybrid_row_there_is_no_verdict() -> None:
    rows = {"no_rag": generation_rows()["no_rag"]}
    text = render_report(retrieval_rows(), rows)
    assert "| Answer correctness | 0.600 (n=30) | ≥ 0.75 | — |" in text
    without_hybrid = {k: v for k, v in retrieval_rows().items() if k != "hybrid"}
    assert "vs target" not in render_report(without_hybrid, None)


def test_rows_from_different_setups_show_every_value_not_only_one() -> None:
    rows = retrieval_rows()
    rows["fts"] = retrieval_row(
        {"recall@5": 0.58}, git_sha="abcdef0" + "1" * 33, git_dirty=True, golden_set_sha256="b" * 64
    )
    text = render_report(rows, None)
    assert "(sha256 8f084817 | bbbbbbbb)" in text
    assert "git 18fd399 | abcdef0 (dirty)" in text


def test_the_date_is_the_utc_day_and_nothing_in_the_output_is_a_clock_reading() -> None:
    rows = retrieval_rows()
    late = datetime(2026, 10, 1, 23, 59, 59, tzinfo=UTC)
    rows = {name: row.model_copy(update={"date": late}) for name, row in rows.items()}
    assert "git 18fd399, 2026-10-01." in render_report(rows, None)


# --- Files and the command ------------------------------------------------------------------------


def write_baselines(tmp_path: Path, *, generation: bool) -> tuple[Path, Path]:
    retrieval_path = tmp_path / "retrieval.json"
    generation_path = tmp_path / "generation.json"
    retrieval_path.write_text(
        json.dumps({n: r.model_dump(mode="json") for n, r in retrieval_rows().items()}),
        encoding="utf-8",
    )
    if generation:
        generation_path.write_text(
            json.dumps({n: r.model_dump(mode="json") for n, r in generation_rows().items()}),
            encoding="utf-8",
        )
    return retrieval_path, generation_path


def test_build_report_reads_the_files_that_exist(tmp_path: Path) -> None:
    retrieval_path, generation_path = write_baselines(tmp_path, generation=False)
    assert build_report(retrieval_path, generation_path) == render_report(retrieval_rows(), None)
    write_baselines(tmp_path, generation=True)
    assert build_report(retrieval_path, generation_path) == render_report(
        retrieval_rows(), generation_rows()
    )


def test_the_command_prints_the_report_and_works_before_the_generation_baseline_exists(
    tmp_path: Path,
) -> None:
    retrieval_path, generation_path = write_baselines(tmp_path, generation=False)
    result = runner.invoke(
        app,
        [
            "eval",
            "report",
            "--retrieval",
            str(retrieval_path),
            "--generation",
            str(generation_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output == f"{RETRIEVAL_TEXT}\n\n{NO_GENERATION}\n\n{LEGEND} {K_NOTE}\n"


def test_the_command_reports_a_baseline_that_does_not_validate(tmp_path: Path) -> None:
    retrieval_path, generation_path = write_baselines(tmp_path, generation=False)
    generation_path.write_text('{"hybrid": {"metrics": {}}}', encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "eval",
            "report",
            "--retrieval",
            str(retrieval_path),
            "--generation",
            str(generation_path),
        ],
    )
    assert result.exit_code == 1
    assert "Cannot read the baselines" in result.output


def test_the_command_reads_the_committed_baselines_by_default() -> None:
    result = runner.invoke(app, ["eval", "report"])
    assert result.exit_code == 0, result.output
    assert result.output == build_report(RETRIEVAL_BASELINE, GENERATION_BASELINE)


# --- The README is the output of the command ------------------------------------------------------

START, END = "<!-- eval-report:start -->", "<!-- eval-report:end -->"


def test_the_readme_block_is_the_report_of_the_committed_baselines() -> None:
    """The README's numbers are pasted from `grounded eval report`, never typed (AGENTS.md §7).
    A baseline PR that changes a file in eval/baselines/ must paste the new output."""
    readme = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert readme.count(START) == 1
    assert readme.count(END) == 1
    block = readme.split(START, 1)[1].split(END, 1)[0]
    assert block.strip() == build_report(RETRIEVAL_BASELINE, GENERATION_BASELINE).strip(), (
        "README.md is out of date: run `uv run grounded eval report` and paste its output between "
        f"{START} and {END}"
    )
