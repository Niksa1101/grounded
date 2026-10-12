"""``grounded eval diff`` (ticket 6.05): two configs of a retrieval eval, question by question.

**Every run in this file is synthetic.** The results are built in the test through the
``RetrievalRun`` schema from invented ranked lists (``synthetic-*`` names, made-up counts), so the
cases cover wins, losses, ties, a label one config does not retrieve, one file or two, and runs that
must be refused. They are not measurements and nothing here is a result of the project. The
per-question metrics come from ``score_question`` (the eval's own scorer), and a few expected values
are worked out by hand in the comments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.diff import Diff, DiffError, compare, render
from grounded.evals.retrieval_runner import golden_set_digest, score_question, write_run
from grounded.retrieval.config import RetrievalConfig, RetrievalMode
from grounded.schemas.eval import (
    GoldenItem,
    RerankRunStats,
    RetrievalConfigResult,
    RetrievalRun,
    RetrievalRunInfo,
)

PAGE = "docs/en/docs/tutorial/p.md"
GOLDEN_SHA = "a" * 64
INDEX_HASH = "b" * 64
MODEL = "synthetic-rerank"
runner = CliRunner()


@dataclass(frozen=True)
class Ref:
    section_id: str
    anchor_path: tuple[str, ...]


def label(anchor: str) -> str:
    return f"{PAGE}#{anchor}"


def item(id: str, grade_2: Sequence[str], grade_1: Sequence[str] = ()) -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "id": id,
            "question": f"synthetic question {id}?",
            "type": "how_to",
            "answerable": True,
            "reference_answer": "synthetic",
            "relevant_sections": [
                *({"section": label(a), "grade": 2} for a in grade_2),
                *({"section": label(a), "grade": 1} for a in grade_1),
            ],
        }
    )


ITEMS = [
    item("q001", ["a1"]),
    item("q002", ["a2"]),
    item("q003", ["a3"]),
    item("q004", ["a4"]),
    item("q005", ["b1", "b2"], grade_1=["c1"]),
]

TOTAL = 8  # the length of every synthetic ranked list


def ranked(placed: Mapping[str, int], total: int = TOTAL) -> list[str]:
    """A ranked list of ``total`` anchors with each anchor of ``placed`` at its 1-based rank and
    filler anchors elsewhere. Two lists with the same anchors placed anywhere hold the same
    candidates, which is what a reranker returns (the same candidates in another order)."""
    slots: list[str | None] = [None] * total
    for anchor, rank in placed.items():
        slots[rank - 1] = anchor
    fillers = iter(f"f{n}" for n in range(1, total + 1))
    return [slot if slot is not None else next(fillers) for slot in slots]


# Where each question's labels sit, per config. Hand-worked expectations (single grade-2 label, so
# the ideal DCG is 3 and nDCG@k = 1 / log2(rank + 1) within the cutoff):
#   q001  3 -> 1   nDCG@5 0.5 -> 1.0 (+0.5), MRR 1/3 -> 1 (+0.667)            a win
#   q002  1 -> 4   nDCG@5 1.0 -> 1/log2(5) = 0.431 (-0.569), MRR 1 -> 0.25     a loss
#   q003  2 -> 2   no change                                                    a tie
#   q004  7 -> 2   nDCG@5 0 -> 1/log2(3) = 0.631 (+0.631), Recall@5 0 -> 1     a win
#   q005  b2 6 -> 5 (b1 stays 1, the grade-1 c1 stays 3): Recall@5 0.5 -> 1     a win
HYBRID = {
    "q001": {"a1": 3},
    "q002": {"a2": 1},
    "q003": {"a3": 2},
    "q004": {"a4": 7},
    "q005": {"b1": 1, "c1": 3, "b2": 6},
}
RERANKED = {
    "q001": {"a1": 1},
    "q002": {"a2": 4},
    "q003": {"a3": 2},
    "q004": {"a4": 2},
    "q005": {"b1": 1, "c1": 3, "b2": 5},
}

STATS = RerankRunStats(
    model=MODEL, live_calls=7, cache_hits=3, latency_p50_ms=100.0, latency_p95_ms=200.0
)


def config_for(mode: RetrievalMode, **overrides: object) -> RetrievalConfig:
    fields: dict[str, object] = {
        "mode": mode,
        "k_dense": 20,
        "k_fts": 20,
        "k_fused": 40,
        "k_context": 5,
        "rrf_k": 60,
    }
    if mode == "hybrid_rerank":
        fields |= {"rerank_provider": "cohere", "rerank_model": MODEL}
    return RetrievalConfig.model_validate(fields | overrides)


def result_for(
    mode: RetrievalMode,
    placements: Mapping[str, Mapping[str, int]],
    *,
    items: Sequence[GoldenItem] = ITEMS,
    k: int = 40,
    config: RetrievalConfig | None = None,
    rerank: RerankRunStats | None = None,
) -> RetrievalConfigResult:
    by_id = {i.id: i for i in items}
    questions = [
        score_question(
            by_id[question_id],
            [Ref(label(anchor), (anchor,)) for anchor in ranked(placed)],
        )
        for question_id, placed in placements.items()
    ]
    names = ("recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10")
    chosen = config or config_for(mode)
    return RetrievalConfigResult(
        config=mode,
        retrieval_config=chosen,
        retrieval_config_hash=chosen.config_hash,
        k=k,
        n=len(questions),
        skipped_unanswerable=5,
        metrics={m: math.fsum(q.metrics[m] for q in questions) / len(questions) for m in names},
        questions=questions,
        rerank=rerank,
    )


def info_for(
    *, golden_sha: str = GOLDEN_SHA, index_hash: str = INDEX_HASH, day: int = 1
) -> RetrievalRunInfo:
    return RetrievalRunInfo(
        date=datetime(2026, 1, day, tzinfo=UTC),
        git_sha="c" * 40,
        git_dirty=False,
        golden_set_version="v1",
        golden_set_sha256=golden_sha,
        index_version_id=1,
        index_config_hash=index_hash,
        fastapi_ref="0.0.0-synthetic",
        fastapi_sha="d" * 40,
        embedding_model="synthetic-embedder",
        embedding_dim=8,
    )


def run_of(*results: RetrievalConfigResult, info: RetrievalRunInfo | None = None) -> RetrievalRun:
    return RetrievalRun(
        info=info or info_for(), configs={result.config: result for result in results}
    )


def hybrid_result(**kwargs: object) -> RetrievalConfigResult:
    return result_for("hybrid", HYBRID, **kwargs)  # type: ignore[arg-type]


def reranked_result(
    placements: Mapping[str, Mapping[str, int]] = RERANKED, **kwargs: object
) -> RetrievalConfigResult:
    kwargs.setdefault("rerank", STATS)
    return result_for("hybrid_rerank", placements, **kwargs)  # type: ignore[arg-type]


def both_in_one_file() -> RetrievalRun:
    return run_of(hybrid_result(), reranked_result())


def diff_of(
    runs: Sequence[RetrievalRun], configs: Sequence[str] = ("hybrid", "hybrid_rerank")
) -> Diff:
    return compare(runs, configs, golden=ITEMS, golden_sha256=GOLDEN_SHA)


def refused(
    runs: Sequence[RetrievalRun],
    configs: Sequence[str] = ("hybrid", "hybrid_rerank"),
    *,
    golden: Sequence[GoldenItem] = ITEMS,
    golden_sha256: str = GOLDEN_SHA,
) -> list[str]:
    with pytest.raises(DiffError) as raised:
        compare(runs, configs, golden=golden, golden_sha256=golden_sha256)
    return raised.value.problems


# --- The numbers ---------------------------------------------------------------------------------


def test_one_file_with_both_configs_gives_the_paired_deltas() -> None:
    diff = diff_of([both_in_one_file()])
    by_id = {q.id: q for q in diff.questions}

    assert (diff.base, diff.other, diff.n) == ("hybrid", "hybrid_rerank", 5)
    assert by_id["q001"].deltas["ndcg@5"] == pytest.approx(0.5)
    assert by_id["q001"].deltas["mrr"] == pytest.approx(1 - 1 / 3)
    assert by_id["q002"].deltas["ndcg@5"] == pytest.approx(1 / math.log2(5) - 1)
    assert by_id["q002"].deltas["mrr"] == pytest.approx(0.25 - 1)
    assert by_id["q003"].deltas["ndcg@5"] == 0.0
    assert by_id["q004"].deltas["ndcg@5"] == pytest.approx(1 / math.log2(3))
    assert by_id["q004"].deltas["recall@5"] == 1.0
    assert by_id["q005"].deltas["recall@5"] == pytest.approx(0.5)
    # The mean delta is the mean of the paired deltas, and the two means differ by exactly that.
    for metric in ("recall@5", "mrr", "ndcg@5", "ndcg@10"):
        assert diff.mean_deltas[metric] == pytest.approx(
            diff.other_means[metric] - diff.base_means[metric]
        )
    assert (diff.wins, diff.losses, diff.ties) == (3, 1, 1)


def test_questions_are_sorted_by_the_headline_delta_losses_first_then_by_id() -> None:
    diff = diff_of([both_in_one_file()])

    assert [q.id for q in diff.questions] == ["q002", "q003", "q005", "q001", "q004"]
    tied = run_of(
        result_for("hybrid", {"q001": {"a1": 3}, "q002": {"a2": 3}}),
        result_for("hybrid_rerank", {"q001": {"a1": 1}, "q002": {"a2": 1}}, rerank=STATS),
    )
    # Equal deltas are ordered by id, whatever the order of the file.
    assert [q.id for q in diff_of([tied]).questions] == ["q001", "q002"]


def test_the_grade_2_labels_of_each_question_show_their_ranks_in_both_configs() -> None:
    by_id = {q.id: q for q in diff_of([both_in_one_file()]).questions}

    assert [(a.label, a.base, a.other) for a in by_id["q001"].answers] == [(label("a1"), 3, 1)]
    # Two grade-2 labels, in the golden set's order; the grade-1 label c1 is not listed.
    assert [(a.label, a.base, a.other) for a in by_id["q005"].answers] == [
        (label("b1"), 1, 1),
        (label("b2"), 6, 5),
    ]


def test_a_label_one_config_does_not_retrieve_is_a_null_rank() -> None:
    dense = result_for("dense", {"q001": {"a1": 2}, "q002": {}}, k=20)  # q002: a2 not retrieved
    hybrid = result_for("hybrid", {"q001": {"a1": 2}, "q002": {"a2": 5}})
    # dense and hybrid hold different candidates by design: the generic pair is not refused.
    diff = compare(
        [run_of(dense, hybrid)],
        ["dense", "hybrid"],
        golden=ITEMS,
        golden_sha256=GOLDEN_SHA,
    )

    by_id = {q.id: q for q in diff.questions}
    assert [(a.base, a.other) for a in by_id["q002"].answers] == [(None, 5)]
    text = render(diff)
    assert "`tutorial/p.md#a2`: null -> 5" in text
    assert "`tutorial/p.md#a1`: 2 -> 2" in text
    assert "k = 20 (dense), 40 (hybrid)" in text  # the lists differ in length: both are shown
    assert by_id["q002"].deltas["ndcg@5"] == pytest.approx(1 / math.log2(6))


def nudged(
    result: RetrievalConfigResult, question_id: str, metric: str, by: float
) -> RetrievalConfigResult:
    """``result`` with one stored per-question number moved by ``by`` and the means recomputed, so
    the file still adds up (a hand-made file would be refused otherwise)."""
    questions = [
        q.model_copy(update={"metrics": {**q.metrics, metric: q.metrics[metric] + by}})
        if q.id == question_id
        else q
        for q in result.questions
    ]
    means = {m: math.fsum(q.metrics[m] for q in questions) / len(questions) for m in result.metrics}
    return result.model_copy(update={"questions": questions, "metrics": means})


@pytest.mark.parametrize(
    ("noise", "outcome"),
    [
        (1e-12, (1, 0, 1)),  # float noise is a tie
        (1e-6, (2, 0, 0)),  # a real difference, however small, is a win
        (-1e-6, (1, 1, 0)),  # and a loss
    ],
)
def test_the_tie_rule_is_float_noise_only(noise: float, outcome: tuple[int, int, int]) -> None:
    # q001 wins by 0.5 in both cases (rank 3 -> 1); q003 keeps its rank and only its stored nDCG@5
    # is moved by ``noise``.
    hybrid = result_for("hybrid", {"q001": {"a1": 3}, "q003": {"a3": 2}})
    reranked = result_for("hybrid_rerank", {"q001": {"a1": 1}, "q003": {"a3": 2}}, rerank=STATS)

    diff = diff_of([run_of(hybrid, nudged(reranked, "q003", "ndcg@5", noise))])

    assert (diff.wins, diff.losses, diff.ties) == outcome


# --- The text ------------------------------------------------------------------------------------


def test_the_markdown_has_the_means_with_n_the_outcome_and_the_per_question_rows() -> None:
    text = render(diff_of([both_in_one_file()]))

    assert text.startswith("# `hybrid_rerank` against `hybrid`\n")
    assert (
        "Paired over the same 5 answerable questions (5 unanswerable ones are not scored)" in text
    )
    assert "golden set v1 (sha256 aaaaaaaa)" in text
    assert "index 0.0.0-synthetic@bbbbbbbb" in text
    assert "k = 40" in text
    assert "| Metric | n | hybrid | hybrid_rerank | Delta |" in text
    assert "| nDCG@5 | 5 | " in text
    assert (
        "`hybrid_rerank` is better on 3 questions, `hybrid` is better on 1, and they tie on 1 "
        "(n = 5)."
    ) in text
    assert "A tie is a difference of at most 1e-9 in nDCG@5 (float noise)" in text
    # q001, worked out by hand above: Recall unchanged, MRR +0.667, nDCG@5 and @10 +0.500.
    assert (
        "| q001 | how_to | 0.000 | 0.000 | +0.667 | +0.500 | +0.500 | `tutorial/p.md#a1`: 3 -> 1 |"
    ) in text
    assert (
        "| q003 | how_to | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | `tutorial/p.md#a3`: 2 -> 2 |"
        in text
    )
    assert "`tutorial/p.md#b1`: 1 -> 1; `tutorial/p.md#b2`: 6 -> 5 |" in text
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_the_table_rows_follow_the_sorted_questions() -> None:
    text = render(diff_of([both_in_one_file()]))

    positions = [text.index(f"| {q} |") for q in ("q002", "q003", "q005", "q001", "q004")]
    assert positions == sorted(positions)


def test_the_honesty_note_is_computed_from_n_and_compares_the_headline_delta() -> None:
    # n = 5, so one question is 1/5 = 0.200. The mean nDCG@5 delta of the fixture is about +0.156.
    small = render(diff_of([both_in_one_file()]))
    assert "1/5 = 0.200" in small
    assert "is smaller than one question's worth" in small
    assert "not reported as a lift or a loss (AGENTS.md §7)" in small

    # n = 2, one question wins by exactly 0.5 of nDCG@5: the mean delta is 0.25 against 1/2.
    one_win = run_of(
        result_for("hybrid", {"q001": {"a1": 3}, "q003": {"a3": 2}}),
        result_for("hybrid_rerank", {"q001": {"a1": 1}, "q003": {"a3": 2}}, rerank=STATS),
    )
    assert "+0.250, is smaller than one question's worth" in render(diff_of([one_win]))

    # Both win by 0.5: the mean delta is exactly one question's worth (a boundary is "at least").
    two_wins = run_of(
        result_for("hybrid", {"q001": {"a1": 3}, "q003": {"a3": 3}}),
        result_for("hybrid_rerank", {"q001": {"a1": 1}, "q003": {"a3": 1}}, rerank=STATS),
    )
    text = render(diff_of([two_wins]))
    assert "1/2 = 0.500" in text
    assert "+0.500, is at least one question's worth" in text


def test_the_rerank_counts_and_latency_are_printed_only_as_the_file_carries_them() -> None:
    carried = render(diff_of([both_in_one_file()]))
    assert (
        f"`hybrid_rerank` reranked with {MODEL}: 7 live calls, 3 answered from the rerank cache; "
        "live latency p50 100 / p95 200 ms."
    ) in carried
    assert "`hybrid`" not in carried.split("## Rerank calls", 1)[1].split("## Per question", 1)[0]

    cache_only = RerankRunStats(model=MODEL, live_calls=0, cache_hits=10)
    text = render(diff_of([run_of(hybrid_result(), reranked_result(rerank=cache_only))]))
    assert "0 live calls, 10 answered from the rerank cache; no live call, so no latency." in text

    absent = render(diff_of([run_of(hybrid_result(), reranked_result(rerank=None))]))
    assert "`hybrid_rerank`: the results file has no rerank call counts or latency." in absent

    plain = run_of(
        result_for("dense", {"q001": {"a1": 2}}, k=20), result_for("hybrid", {"q001": {"a1": 2}})
    )
    no_rerank = render(diff_of([plain], ("dense", "hybrid")))
    assert "## Rerank calls" not in no_rerank  # nothing to say, nothing invented


def test_the_text_is_deterministic_and_carries_no_date_or_path() -> None:
    first = render(diff_of([both_in_one_file()]))
    later = run_of(hybrid_result(), reranked_result(), info=info_for(day=20))

    assert render(diff_of([both_in_one_file()])) == first
    assert render(diff_of([later])) == first  # the run's date is not in the text
    assert "2026" not in first
    assert "\\" not in first


# --- One file or two -----------------------------------------------------------------------------


def test_two_files_give_the_same_comparison_as_one() -> None:
    one = both_in_one_file()
    first, second = run_of(hybrid_result()), run_of(reranked_result())

    two = diff_of([first, second])

    assert two.questions == diff_of([one]).questions
    assert render(two) == render(diff_of([one])).replace("one results file", "two results files")


def test_with_two_files_each_config_is_read_from_its_own_file() -> None:
    # File 1 is the fixture: hybrid has q001 at rank 3 and hybrid_rerank at rank 1. File 2 swaps
    # them: its hybrid has q001 at rank 1 and its hybrid_rerank at rank 3.
    file_1 = both_in_one_file()
    file_2 = run_of(
        result_for("hybrid", {**HYBRID, "q001": {"a1": 1}}),
        reranked_result({**RERANKED, "q001": {"a1": 3}}),
    )

    def q001(runs: list[RetrievalRun]) -> tuple[int | None, int | None]:
        [answer] = {q.id: q for q in diff_of(runs).questions}["q001"].answers
        return answer.base, answer.other

    assert q001([file_1, file_2]) == (3, 3)  # hybrid of file 1, hybrid_rerank of file 2
    assert q001([file_2, file_1]) == (1, 1)  # hybrid of file 2, hybrid_rerank of file 1
    assert q001([file_1]) == (3, 1)


def test_a_missing_config_is_named_with_the_file_and_what_the_file_holds() -> None:
    only_hybrid = run_of(hybrid_result())

    [one_file] = refused([only_hybrid])
    assert one_file == "config 'hybrid_rerank' is not in the results file (it has: hybrid)"
    [second_file] = refused([only_hybrid, run_of(hybrid_result())])
    assert second_file == "config 'hybrid_rerank' is not in results file 2 (it has: hybrid)"
    both = refused([run_of(reranked_result()), run_of(hybrid_result())])
    assert len(both) == 2  # each lookup is reported, not only the first


def test_the_number_of_files_and_configs_is_checked() -> None:
    run = both_in_one_file()

    assert "two different configs" in refused([run], ["hybrid"])[0]
    assert "two different configs" in refused([run], ["hybrid", "hybrid"])[0]
    assert "one results file" in refused([run, run, run])[0]


# --- Runs that are not comparable ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "second_info"),
    [
        ("golden_set_sha256", info_for(golden_sha="e" * 64)),
        ("index_config_hash", info_for(index_hash="f" * 64)),
    ],
)
def test_two_files_scored_on_different_data_are_refused(
    field: str, second_info: RetrievalRunInfo
) -> None:
    first = run_of(hybrid_result())
    second = run_of(reranked_result(), info=second_info)

    problems = refused([first, second])

    assert any(field in p and "different data" in p for p in problems), problems


def test_a_golden_file_that_is_not_the_one_the_run_scored_is_refused() -> None:
    problems = refused([both_in_one_file()], golden_sha256="9" * 64)

    assert problems == [
        "the golden-set file (sha256 99999999) is not the one the run scored (sha256 aaaaaaaa); "
        "pass that file with --golden"
    ]


def test_other_retrieval_settings_mean_other_candidates() -> None:
    narrower = config_for("hybrid_rerank", k_fused=30)
    problems = refused(
        [run_of(hybrid_result(), reranked_result(config=narrower, k=30))],
    )

    assert any("retrieval settings differ (k_fused 40 vs 30)" in p for p in problems), problems
    assert any("k differs" in p for p in problems), problems


def test_questions_that_differ_are_refused_and_named() -> None:
    fewer = {k: v for k, v in RERANKED.items() if k != "q005"}

    problems = refused([run_of(hybrid_result(), reranked_result(fewer))])

    assert any(
        "the questions differ: only in hybrid: q005; only in hybrid_rerank: none" in p
        for p in problems
    ), problems


def test_a_question_that_is_not_in_the_golden_file_is_refused() -> None:
    golden = [i for i in ITEMS if i.id != "q005"]

    problems = refused([both_in_one_file()], golden=golden)

    assert problems == ["q005 is not an answerable question of the golden set"]


def test_a_file_that_lacks_a_label_of_the_golden_set_is_refused_for_either_config() -> None:
    reranked = reranked_result()
    thin = [q.model_copy(update={"ranks": {}}) if q.id == "q001" else q for q in reranked.questions]

    problems = refused([run_of(hybrid_result(), reranked.model_copy(update={"questions": thin}))])

    assert problems == [
        f"hybrid_rerank q001: labels of the golden set missing in the file: ['{label('a1')}']"
    ]


def test_a_file_whose_means_are_not_the_means_of_its_rows_is_refused() -> None:
    edited = reranked_result().model_copy(
        update={"metrics": {**reranked_result().metrics, "ndcg@5": 0.99}}
    )

    problems = refused([run_of(hybrid_result(), edited)])

    assert len(problems) == 1
    assert "hybrid_rerank: the mean ndcg@5 in the file (0.99) is not the mean of its" in problems[0]


def test_a_file_whose_n_is_not_its_row_count_is_refused() -> None:
    wrong = reranked_result().model_copy(update={"n": 4})

    problems = refused([run_of(hybrid_result(), wrong)])

    assert problems == ["hybrid_rerank: n is 4 but the file has 5 rows"]


# --- A rerank run that is not a rerank run -------------------------------------------------------


def test_a_reranked_list_with_other_candidates_is_refused() -> None:
    reranked = reranked_result()
    # One candidate of q004 replaced by a chunk hybrid never retrieved.
    rows = [
        q.model_copy(update={"retrieved": [*q.retrieved[:-1], "elsewhere.md#x"]})
        if q.id == "q004"
        else q
        for q in reranked.questions
    ]

    problems = refused([run_of(hybrid_result(), reranked.model_copy(update={"questions": rows}))])

    assert problems == [
        "hybrid_rerank does not rank the same candidates as hybrid (q004): a reranker only "
        "reorders them"
    ]


def test_a_reranked_list_equal_to_the_hybrid_order_everywhere_is_a_degraded_run() -> None:
    same_order = reranked_result(HYBRID)

    problems = refused([run_of(hybrid_result(), same_order)])

    assert len(problems) == 1
    assert "has the hybrid order for every question: the reranker did not run" in problems[0]
    assert "a degraded run is not a result" in problems[0]


def test_a_rerank_run_without_a_single_call_or_with_another_model_is_refused() -> None:
    no_calls = RerankRunStats(model=MODEL, live_calls=0, cache_hits=0)
    other_model = RerankRunStats(model="synthetic-other", live_calls=7, cache_hits=3)

    [calls] = refused([run_of(hybrid_result(), reranked_result(rerank=no_calls))])
    [model] = refused([run_of(hybrid_result(), reranked_result(rerank=other_model))])

    assert calls == "hybrid_rerank: the run recorded no rerank call at all"
    assert model == (
        "hybrid_rerank: the rerank model of the calls (synthetic-other) is not the one of its "
        "config (synthetic-rerank)"
    )


def test_a_plain_config_that_says_it_reranked_is_refused() -> None:
    odd = config_for("hybrid", rerank_provider="cohere", rerank_model=MODEL)

    problems = refused([run_of(hybrid_result(config=odd), reranked_result())])

    assert problems == ["hybrid: the config says it used a reranker, so it is not plain"]


def test_the_config_order_may_be_reversed() -> None:
    diff = diff_of([both_in_one_file()], ("hybrid_rerank", "hybrid"))

    assert (diff.base, diff.other) == ("hybrid_rerank", "hybrid")
    assert (
        diff.questions[0].id == "q004"
    )  # the delta is hybrid minus hybrid_rerank: q004 loses most
    assert (diff.wins, diff.losses, diff.ties) == (1, 3, 1)


def test_every_reason_is_listed_together() -> None:
    first = run_of(hybrid_result())
    second = run_of(
        reranked_result({k: v for k, v in RERANKED.items() if k != "q005"}),
        info=info_for(index_hash="f" * 64),
    )

    problems = refused([first, second], golden_sha256="9" * 64)

    joined = "\n".join(problems)
    assert "index_config_hash" in joined
    assert "is not the one the run scored" in joined
    assert "the questions differ" in joined
    assert len(problems) >= 3


# --- The command ---------------------------------------------------------------------------------


@pytest.fixture
def golden_file(tmp_path: Path) -> Path:
    path = tmp_path / "golden_set.v1.jsonl"
    path.write_bytes(("\n".join(i.model_dump_json() for i in ITEMS) + "\n").encode("utf-8"))
    return path


def run_with_golden_hash(golden: Path, *results: RetrievalConfigResult) -> RetrievalRun:
    return run_of(*results, info=info_for(golden_sha=golden_set_digest(golden)))


def test_the_command_prints_the_comparison_of_one_file(tmp_path: Path, golden_file: Path) -> None:
    path = tmp_path / "compare.json"
    write_run(path, run_with_golden_hash(golden_file, hybrid_result(), reranked_result()))

    done = runner.invoke(
        app,
        [
            "eval",
            "diff",
            "--config",
            "hybrid",
            "--config",
            "hybrid_rerank",
            "--results",
            str(path),
            "--golden",
            str(golden_file),
        ],
    )

    assert done.exit_code == 0, done.output
    assert done.output.startswith("# `hybrid_rerank` against `hybrid`\n")
    assert "| q001 | how_to |" in done.output
    assert "is smaller than one question's worth" in done.output


def test_the_command_defaults_to_the_hybrid_pair_and_reads_two_files(
    tmp_path: Path, golden_file: Path
) -> None:
    first, second = tmp_path / "hybrid.json", tmp_path / "rerank.json"
    write_run(first, run_with_golden_hash(golden_file, hybrid_result()))
    write_run(second, run_with_golden_hash(golden_file, reranked_result()))

    done = runner.invoke(
        app,
        [
            "eval",
            "diff",
            "--results",
            str(first),
            "--results",
            str(second),
            "--golden",
            str(golden_file),
        ],
    )

    assert done.exit_code == 0, done.output
    assert "from two results files" in done.output


def test_the_command_reads_the_golden_set_the_run_recorded_when_none_is_given(
    tmp_path: Path, golden_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(grounded.cli, "GOLDEN_DIR", tmp_path)  # golden_set.v1.jsonl is in tmp_path
    path = tmp_path / "compare.json"
    write_run(path, run_with_golden_hash(golden_file, hybrid_result(), reranked_result()))

    done = runner.invoke(app, ["eval", "diff", "--results", str(path)])

    assert done.exit_code == 0, done.output


def test_the_command_refuses_with_every_reason_and_exit_1(
    tmp_path: Path, golden_file: Path
) -> None:
    path = tmp_path / "compare.json"
    write_run(path, run_of(hybrid_result(), reranked_result()))  # the file's golden hash is made up

    done = runner.invoke(
        app, ["eval", "diff", "--results", str(path), "--golden", str(golden_file)]
    )

    assert done.exit_code == 1
    assert "Not compared, the runs cannot be compared:" in done.output
    assert "- the golden-set file (sha256 " in done.output
    assert "# `hybrid_rerank`" not in done.output  # no table beside a refusal


def test_the_command_says_which_file_cannot_be_read(tmp_path: Path, golden_file: Path) -> None:
    missing = tmp_path / "nope.json"
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    for path in (missing, broken):
        done = runner.invoke(
            app, ["eval", "diff", "--results", str(path), "--golden", str(golden_file)]
        )
        assert done.exit_code == 1
        assert f"Cannot read the results file {path}" in done.output


def test_the_command_says_when_the_golden_file_cannot_be_read(
    tmp_path: Path, golden_file: Path
) -> None:
    path = tmp_path / "compare.json"
    write_run(path, run_with_golden_hash(golden_file, hybrid_result(), reranked_result()))

    done = runner.invoke(
        app, ["eval", "diff", "--results", str(path), "--golden", str(tmp_path / "gone.jsonl")]
    )

    assert done.exit_code == 1
    assert "Cannot read the golden set" in done.output
