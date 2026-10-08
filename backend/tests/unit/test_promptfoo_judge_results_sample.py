"""The recorded promptfoo output with the two judge metrics
(tests/fixtures/promptfoo/results_sample_judge.json) against the contract of Tech §15.3: what
promptfoo keeps of a judge assertion's result, so the parser of the generation results (4.07) is
built on the real shape.

The sample is a recording, not a live check: if promptfoo or the assertions change, it has to be
recorded again (the README next to it says how it was made). The last test is the reading rule a
parser must follow, written out once on the sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

SAMPLE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample_judge.json"
)
PASSED, FAILED, ERRORED = 0, 1, 2  # promptfoo's failureReason
JUDGE_METRICS = ("faithfulness", "correctness")
SKIPPED = "skipped: not asked, the run stopped on an earlier ProviderRateLimited"


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    return document["results"]["results"]


def row_of(rows: list[dict[str, Any]], label: str, golden_id: str) -> dict[str, Any]:
    [found] = [
        r
        for r in rows
        if r["provider"]["label"] == label and r["metadata"]["golden"]["id"] == golden_id
    ]
    return found


def components(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {c["assertion"]["metric"]: c for c in row["gradingResult"]["componentResults"]}


def judge_of(rows: list[dict[str, Any]], label: str, golden_id: str, metric: str) -> dict[str, Any]:
    return components(row_of(rows, label, golden_id))[metric]


# --- the rows ------------------------------------------------------------------------------------


def test_every_row_was_graded_because_a_judge_error_does_not_error_the_row(
    rows: list[dict[str, Any]],
) -> None:
    assert len(rows) == 14
    for row in rows:
        assert row["gradingResult"] is not None
        assert row["failureReason"] in (PASSED, FAILED), row["metadata"]["golden"]["id"]
        assert set(components(row)) == {
            "schema_first_try",
            "citation_validity",
            "refusal_correctness",
            "citation_precision",
            *JUDGE_METRICS,
        }


def test_a_row_with_an_errored_judge_component_is_a_failed_assertion_not_an_errored_row(
    rows: list[dict[str, Any]],
) -> None:
    row = row_of(rows, "hybrid", "q008")

    assert components(row)["faithfulness"]["errored"] is True
    assert row["failureReason"] == FAILED
    # promptfoo copies the failing reason into `error`: it is not the provider's tagged error.
    assert row["error"].startswith("judge error (ProviderBadOutput)")
    assert "quota=" not in row["error"]


def test_promptfoo_did_not_call_the_provider_again_for_a_judge_error_that_says_429(
    document: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    reasons = [c["reason"] for r in rows for c in components(r).values()]
    assert any("429" in reason and "rate limit" in reason for reason in reasons)

    # One generator call per row: its retry-on-429 applies to provider errors, not to grading.
    assert document["results"]["stats"]["tokenUsage"]["numRequests"] == len(rows)


# --- the component shapes -------------------------------------------------------------------------


def test_every_judge_component_says_whether_it_is_errored_and_whether_it_is_not_applicable(
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        for metric in JUDGE_METRICS:
            component = components(row)[metric]
            assert isinstance(component["errored"], bool)
            assert isinstance(component["not_applicable"], bool)
            assert not (component["errored"] and component["not_applicable"])
            if component["not_applicable"]:
                assert (component["pass"], component["score"]) == (True, 1)
                assert component["reason"].startswith("N/A: ")
                assert "judge" not in component
            else:
                assert component["judge"]["metric"] == metric
                assert set(component["judge"]) >= {
                    "metric",
                    "prompt_version",
                    "judge_provider",
                    "judge_model",
                    "usage",
                    "error",
                }
            if component["errored"]:
                assert (component["pass"], component["score"]) == (False, 0)  # placeholders
                assert component["judge"]["error"] is not None
            elif not component["not_applicable"]:
                assert component["judge"]["error"] is None


def test_faithfulness_is_not_applicable_in_no_rag_and_for_a_refusal(
    rows: list[dict[str, Any]],
) -> None:
    na = {
        (r["provider"]["label"], r["metadata"]["golden"]["id"])
        for r in rows
        if components(r)["faithfulness"]["not_applicable"]
    }

    assert na == {
        ("no_rag", g) for g in ("q003", "q007", "q008", "q015", "q045", "q047", "q049")
    } | {("hybrid", "q045")}
    assert "refused" in judge_reason(rows, "hybrid", "q045")


def judge_reason(rows: list[dict[str, Any]], label: str, golden_id: str) -> str:
    return judge_of(rows, label, golden_id, "faithfulness")["reason"]


def test_correctness_is_judged_in_both_configs_and_for_a_refusal(
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        assert components(row)["correctness"]["not_applicable"] is False

    refusal = judge_of(rows, "hybrid", "q045", "correctness")
    assert (refusal["score"], refusal["pass"]) == (1, True)
    assert refusal["judge"]["verdict"] == "CORRECT"


def test_correctness_scores_are_one_half_or_zero_from_the_grade(rows: list[dict[str, Any]]) -> None:
    expected = {"CORRECT": 1, "PARTIALLY_CORRECT": 0.5, "INCORRECT": 0}
    graded = [
        c
        for r in rows
        if not (c := components(r)["correctness"])["errored"] and not c["not_applicable"]
    ]

    assert len(graded) == 9
    for component in graded:
        assert component["score"] == expected[component["judge"]["verdict"]]
        assert component["pass"] is (component["score"] == 1)
        assert component["judge"]["reason"]
        assert component["judge"]["attempts"] == 1


# --- faithfulness: the per-claim record -----------------------------------------------------------


def test_the_per_claim_verdicts_reasons_and_confidence_are_kept(rows: list[dict[str, Any]]) -> None:
    row = row_of(rows, "hybrid", "q003")
    component = components(row)["faithfulness"]
    judge = component["judge"]
    provider_claims = row["response"]["metadata"]["claims"]

    assert (component["score"], component["pass"]) == (0.5, False)
    assert (judge["n_claims"], judge["n_supported"], judge["n_errored"]) == (2, 1, 0)
    assert judge["prompt_version"].startswith("judge_faithfulness_v1@")
    assert judge["usage"]["input_tokens"] == 2420
    first, second = judge["claims"]
    assert (first["verdict"], second["verdict"]) == ("SUPPORTED", "NOT_SUPPORTED")
    for claim, recorded in zip(judge["claims"], provider_claims, strict=True):
        assert claim["claim"] == recorded["text"]
        assert claim["confidence"] == recorded["confidence"]  # the server's, joined by position
        assert claim["reason"]
        assert claim["error"] is None
        assert claim["cited_labels"] == ["c1", "c2"][: len(recorded["chunk_ids"])]
    assert set(first) == {
        "claim_index",
        "claim",
        "cited_labels",
        "confidence",
        "verdict",
        "reason",
        "decided_locally",
        "cache_hit",
        "attempts",
        "usage",
        "error",
    }


def test_a_claim_with_no_valid_source_is_not_supported_without_a_judge_call(
    rows: list[dict[str, Any]],
) -> None:
    judge = judge_of(rows, "hybrid", "q007", "faithfulness")["judge"]

    local = judge["claims"][1]
    assert (local["verdict"], local["decided_locally"], local["attempts"]) == (
        "NOT_SUPPORTED",
        True,
        0,
    )
    assert local["cited_labels"] == []
    assert local["usage"] == {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    assert judge_of(rows, "hybrid", "q007", "faithfulness")["score"] == 0.5


# --- errored: never a score -----------------------------------------------------------------------


def test_a_judge_that_cannot_grade_a_claim_makes_the_case_errored_and_keeps_the_rest(
    rows: list[dict[str, Any]],
) -> None:
    component = judge_of(rows, "hybrid", "q008", "faithfulness")

    assert (component["errored"], component["not_applicable"]) == (True, False)
    error = component["judge"]["error"]
    assert (error["kind"], error["is_quota"], error["provider_side"]) == (
        "ProviderBadOutput",
        False,
        False,
    )
    first, second = component["judge"]["claims"]
    assert (first["verdict"], first["error"]["kind"], first["attempts"]) == (
        None,
        "ProviderBadOutput",
        2,
    )
    assert second["verdict"] == "SUPPORTED"  # judged, and kept
    assert (component["judge"]["n_supported"], component["judge"]["n_errored"]) == (1, 1)


def test_a_provider_side_judge_failure_is_tagged_so_it_can_count_toward_inconclusive(
    rows: list[dict[str, Any]],
) -> None:
    for label in ("no_rag", "hybrid"):
        component = judge_of(rows, label, "q015", "correctness")
        assert component["errored"] is True
        error = component["judge"]["error"]
        assert (error["kind"], error["is_quota"], error["provider_side"]) == (
            "ProviderTimeout",
            False,
            True,
        )
        assert component["judge"]["verdict"] is None


def test_a_daily_quota_stops_the_judging_of_the_run(rows: list[dict[str, Any]]) -> None:
    quota = judge_of(rows, "hybrid", "q047", "correctness")
    assert (quota["errored"], quota["judge"]["error"]["is_quota"]) == (True, True)
    assert quota["judge"]["error"]["detail"].startswith("429 Too Many Requests")

    # After it, no judge assertion of the run asked the judge again: same tag, "skipped", no usage.
    skipped = [
        judge_of(rows, "hybrid", "q047", "faithfulness"),  # the other assertion of the same row
        judge_of(rows, "no_rag", "q049", "correctness"),
        judge_of(rows, "hybrid", "q049", "faithfulness"),
        judge_of(rows, "hybrid", "q049", "correctness"),
    ]
    for component in skipped:
        error = component["judge"]["error"]
        assert (error["kind"], error["is_quota"], error["provider_side"]) == (
            "ProviderRateLimited",
            True,
            True,
        )
        assert error["detail"] == SKIPPED
        assert component["judge"]["usage"]["input_tokens"] == 0
        assert component["judge"]["prompt_version"] is None
    # What is not applicable stays not applicable after the stop: nothing was to be asked.
    assert judge_of(rows, "no_rag", "q049", "faithfulness")["not_applicable"] is True


# --- how a parser reads it ------------------------------------------------------------------------


def reading(rows: list[dict[str, Any]], label: str, metric: str) -> tuple[float | None, int, int]:
    """The reading rule (Tech §15.3): leave out not-applicable and errored components; the mean of
    the rest, with ``n`` (scored cases) and the number of errored ones."""
    scores: list[float] = []
    errored = 0
    for row in rows:
        if row["provider"]["label"] != label:
            continue
        component = components(row)[metric]
        if component["not_applicable"]:
            continue
        if component["errored"]:
            errored += 1
            continue
        scores.append(component["score"])
    mean = sum(scores) / len(scores) if scores else None
    return mean, len(scores), errored


def test_the_means_leave_out_not_applicable_and_errored_cases(rows: list[dict[str, Any]]) -> None:
    # hybrid faithfulness: q003 0.5, q007 0.5, q015 1.0 scored; q008, q047 and q049 errored; q045
    # is N/A.
    mean, n, errored = reading(rows, "hybrid", "faithfulness")
    assert (n, errored) == (3, 3)
    assert mean == pytest.approx(2 / 3)
    # no_rag has no faithfulness at all: n = 0, and the mean is undefined, not 0 and not 1.
    assert reading(rows, "no_rag", "faithfulness") == (None, 0, 0)
    # correctness: scored in both configs, with the q015 timeouts and the quota stop left out.
    hybrid = reading(rows, "hybrid", "correctness")
    no_rag = reading(rows, "no_rag", "correctness")
    assert (hybrid[1], hybrid[2]) == (4, 3)
    assert (no_rag[1], no_rag[2]) == (5, 2)
    # promptfoo's own averages count the placeholders (1.0 for N/A, 0.0 for errored): not results.
    named = rows[0]["gradingResult"]["namedScores"]
    assert named["faithfulness"] == 1
