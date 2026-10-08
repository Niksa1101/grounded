"""The recorded promptfoo output (tests/fixtures/promptfoo/results_sample.json) against the contract
of Tech §15.3: what promptfoo keeps of our results, so a parser can rely on it (4.07).

The sample is a recording, not a live check: if promptfoo or the provider changes, it has to be
recorded again (the README next to it says how it was made)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample.json"
METRICS = {"schema_first_try", "citation_validity", "refusal_correctness", "citation_precision"}
TAG = re.compile(r"^\[(?P<kind>\w+) quota=(?P<quota>true|false)\] ")

PASSED, FAILED, ERRORED = 0, 1, 2  # promptfoo's failureReason


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    document = json.loads(SAMPLE.read_text(encoding="utf-8"))
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


def test_there_is_a_row_per_question_and_provider(rows: list[dict[str, Any]]) -> None:
    pairs = {(r["provider"]["label"], r["metadata"]["golden"]["id"]) for r in rows}

    assert len(rows) == len(pairs) == 14
    assert {label for label, _ in pairs} == {"no_rag", "hybrid"}


def test_every_row_carries_its_golden_payload_in_the_merged_metadata(
    rows: list[dict[str, Any]],
) -> None:
    for row in rows:
        golden = row["metadata"]["golden"]
        assert set(golden) == {
            "id",
            "type",
            "answerable",
            "relevant",
            "reference_answer",
            "golden_set_version",
            "golden_set_sha256",
        }
        assert re.fullmatch(r"[0-9a-f]{64}", golden["golden_set_sha256"])
        # promptfoo redacts the long token-like digest in the echo of the test case, not here.
        assert row["testCase"]["metadata"]["golden"]["golden_set_sha256"] == "[REDACTED]"


def test_an_errored_row_carries_the_tag_the_gate_counts(rows: list[dict[str, Any]]) -> None:
    errored = [r for r in rows if r["failureReason"] == ERRORED]

    assert len(errored) == 8
    for row in errored:
        tag = TAG.match(row["error"])
        assert tag is not None, row["error"]
        metadata = row["response"]["metadata"]
        assert metadata["error_kind"] == tag["kind"]
        assert metadata["is_quota"] is (tag["quota"] == "true")
        assert metadata["rateLimitKind"] == "quota"  # promptfoo's switch: never retry it
        assert row["gradingResult"] is None  # no assertion runs on an errored case
        assert row["success"] is False


def test_the_error_kinds_the_scripted_run_produced(rows: list[dict[str, Any]]) -> None:
    kinds = {
        (r["provider"]["label"], r["metadata"]["golden"]["id"]): r["response"]["metadata"][
            "error_kind"
        ]
        for r in rows
        if r["failureReason"] == ERRORED
    }

    assert kinds[("hybrid", "q008")] == "ProviderBadOutput"
    assert kinds[("hybrid", "q015")] == "BackoffExhaustedError"
    assert kinds[("hybrid", "q047")] == "ProviderRateLimited"
    assert kinds[("hybrid", "q049")] == "ProviderRateLimited"  # skipped after the quota


def test_the_backoff_error_names_its_base_and_the_time_it_waited(
    rows: list[dict[str, Any]],
) -> None:
    metadata = row_of(rows, "hybrid", "q015")["response"]["metadata"]

    assert metadata["error_bases"] == ["ProviderRateLimited", "ProviderError"]
    assert metadata["is_quota"] is False
    assert metadata["waited_s"] == 120.0


def test_a_bad_output_row_keeps_the_tokens_it_spent(rows: list[dict[str, Any]]) -> None:
    row = row_of(rows, "hybrid", "q008")

    assert row["tokenUsage"]["total"] > 0
    assert row["response"]["metadata"]["validation_retries"] == 1
    assert row["response"]["metadata"]["validation_error"]


def test_a_skipped_row_repeats_the_quota_tag_without_a_call(rows: list[dict[str, Any]]) -> None:
    for label in ("no_rag", "hybrid"):
        row = row_of(rows, label, "q049")
        metadata = row["response"]["metadata"]
        assert "skipped:" in row["error"]
        assert metadata["skipped"] is True
        assert metadata["is_quota"] is True
        assert row["cost"] == 0


def test_a_scored_row_has_one_component_per_metric(rows: list[dict[str, Any]]) -> None:
    scored = [r for r in rows if r["failureReason"] != ERRORED]

    assert len(scored) == 6
    for row in scored:
        assert set(components(row)) == METRICS


def test_not_applicable_components_are_marked_and_the_rest_are_not(
    rows: list[dict[str, Any]],
) -> None:
    no_rag = components(row_of(rows, "no_rag", "q003"))
    hybrid = components(row_of(rows, "hybrid", "q003"))
    refusal = components(row_of(rows, "hybrid", "q045"))

    assert {m for m, c in no_rag.items() if c["not_applicable"]} == {
        "citation_validity",
        "citation_precision",
    }
    assert not any(c["not_applicable"] for c in hybrid.values())
    assert {m for m, c in refusal.items() if c["not_applicable"]} == {"citation_precision"}
    for component in (*no_rag.values(), *hybrid.values(), *refusal.values()):
        if component["not_applicable"]:
            assert (component["pass"], component["score"]) == (True, 1)
            assert component["reason"].startswith("N/A: ")


def test_a_failed_assertion_is_a_failure_reason_one(rows: list[dict[str, Any]]) -> None:
    row = row_of(rows, "hybrid", "q007")

    assert row["failureReason"] == FAILED
    assert components(row)["citation_validity"]["score"] == 0
    assert components(row)["citation_validity"]["pass"] is False
    assert row["response"]["metadata"]["invalid_citation_count"] == 2


def test_a_success_row_has_the_output_and_the_metadata_the_assertions_read(
    rows: list[dict[str, Any]],
) -> None:
    row = row_of(rows, "hybrid", "q003")
    metadata = row["response"]["metadata"]
    first_citation = row["response"]["output"]["citations"][0]

    assert row["failureReason"] == PASSED
    assert row["response"]["output"]["status"] == "answered"
    assert metadata["mode"] == "hybrid"
    assert metadata["context"][0]["chunk_id"] == first_citation["chunk_id"]
    assert metadata["llm_cache_hits"] == 0
    assert metadata["cold_start"] is True
    usage = row["tokenUsage"]
    assert usage["total"] == usage["prompt"] + usage["completion"]
