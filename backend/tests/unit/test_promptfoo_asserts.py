"""The deterministic promptfoo assertions (evals/promptfoo_asserts.py): pure functions on the
``(output, context)`` pair promptfoo hands them. No promptfoo, no database, no network."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from grounded.evals import promptfoo_asserts as asserts

PAGE = "docs/en/docs/tutorial/background-tasks.md"
OTHER = "docs/en/docs/tutorial/other.md"


def chunk(chunk_id: int, source: str, *anchors: str) -> dict[str, Any]:
    return {
        "label": f"c{chunk_id}",
        "chunk_id": chunk_id,
        "section_id": f"{source}#{anchors[-1] if anchors else ''}",
        "anchor_path": list(anchors),
        "content": "text",
    }


# 1 and 2 match the labels below; 3 is another page; 4 is a sibling section on the labelled page.
CONTEXT = [
    chunk(1, PAGE, "using-backgroundtasks"),
    chunk(2, PAGE, "create-a-task-function"),
    chunk(3, OTHER, "elsewhere"),
    chunk(4, PAGE, "dependency-injection"),
]


def golden(*, answerable: bool = True) -> dict[str, Any]:
    relevant = (
        {f"{PAGE}#using-backgroundtasks": 2, f"{PAGE}#create-a-task-function": 1}
        if answerable
        else {}
    )
    return {"id": "q017", "type": "how_to", "answerable": answerable, "relevant": relevant}


def answer(status: str = "answered", cited: tuple[int, ...] = (1,)) -> dict[str, Any]:
    return {
        "status": status,
        "citations": [{"n": n, "chunk_id": chunk_id} for n, chunk_id in enumerate(cited, start=1)],
    }


def case(
    *,
    mode: str = "hybrid",
    retries: int = 0,
    invalid: int | None = 0,
    answerable: bool = True,
    context: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The context promptfoo passes: the provider's metadata and the test's golden payload."""
    metadata = {
        "mode": mode,
        "validation_retries": retries,
        "invalid_citation_count": invalid,
        "context": CONTEXT if context is None else context,
    }
    return {
        "vars": {"question": "q"},
        "test": {"metadata": {"golden": golden(answerable=answerable)}},
        "providerResponse": {"output": {}, "metadata": metadata},
        "metadata": metadata,
    }


def is_na(result: dict[str, Any]) -> bool:
    return bool(result["not_applicable"]) and result["reason"].startswith("N/A: ")


# --- shape -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "check",
    [
        asserts.schema_first_try,
        asserts.citation_validity,
        asserts.refusal_correctness,
        asserts.citation_precision,
    ],
)
def test_every_result_has_the_four_keys_and_a_score_in_range(check: Any) -> None:
    result = check(answer(), case())

    assert set(result) == {"pass", "score", "reason", "not_applicable"}
    assert isinstance(result["pass"], bool)
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["reason"], str)


def test_a_not_applicable_result_cannot_fail_promptfoo_and_says_so() -> None:
    result = asserts.citation_validity(answer(), case(mode="no_rag"))

    assert (result["pass"], result["score"]) == (True, 1.0)
    assert is_na(result)


# --- schema first try --------------------------------------------------------------------------


def test_schema_first_try_passes_without_a_retry() -> None:
    result = asserts.schema_first_try(answer(), case(retries=0))

    assert (result["pass"], result["score"], result["not_applicable"]) == (True, 1.0, False)


def test_schema_first_try_fails_after_a_retry() -> None:
    result = asserts.schema_first_try(answer(), case(retries=1))

    assert (result["pass"], result["score"]) == (False, 0.0)
    assert "1 validation retry" in result["reason"]


def test_schema_first_try_applies_to_no_rag_too() -> None:
    assert not asserts.schema_first_try(answer(), case(mode="no_rag", retries=1))["pass"]
    assert asserts.schema_first_try(answer(), case(mode="no_rag", retries=0))["pass"]


# --- citation validity -------------------------------------------------------------------------


def test_citation_validity_passes_with_no_invalid_reference() -> None:
    assert asserts.citation_validity(answer(), case(invalid=0))["score"] == 1.0


def test_citation_validity_fails_on_any_invalid_reference() -> None:
    result = asserts.citation_validity(answer(), case(invalid=2))

    assert (result["pass"], result["score"]) == (False, 0.0)
    assert "2 invalid" in result["reason"]


def test_citation_validity_is_not_applicable_in_no_rag() -> None:
    assert is_na(asserts.citation_validity(answer(), case(mode="no_rag", invalid=None)))


def test_citation_validity_still_applies_to_a_refusal() -> None:
    result = asserts.citation_validity(answer("insufficient_context", ()), case(invalid=1))

    assert not result["not_applicable"]
    assert not result["pass"]


# --- refusal correctness -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answerable", "status", "expected"),
    [
        (True, "answered", True),
        (True, "partial", True),
        (True, "insufficient_context", False),
        (False, "insufficient_context", True),
        (False, "answered", False),
        (False, "partial", False),
    ],
)
def test_refusal_correctness_is_answerable_iff_not_refused(
    answerable: bool, status: str, expected: bool
) -> None:
    result = asserts.refusal_correctness(answer(status), case(answerable=answerable))

    assert (result["pass"], result["score"]) == (expected, 1.0 if expected else 0.0)
    assert result["not_applicable"] is False


def test_refusal_correctness_applies_to_no_rag() -> None:
    result = asserts.refusal_correctness(answer("answered"), case(mode="no_rag"))

    assert result["pass"] is True


def test_refusal_correctness_reads_a_json_string_output_too() -> None:
    result = asserts.refusal_correctness(json.dumps(answer("answered")), case())

    assert result["pass"] is True


# --- citation precision ------------------------------------------------------------------------


def test_citation_precision_is_the_share_of_cited_chunks_that_match_a_label() -> None:
    result = asserts.citation_precision(answer(cited=(1, 3)), case())

    assert result["score"] == 0.5
    assert result["pass"] is False
    assert result["not_applicable"] is False
    assert "1 of 2" in result["reason"]


def test_citation_precision_passes_only_when_every_cited_chunk_matches() -> None:
    result = asserts.citation_precision(answer(cited=(1, 2)), case())

    assert (result["pass"], result["score"]) == (True, 1.0)


def test_citation_precision_counts_a_grade_1_label() -> None:
    result = asserts.citation_precision(answer(cited=(2,)), case())

    assert result["score"] == 1.0  # chunk 2 matches only the grade-1 label


def test_citation_precision_scores_zero_when_nothing_cited_matches() -> None:
    result = asserts.citation_precision(answer(cited=(3, 4)), case())

    assert (result["pass"], result["score"], result["not_applicable"]) == (False, 0.0, False)


def test_citation_precision_counts_a_chunk_cited_twice_once() -> None:
    twice = {"status": "answered", "citations": [{"chunk_id": 1}, {"chunk_id": 1}, {"chunk_id": 3}]}

    assert asserts.citation_precision(twice, case())["score"] == 0.5


def test_citation_precision_matches_an_h3_chunk_under_an_h2_label() -> None:
    nested = [chunk(1, PAGE, "using-backgroundtasks", "deeper")]

    result = asserts.citation_precision(answer(cited=(1,)), case(context=nested))

    assert result["score"] == 1.0


def test_citation_precision_matches_a_whole_page_label() -> None:
    context = case()
    context["test"]["metadata"]["golden"]["relevant"] = {PAGE: 2}

    assert asserts.citation_precision(answer(cited=(1, 4, 3)), context)["score"] == pytest.approx(
        2 / 3
    )


@pytest.mark.parametrize(
    ("output", "context", "why"),
    [
        (answer(), case(mode="no_rag"), "no_rag"),
        (answer("insufficient_context", ()), case(), "cites nothing"),
        (answer("answered", ()), case(), "cites nothing"),
        (answer(cited=(3,)), case(answerable=False), "unanswerable"),
    ],
)
def test_citation_precision_is_not_applicable(
    output: dict[str, Any], context: dict[str, Any], why: str
) -> None:
    result = asserts.citation_precision(output, context)

    assert is_na(result)
    assert why in result["reason"]


# --- malformed input ---------------------------------------------------------------------------


def malformed(result: dict[str, Any]) -> bool:
    return (
        result["pass"] is False
        and result["score"] == 0.0
        and result["reason"].startswith("malformed input: ")
    )


def test_malformed_input_fails_the_assertion_instead_of_raising() -> None:
    assert malformed(asserts.schema_first_try(answer(), {}))
    assert malformed(asserts.schema_first_try(answer(), case(retries="1")))  # type: ignore[arg-type]
    assert malformed(asserts.citation_validity(answer(), case(invalid=None)))
    assert malformed(asserts.refusal_correctness("not json", case()))
    assert malformed(asserts.refusal_correctness([], case()))
    assert malformed(asserts.refusal_correctness({"status": 1}, case()))
    no_golden = case()
    no_golden["test"] = {"metadata": {}}
    assert malformed(asserts.refusal_correctness(answer(), no_golden))
    assert malformed(asserts.citation_precision({"status": "answered"}, case()))
    assert malformed(asserts.citation_precision(answer(cited=(99,)), case()))  # not in context
    assert malformed(asserts.citation_precision(answer(), case(context=[{"chunk_id": 1}])))


def test_a_bool_is_not_an_integer_count() -> None:
    assert malformed(asserts.schema_first_try(answer(), case(retries=True)))  # type: ignore[arg-type]


def test_metadata_falls_back_to_the_context_metadata_key() -> None:
    context = case()
    del context["providerResponse"]

    assert asserts.schema_first_try(answer(), context)["pass"] is True


# --- weight ------------------------------------------------------------------------------------


def test_importing_the_assertions_does_not_load_the_pipeline_or_settings() -> None:
    """Every assertion call is a new Python process (Tech §15.3): keep its start cheap."""
    probe = (
        "import sys; import grounded.evals.promptfoo_asserts as a; "
        "a.citation_precision({'status': 'answered', 'citations': []}, "
        "{'providerResponse': {'metadata': {'mode': 'hybrid'}}, "
        "'test': {'metadata': {'golden': {'answerable': True, 'relevant': {}}}}}); "
        "heavy = [m for m in ('pydantic', 'psycopg', 'google', 'grounded.settings', "
        "'grounded.generation.pipeline') if m in sys.modules]; "
        "print(','.join(heavy))"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=60
    )

    assert done.stdout.strip() == ""
