"""Test cases for the promptfoo run (evals/promptfoo_tests.py): built from the golden set, offline.

The committed ``golden_set.v1.jsonl`` is only read here. Cases that need a particular file shape
use a small one written to ``tmp_path``."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from grounded.evals import promptfoo_tests
from grounded.evals.golden import GOLDEN_DIR, load_golden_set
from grounded.evals.promptfoo_tests import (
    DEFAULT_GOLDEN_SET,
    PromptfooTestsError,
    build_tests,
    generate_tests,
    parse_question_ids,
)
from grounded.evals.retrieval_runner import RetrievalEvalError, golden_set_digest


def row(n: int, *, kind: str = "factual", answerable: bool = True) -> dict[str, Any]:
    return {
        "id": f"q{n:03d}",
        "question": f"Question number {n}?",
        "type": kind,
        "answerable": answerable,
        "reference_answer": f"Answer {n}.",
        "relevant_sections": (
            [
                {"section": "docs/en/docs/a.md#top", "grade": 2},
                {"section": "docs/en/docs/b.md", "grade": 1},
            ]
            if answerable
            else []
        ),
    }


@pytest.fixture
def golden_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    rows = [row(1), row(2, kind="unanswerable", answerable=False), row(3)]
    (tmp_path / "golden_set.v7.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(promptfoo_tests, "GOLDEN_DIR", tmp_path)
    return tmp_path


def test_one_case_per_item_in_file_order(golden_dir: Path) -> None:
    tests = build_tests(golden_set="golden_set.v7.jsonl")

    assert [t["metadata"]["golden"]["id"] for t in tests] == ["q001", "q002", "q003"]
    assert [t["description"] for t in tests] == [
        "q001 (factual)",
        "q002 (unanswerable)",
        "q003 (factual)",
    ]


def test_vars_hold_only_the_question(golden_dir: Path) -> None:
    # promptfoo prints every var as a column and sends them to the prompt: nothing else belongs.
    [first, *_] = build_tests(golden_set="golden_set.v7.jsonl")

    assert first["vars"] == {"question": "Question number 1?"}


def test_the_golden_payload_carries_what_the_assertions_and_the_gate_read(
    golden_dir: Path,
) -> None:
    [answerable, unanswerable, _] = build_tests(golden_set="golden_set.v7.jsonl")

    assert answerable["metadata"]["golden"] == {
        "id": "q001",
        "type": "factual",
        "answerable": True,
        "relevant": {"docs/en/docs/a.md#top": 2, "docs/en/docs/b.md": 1},
        "reference_answer": "Answer 1.",
        "golden_set_version": "v7",
        "golden_set_sha256": golden_set_digest(golden_dir / "golden_set.v7.jsonl"),
    }
    assert unanswerable["metadata"]["golden"]["answerable"] is False
    assert unanswerable["metadata"]["golden"]["relevant"] == {}


def test_the_payload_is_json_serializable(golden_dir: Path) -> None:
    # promptfoo receives the cases as JSON from the generator process.
    json.dumps(build_tests(golden_set="golden_set.v7.jsonl"))


def test_ids_select_a_subset_in_file_order_not_in_the_order_given(golden_dir: Path) -> None:
    tests = build_tests(golden_set="golden_set.v7.jsonl", ids=["q003", "q001"])

    assert [t["metadata"]["golden"]["id"] for t in tests] == ["q001", "q003"]


def test_an_unknown_id_is_an_error_not_a_smaller_run(golden_dir: Path) -> None:
    with pytest.raises(PromptfooTestsError, match="q099"):
        build_tests(golden_set="golden_set.v7.jsonl", ids=["q001", "q099"])


@pytest.mark.parametrize("name", ["../golden_set.v7.jsonl", "sub/golden_set.v7.jsonl"])
def test_the_golden_set_is_a_file_name_not_a_path(golden_dir: Path, name: str) -> None:
    with pytest.raises(PromptfooTestsError, match="file name"):
        build_tests(golden_set=name)


def test_an_unversioned_golden_set_name_is_refused(golden_dir: Path) -> None:
    (golden_dir / "mine.jsonl").write_text(json.dumps(row(1)) + "\n", encoding="utf-8")

    with pytest.raises(RetrievalEvalError, match=r"golden_set\.v<N>\.jsonl"):
        build_tests(golden_set="mine.jsonl")


def test_the_generator_reads_the_config_the_yaml_passes(golden_dir: Path) -> None:
    tests = generate_tests({"golden_set": "golden_set.v7.jsonl", "ids": ["q002"]})

    assert [t["metadata"]["golden"]["id"] for t in tests] == ["q002"]


def test_the_generator_runs_the_default_set_without_a_config() -> None:
    assert len(generate_tests()) == len(generate_tests(None)) == len(generate_tests({}))


# --- the committed set -------------------------------------------------------------------------


def test_the_committed_golden_set_becomes_one_case_per_item() -> None:
    items = load_golden_set(GOLDEN_DIR / DEFAULT_GOLDEN_SET)

    tests = build_tests()

    assert [t["metadata"]["golden"]["id"] for t in tests] == [i.id for i in items]
    assert [t["vars"]["question"] for t in tests] == [i.question for i in items]
    digest = golden_set_digest(GOLDEN_DIR / DEFAULT_GOLDEN_SET)
    assert {t["metadata"]["golden"]["golden_set_sha256"] for t in tests} == {digest}
    assert {t["metadata"]["golden"]["golden_set_version"] for t in tests} == {"v1"}
    unanswerable = [t for t in tests if not t["metadata"]["golden"]["answerable"]]
    assert unanswerable
    assert all(t["metadata"]["golden"]["type"] == "unanswerable" for t in unanswerable)


# --- EVAL_QUESTION_IDS -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("q003,q045", ["q003", "q045"]),
        ("q003, q045  q012", ["q003", "q045", "q012"]),
        (" q003 ", ["q003"]),
        ("", None),
        ("  ,  ", None),
        (None, None),
    ],
)
def test_parse_question_ids(text: str | None, expected: list[str] | None) -> None:
    assert parse_question_ids(text) == expected


def test_every_case_of_a_call_carries_the_same_fresh_run_id(golden_dir: Path) -> None:
    first = build_tests(golden_set="golden_set.v7.jsonl")
    second = build_tests(golden_set="golden_set.v7.jsonl")

    ids = {t["metadata"]["run_id"] for t in first}
    assert len(ids) == 1
    assert ids != {t["metadata"]["run_id"] for t in second}  # a new call is a new run
    assert re.fullmatch(r"[0-9a-f]{32}", ids.pop())


def test_a_given_run_id_is_used_as_it_is(golden_dir: Path) -> None:
    tests = build_tests(golden_set="golden_set.v7.jsonl", run_id="fixed")

    assert {t["metadata"]["run_id"] for t in tests} == {"fixed"}
