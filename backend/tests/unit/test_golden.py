"""Golden-set rows (schemas/eval.py) and helpers (evals/golden.py): loading, sampling, section
listing and label resolution against corpus_mini's chunks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from grounded.evals.golden import (
    GOLDEN_DIR,
    GoldenSetError,
    IndexedChunk,
    load_golden_set,
    normalize_page,
    page_sections,
    resolve_labels,
    sample_sections,
    type_counts,
)
from grounded.ingest.pipeline import prepare_corpus
from grounded.ingest.types import Chunk, ChunkingConfig, CorpusCheckout
from grounded.schemas.eval import GoldenItem

CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"
CFG = ChunkingConfig(max_tokens=60, overlap_tokens=10, min_tokens=5, tokenizer="words")
BG = "docs/en/docs/tutorial/background-tasks.md"


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return prepare_corpus(CorpusCheckout(CORPUS_MINI, "0.0.1", "a" * 40), CFG, words).chunks


def row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": "q001",
        "question": "How do I run a function after returning a response?",
        "type": "how_to",
        "answerable": True,
        "reference_answer": "Use BackgroundTasks.",
        "relevant_sections": [
            {"section": f"{BG}#using-backgroundtasks", "grade": 2},
            {"section": f"{BG}#create-a-task-function", "grade": 1},
        ],
    }
    return base | overrides


def unanswerable(**overrides: Any) -> dict[str, Any]:
    return row(type="unanswerable", answerable=False, relevant_sections=[], **overrides)


# --- GoldenItem --------------------------------------------------------------------------------


def test_valid_items() -> None:
    item = GoldenItem.model_validate(row())
    assert item.relevant == {f"{BG}#using-backgroundtasks": 2, f"{BG}#create-a-task-function": 1}
    assert item.source_section is None
    assert item.notes == ""
    assert GoldenItem.model_validate(unanswerable()).relevant == {}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"answerable": False}, "answerable must be false exactly when"),
        ({"type": "unanswerable"}, "answerable must be false exactly when"),
        (
            {"type": "unanswerable", "answerable": False},
            "an unanswerable item has no relevant sections",
        ),
        (
            {"relevant_sections": [{"section": f"{BG}#using-backgroundtasks", "grade": 1}]},
            "at least one grade-2 section",
        ),
        ({"type": "multi_section"}, "at least two grade-2 sections"),
        (
            {
                "relevant_sections": [
                    {"section": f"{BG}#using-backgroundtasks", "grade": 2},
                    {"section": f"{BG}#using-backgroundtasks", "grade": 1},
                ]
            },
            "labeled twice",
        ),
        (
            {
                "relevant_sections": [
                    {"section": BG, "grade": 2},
                    {"section": f"{BG}#create-a-task-function", "grade": 1},
                ]
            },
            "nested labels",
        ),
        ({"id": "q1"}, "String should match pattern"),
        ({"question": "Hi"}, "at least 3 characters"),
        ({"question": "x" * 501}, "at most 500 characters"),
        ({"type": "opinion"}, "Input should be"),
        ({"reference_answer": ""}, "at least 1 character"),
        ({"source_section": "tutorial/x.md"}, "String should match pattern"),
        ({"extra": 1}, "Extra inputs are not permitted"),
    ],
)
def test_invalid_items(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        GoldenItem.model_validate(row(**overrides))


@pytest.mark.parametrize(
    "section",
    [
        "tutorial/background-tasks.md",  # missing docs/en/docs/
        f"{BG}#",  # empty anchor
        f"{BG}#a#b",
        "docs/en/docs/tutorial/background-tasks",  # not a .md page
        f"{BG}#has space",
    ],
)
def test_invalid_section_labels(section: str) -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        GoldenItem.model_validate(row(relevant_sections=[{"section": section, "grade": 2}]))


def test_grade_must_be_1_or_2() -> None:
    with pytest.raises(ValidationError):
        GoldenItem.model_validate(row(relevant_sections=[{"section": BG, "grade": 3}]))


def test_multi_section_with_two_grade_2_labels() -> None:
    sections = [
        {"section": f"{BG}#using-backgroundtasks", "grade": 2},
        {"section": "docs/en/docs/index.md", "grade": 2},
    ]
    item = GoldenItem.model_validate(row(type="multi_section", relevant_sections=sections))
    assert len(item.relevant) == 2


# --- load_golden_set ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict[str, Any] | str]) -> Path:
    lines = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_skips_blank_lines(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "g.jsonl", [row(), "", "   ", unanswerable(id="q002")])
    assert [item.id for item in load_golden_set(path)] == ["q001", "q002"]


def test_load_reports_every_problem_with_line_numbers(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "g.jsonl",
        [
            row(),
            row(id="q002", type="multi_section"),
            "{not json",
            row(),  # duplicate q001
        ],
    )
    with pytest.raises(GoldenSetError) as excinfo:
        load_golden_set(path)
    message = str(excinfo.value)
    assert "g.jsonl:2: item: Value error, a multi_section item needs" in message
    assert "g.jsonl:3: " in message
    assert "g.jsonl:4: duplicate id q001 (first on line 1)" in message


def test_load_rejects_an_empty_file(tmp_path: Path) -> None:
    with pytest.raises(GoldenSetError, match="no items"):
        load_golden_set(write_jsonl(tmp_path / "g.jsonl", [""]))


def test_type_counts_list_every_type() -> None:
    items = [GoldenItem.model_validate(row()), GoldenItem.model_validate(unanswerable(id="q002"))]
    assert type_counts(items) == {
        "factual": 0,
        "how_to": 1,
        "code": 0,
        "multi_section": 0,
        "unanswerable": 1,
    }


def test_committed_candidates_are_valid() -> None:
    items = load_golden_set(GOLDEN_DIR / "candidates.v1.jsonl")
    assert len(items) >= 45
    assert all(count > 0 for count in type_counts(items).values())


def test_golden_set_v1_is_the_approved_selection_of_candidates() -> None:
    golden = load_golden_set(GOLDEN_DIR / "golden_set.v1.jsonl")
    candidates = {item.id: item for item in load_golden_set(GOLDEN_DIR / "candidates.v1.jsonl")}
    assert len(golden) == 30
    assert all(item == candidates[item.id] for item in golden)
    assert type_counts(golden) == {
        "factual": 8,
        "how_to": 8,
        "code": 5,
        "multi_section": 4,
        "unanswerable": 5,
    }


def test_normalize_page() -> None:
    assert normalize_page("tutorial/x.md") == "docs/en/docs/tutorial/x.md"
    assert normalize_page("/tutorial/x.md") == "docs/en/docs/tutorial/x.md"
    assert normalize_page("tutorial\\x.md") == "docs/en/docs/tutorial/x.md"
    assert normalize_page("docs/en/docs/tutorial/x.md") == "docs/en/docs/tutorial/x.md"


# --- sampling and sections ---------------------------------------------------------------------


def test_sample_sections_is_seeded_and_skips_intros(chunks: list[Chunk]) -> None:
    sample = sample_sections(chunks, 3, seed=7)
    assert sample == sample_sections(chunks, 3, seed=7)
    assert sorted(sample) == [
        f"{BG}#create-a-task-function",
        f"{BG}#technical-details",
        f"{BG}#using-backgroundtasks",
    ]
    with pytest.raises(ValueError, match="asked for 4 sections, the corpus has 3"):
        sample_sections(chunks, 4, seed=7)


def test_page_sections_follow_document_order(chunks: list[Chunk]) -> None:
    sections = page_sections(chunks, BG)
    assert [info.section_id for info in sections] == [
        f"{BG}#",
        f"{BG}#using-backgroundtasks",
        f"{BG}#create-a-task-function",
        f"{BG}#technical-details",
    ]
    assert [info.heading_level for info in sections] == [1, 2, 2, 3]
    by_id = {info.section_id: info for info in sections}
    tech = by_id[f"{BG}#technical-details"]
    assert tech.breadcrumb_text.endswith(
        "Background Tasks > Create a task function > Technical Details"
    )
    assert tech.chunk_count == 1
    assert tech.token_count == 10
    assert page_sections(chunks, "docs/en/docs/nope.md") == []


# --- resolve_labels ----------------------------------------------------------------------------


def items_with(*sections: tuple[str, int]) -> list[GoldenItem]:
    relevant = [{"section": section, "grade": grade} for section, grade in sections]
    return [GoldenItem.model_validate(row(relevant_sections=relevant))]


def test_labels_that_resolve(chunks: list[Chunk]) -> None:
    items = items_with((f"{BG}#using-backgroundtasks", 2), ("docs/en/docs/index.md", 1))
    assert resolve_labels(items, chunks) == []
    assert resolve_labels([GoldenItem.model_validate(unanswerable())], chunks) == []


def test_unresolved_labels_say_why(chunks: list[Chunk]) -> None:
    items = items_with((f"{BG}#no-such-anchor", 2), ("docs/en/docs/missing.md#x", 1))
    problems = resolve_labels(items, chunks)
    assert problems == [
        f"q001: {BG}#no-such-anchor matches no chunk: the anchor is not a section of its own "
        "(typo, or merged into the previous sibling: label that one or the parent)",
        "q001: docs/en/docs/missing.md#x matches no chunk: no such page in the corpus",
    ]


def test_nested_h2_and_h3_labels_are_reported_once(chunks: list[Chunk]) -> None:
    items = items_with((f"{BG}#create-a-task-function", 2), (f"{BG}#technical-details", 1))
    assert resolve_labels(items, chunks) == [
        f"q001: {BG}#create-a-task-function and {BG}#technical-details all match chunk "
        f"{BG}#technical-details (nested labels: keep one)"
    ]


def test_resolution_works_on_indexed_chunks() -> None:
    indexed = [
        IndexedChunk(f"{BG}#technical-details", ("create-a-task-function", "technical-details"))
    ]
    assert resolve_labels(items_with((f"{BG}#create-a-task-function", 2)), indexed) == []
    assert resolve_labels(items_with((f"{BG}#using-backgroundtasks", 2)), indexed) != []
