"""Question normalization and hash (Tech §11): table-driven."""

# ruff: noqa: RUF001  (the full-width characters are the test data)

from __future__ import annotations

import hashlib

import pytest

from grounded.infra.hashing import normalize_question, question_hash


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("How do I run a task?", "how do i run a task"),
        ("  How   do I\trun\na task?  ", "how do i run a task"),
        ("how do i run a task", "how do i run a task"),
        ("Why?!", "why"),
        ("Why ?", "why"),  # punctuation behind a space
        ("Why . . .", "why"),
        ("What is it…", "what is it"),
        ("什么是路径参数。", "什么是路径参数"),
        ("What is a path parameter？", "what is a path parameter"),  # full-width ? folds to ?
        ("ＦａｓｔＡＰＩ", "fastapi"),  # NFKC: full-width letters
        ("What is C#?", "what is c#"),  # only sentence punctuation goes: C# is not C
        ("What does foo() do?", "what does foo() do"),
        ("Use `Depends`.", "use `depends`"),
        ("?Why", "?why"),  # leading punctuation is kept
        ("What, exactly, is it?", "what, exactly, is it"),  # inner punctuation is kept
        ("", ""),
        ("?!", ""),
        ("   ", ""),
    ],
)
def test_normalize_question(raw: str, expected: str) -> None:
    assert normalize_question(raw) == expected


def test_normalization_is_idempotent() -> None:
    once = normalize_question("  Why   does it  fail?!  ")
    assert normalize_question(once) == once


def test_the_hash_is_sha256_of_the_normalized_question() -> None:
    expected = hashlib.sha256(b"how do i run a task").hexdigest()
    assert question_hash("How do I  run a task?") == expected
    assert question_hash("how do i run a task") == expected


def test_different_questions_have_different_hashes() -> None:
    assert question_hash("What is C#?") != question_hash("What is C?")
