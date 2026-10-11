"""The question rules of ``POST /v1/ask`` (5.04, Tech §12): trimmed first, 3 to 500 characters, no
control characters. The 4 KB body limit and the rate limit are in
``tests/integration/test_ask_protection.py``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from grounded.schemas.api import AskRequest


def question_of(text: str) -> str:
    return AskRequest(question=text).question


# --- the question: trimmed first, then 3-500 characters, no control characters -------------------


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("abc", "abc"),
        ("  abc  ", "abc"),  # the length rule applies to the trimmed text
        ("\n\tHow do I run a task?\r\n ", "How do I run a task?"),  # trailing newline is trimmed
        ("a" * 500, "a" * 500),
        ("  " + "a" * 500 + "  ", "a" * 500),
        ("what is  C#?", "what is  C#?"),  # inner spaces are the pipeline's business
    ],
)
def test_a_valid_question_is_trimmed_and_kept(text: str, kept: str) -> None:
    assert question_of(text) == kept


@pytest.mark.parametrize(
    "text", ["", "ab", "  ab  ", "     ", "a" * 501, " " + "a" * 501 + " "], ids=repr
)
def test_a_question_outside_3_to_500_characters_after_trimming_is_refused(text: str) -> None:
    with pytest.raises(ValidationError):
        AskRequest(question=text)


@pytest.mark.parametrize(
    "text",
    ["a\x00bc", "ab\x1bc[31m", "ab\ncd", "ab\tcd", "ab\x7fcd", "ab\x85cd", "how do I\r\nrun it?"],
    ids=repr,
)
def test_a_control_character_inside_the_question_is_refused(text: str) -> None:
    with pytest.raises(ValidationError, match="control characters") as caught:
        AskRequest(question=text)
    assert text not in str(caught.value)  # the message names the rule, not the input


def test_ordinary_unicode_is_not_a_control_character() -> None:
    assert question_of("Čemu služi Depends? 日本語 🙂") == "Čemu služi Depends? 日本語 🙂"
