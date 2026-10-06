from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from grounded.schemas.llm import LLMAnswer, LLMClaim


def _claim(**overrides: Any) -> dict[str, Any]:
    return {
        "text": "Use BackgroundTasks.",
        "citation_ids": ["c1"],
        "self_confidence": 0.8,
    } | overrides


def _answer(**overrides: Any) -> dict[str, Any]:
    return {
        "status": "answered",
        "answer_markdown": "Use BackgroundTasks [c1].",
        "claims": [_claim()],
    } | overrides


def test_valid_answer_defaults_follow_ups_to_empty() -> None:
    answer = LLMAnswer.model_validate(_answer())
    assert answer.follow_up_questions == []
    assert answer.claims[0].citation_ids == ["c1"]


def test_insufficient_context_may_have_no_claims() -> None:
    answer = LLMAnswer.model_validate(
        _answer(status="insufficient_context", answer_markdown="Not covered.", claims=[])
    )
    assert answer.claims == []


@pytest.mark.parametrize("label", ["c0", "c10", "C1", "c", "1", "c1 ", " c1", "x1", "c1,c2", ""])
def test_bad_citation_label_is_rejected(label: str) -> None:
    with pytest.raises(ValidationError):
        LLMClaim.model_validate(_claim(citation_ids=[label]))


@pytest.mark.parametrize("label", ["c1", "c5", "c9"])
def test_good_citation_label_is_accepted(label: str) -> None:
    assert LLMClaim.model_validate(_claim(citation_ids=[label])).citation_ids == [label]


def test_one_bad_label_among_good_ones_fails_the_claim() -> None:
    with pytest.raises(ValidationError):
        LLMClaim.model_validate(_claim(citation_ids=["c1", "source-2"]))


def test_citation_ids_limit_is_five() -> None:
    assert LLMClaim.model_validate(_claim(citation_ids=["c1", "c2", "c3", "c4", "c5"]))
    with pytest.raises(ValidationError):
        LLMClaim.model_validate(_claim(citation_ids=["c1", "c2", "c3", "c4", "c5", "c6"]))


def test_claim_without_citations_is_schema_valid() -> None:
    # "No valid citation" is a semantic rule handled in citations.py (Tech §9.5), not a schema one.
    assert LLMClaim.model_validate(_claim(citation_ids=[])).citation_ids == []


@pytest.mark.parametrize("text", ["", "x" * 501])
def test_claim_text_length_is_enforced(text: str) -> None:
    with pytest.raises(ValidationError):
        LLMClaim.model_validate(_claim(text=text))


def test_claim_text_at_the_limit_is_accepted() -> None:
    assert LLMClaim.model_validate(_claim(text="x" * 500))


@pytest.mark.parametrize("value", [-0.01, 1.01, 2, -1])
def test_self_confidence_out_of_range_is_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        LLMClaim.model_validate(_claim(self_confidence=value))


@pytest.mark.parametrize("value", [0, 0.0, 0.5, 1, 1.0])
def test_self_confidence_bounds_are_inclusive(value: float) -> None:
    assert LLMClaim.model_validate(_claim(self_confidence=value))


def test_claims_limit_is_eight() -> None:
    assert LLMAnswer.model_validate(_answer(claims=[_claim()] * 8))
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate(_answer(claims=[_claim()] * 9))


def test_follow_up_questions_limit_is_three() -> None:
    assert LLMAnswer.model_validate(_answer(follow_up_questions=["a?", "b?", "c?"]))
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate(_answer(follow_up_questions=["a?", "b?", "c?", "d?"]))


def test_answer_markdown_limit_is_4000() -> None:
    assert LLMAnswer.model_validate(_answer(answer_markdown="x" * 4000))
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate(_answer(answer_markdown="x" * 4001))


@pytest.mark.parametrize("status", ["ok", "ANSWERED", "", "refused"])
def test_unknown_status_is_rejected(status: str) -> None:
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate(_answer(status=status))


@pytest.mark.parametrize("missing", ["status", "answer_markdown", "claims"])
def test_required_fields(missing: str) -> None:
    data = _answer()
    del data[missing]
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate(data)


def test_self_confidence_is_required() -> None:
    data = _claim()
    del data["self_confidence"]
    with pytest.raises(ValidationError):
        LLMClaim.model_validate(data)


# --- Schema simplicity (Tech §9.1) ----------------------------------------------------------------
# Both providers' structured-output modes must accept the JSON schema. Pydantic puts the nested
# ``LLMClaim`` under ``$defs``, so a plain ``$ref`` is expected; what the providers can't handle is
# unions, nullable types and recursion.

_UNIONS = {"anyOf", "oneOf", "allOf", "not"}


def _nodes(node: object) -> list[dict[str, Any]]:
    """Every dict in the schema tree."""
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        typed: dict[str, Any] = node  # pyright: ignore[reportUnknownVariableType]
        found.append(typed)
        for value in typed.values():
            found.extend(_nodes(value))
    elif isinstance(node, list):
        for item in node:  # pyright: ignore[reportUnknownVariableType]
            found.extend(_nodes(item))
    return found


def _refs(node: object) -> set[str]:
    return {str(n["$ref"]).removeprefix("#/$defs/") for n in _nodes(node) if "$ref" in n}


def test_schema_has_no_unions() -> None:
    schema = LLMAnswer.model_json_schema()
    offenders = [key for n in _nodes(schema) for key in n if key in _UNIONS]
    assert offenders == []


def test_schema_types_are_plain_strings() -> None:
    # A list ("type": ["string", "null"]) is how nullable shows up.
    for n in _nodes(LLMAnswer.model_json_schema()):
        assert not isinstance(n.get("type"), list), n


def test_schema_enums_are_string_literals() -> None:
    enums = [n for n in _nodes(LLMAnswer.model_json_schema()) if "enum" in n]
    assert enums, "status should be an enum"
    for n in enums:
        assert n.get("type") == "string"
        assert all(isinstance(v, str) for v in n["enum"])


def test_schema_refs_are_local_and_not_recursive() -> None:
    schema = LLMAnswer.model_json_schema()
    defs: dict[str, Any] = schema.get("$defs", {})
    assert _refs(schema) <= set(defs), "every $ref points into this schema's $defs"

    graph = {name: _refs(body) for name, body in defs.items()}

    def reaches_itself(start: str) -> bool:
        stack, seen = list(graph[start]), set[str]()
        while stack:
            name = stack.pop()
            if name == start:
                return True
            if name not in seen:
                seen.add(name)
                stack.extend(graph[name])
        return False

    assert [name for name in graph if reaches_itself(name)] == []
