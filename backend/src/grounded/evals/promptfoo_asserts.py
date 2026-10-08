"""The deterministic promptfoo assertions (Tech §15.3, 4.05).

``eval/promptfoo/asserts.py`` re-exports these four functions; the config names them
(``value: file://asserts.py:<function>`` with ``metric: <function>``). promptfoo starts a new Python
process for **every** assertion call, so this module imports only the standard library at the top
and ``metrics.section_matches`` (itself standard-library only) inside the one function that needs
it: no pipeline, no settings, no database, no pydantic.

Contract of promptfoo 0.123.1, as verified in the 4.05 run: ``fn(output, context)`` where ``output``
is the provider's ``output`` (the ``AskResponse`` dict) and ``context`` has ``vars``, ``test`` and
``providerResponse``. The case's golden payload is ``context["test"]["metadata"]["golden"]``
(``promptfoo_tests.golden_case``); what the provider recorded is ``providerResponse.metadata``
(``promptfoo_provider``). The return value is ``{pass, score, reason, not_applicable}``:

- ``pass`` marks a perfect score (promptfoo's table); the metric itself is ``score``. The gate
  aggregates scores (4.08), not promptfoo's pass rate.
- **N/A convention (shared with 4.06's faithfulness).** A metric that does not apply to a case
  returns ``pass=True, score=1.0`` (so it never fails promptfoo's own check), ``reason`` starting
  with ``"N/A: "`` and ``not_applicable=True``. An aggregation must skip every component result with
  ``not_applicable`` true: its ``score`` is a placeholder, not a measurement. promptfoo's own
  per-metric averages (``namedScores``) include those placeholders and must not be read as results.
- Input that is missing or ill-formed fails the assertion with a ``malformed input`` reason instead
  of raising, so one bad row is visible and does not stop the others.

Which cases a metric skips: citation validity and citation precision are N/A in ``no_rag`` (there
are no sources to cite); citation precision is also N/A for an unanswerable item (it has no labelled
section to match) and for an answer with no citation (a refusal).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

type Result = dict[str, Any]
type Assertion = Callable[[Any, Mapping[str, Any]], Result]

NO_RAG: Final = "no_rag"
INSUFFICIENT_CONTEXT: Final = "insufficient_context"


class _MalformedError(Exception):
    """The input is not what the provider and the test generator write; ``args[0]`` says what."""


def _result(passed: bool, score: float, reason: str) -> Result:
    return {"pass": passed, "score": score, "reason": reason, "not_applicable": False}


def _not_applicable(reason: str) -> Result:
    return {"pass": True, "score": 1.0, "reason": f"N/A: {reason}", "not_applicable": True}


def _guarded(check: Callable[[Any, Mapping[str, Any]], Result]) -> Assertion:
    @functools.wraps(check)
    def run(output: Any, context: Mapping[str, Any]) -> Result:
        try:
            return check(output, context)
        except _MalformedError as exc:
            return _result(False, 0.0, f"malformed input: {exc}")

    return run


# --- Reading the inputs ------------------------------------------------------------------------


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _MalformedError(f"{what} is not an object")
    return value  # pyright: ignore[reportUnknownVariableType]


def _provider_metadata(context: Mapping[str, Any]) -> Mapping[str, Any]:
    response = context.get("providerResponse")
    metadata = _mapping(response, "providerResponse").get("metadata") if response else None
    return _mapping(metadata if metadata is not None else context.get("metadata"), "metadata")


def _golden(context: Mapping[str, Any]) -> Mapping[str, Any]:
    test = _mapping(context.get("test"), "test")
    metadata = _mapping(test.get("metadata"), "test metadata")
    return _mapping(metadata.get("golden"), "golden payload")


def _answer(output: Any) -> Mapping[str, Any]:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError as exc:
            raise _MalformedError("output is not JSON") from exc
    return _mapping(output, "output")


def _int(source: Mapping[str, Any], key: str) -> int:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _MalformedError(f"{key} is not an integer")
    return value


def _bool(source: Mapping[str, Any], key: str) -> bool:
    value = source.get(key)
    if not isinstance(value, bool):
        raise _MalformedError(f"{key} is not a boolean")
    return value


def _text(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str):
        raise _MalformedError(f"{key} is not a string")
    return value


def _is_no_rag(metadata: Mapping[str, Any]) -> bool:
    return _text(metadata, "mode") == NO_RAG


# --- The assertions ----------------------------------------------------------------------------


@_guarded
def schema_first_try(output: Any, context: Mapping[str, Any]) -> Result:
    """The model's answer was valid on the first attempt: ``validation_retries == 0``."""
    retries = _int(_provider_metadata(context), "validation_retries")
    if retries == 0:
        return _result(True, 1.0, "valid on the first attempt")
    return _result(False, 0.0, f"{retries} validation retr{'y' if retries == 1 else 'ies'}")


@_guarded
def citation_validity(output: Any, context: Mapping[str, Any]) -> Result:
    """No invented label: the pre-filter ``invalid_citation_count`` is 0 (Tech §9.6)."""
    metadata = _provider_metadata(context)
    if _is_no_rag(metadata):
        return _not_applicable("no_rag has no sources to cite")
    invalid = _int(metadata, "invalid_citation_count")
    if invalid == 0:
        return _result(True, 1.0, "every citation label was valid")
    return _result(False, 0.0, f"{invalid} invalid citation reference(s) removed")


@_guarded
def refusal_correctness(output: Any, context: Mapping[str, Any]) -> Result:
    """The model refused exactly when the question is unanswerable.

    Answerable and ``status != insufficient_context``, or unanswerable and ``status ==
    insufficient_context``, passes; anything else fails. ``partial`` counts as an answer.
    """
    answerable = _bool(_golden(context), "answerable")
    status = _text(_answer(output), "status")
    refused = status == INSUFFICIENT_CONTEXT
    if answerable != refused:
        return _result(True, 1.0, f"{'answerable' if answerable else 'unanswerable'}, {status}")
    if answerable:
        return _result(False, 0.0, "answerable question, but the model refused")
    return _result(False, 0.0, f"unanswerable question, but the model answered ({status})")


@_guarded
def citation_precision(output: Any, context: Mapping[str, Any]) -> Result:
    """The share of cited chunks that match a labelled section of grade >= 1 (Tech §15.1 rule).

    A cited chunk is a distinct entry of the response's ``citations``; its section comes from the
    chunk the provider recorded under that ``chunk_id``. The matching rule is the retrieval
    metrics' own (``metrics.section_matches``).
    """
    from grounded.evals.metrics import section_matches

    metadata = _provider_metadata(context)
    golden = _golden(context)
    if _is_no_rag(metadata):
        return _not_applicable("no_rag has no sources to cite")
    if not _bool(golden, "answerable"):
        return _not_applicable("an unanswerable question has no labelled section")
    cited = _cited_chunk_ids(_answer(output))
    if not cited:
        return _not_applicable("the answer cites nothing")

    labels = [
        label
        for label, grade in _mapping(golden.get("relevant"), "relevant").items()
        if isinstance(grade, int) and grade >= 1
    ]
    chunks = _context_chunks(metadata)
    matched = 0
    for chunk_id in cited:
        chunk = chunks.get(chunk_id)
        if chunk is None:
            raise _MalformedError(f"cited chunk {chunk_id} is not in the recorded context")
        source_path = _text(chunk, "section_id").partition("#")[0]
        anchors = _strings(chunk.get("anchor_path"), "anchor_path")
        matched += any(section_matches(label, source_path, anchors) for label in labels)
    return _result(
        matched == len(cited),
        matched / len(cited),
        f"{matched} of {len(cited)} cited chunk(s) match a labelled section",
    )


def _cited_chunk_ids(answer: Mapping[str, Any]) -> list[int]:
    citations = answer.get("citations")
    if not isinstance(citations, Sequence) or isinstance(citations, str):
        raise _MalformedError("citations is not a list")
    ids: list[int] = []
    for citation in citations:  # pyright: ignore[reportUnknownVariableType]
        chunk_id = _int(_mapping(citation, "a citation"), "chunk_id")
        if chunk_id not in ids:  # one chunk is one citation; a repeat must not count twice
            ids.append(chunk_id)
    return ids


def _context_chunks(metadata: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    entries = metadata.get("context")
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        raise _MalformedError("context is not a list")
    chunks: dict[int, Mapping[str, Any]] = {}
    for entry in entries:  # pyright: ignore[reportUnknownVariableType]
        chunk = _mapping(entry, "a context entry")
        chunks[_int(chunk, "chunk_id")] = chunk
    return chunks


def _strings(value: Any, what: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise _MalformedError(f"{what} is not a list")
    items: list[str] = []
    for item in value:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(item, str):
            raise _MalformedError(f"{what} holds a non-string")
        items.append(item)
    return items
