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
section to match) and for an answer with no citation (a refusal). Faithfulness is N/A in ``no_rag``,
for a refusal (``insufficient_context``) and for an answer with no claim.

**The two judge assertions** (``faithfulness``, ``correctness``, 4.06) decide N/A and parse their
inputs here, with the standard library only, and import ``promptfoo_judge`` (the judge, pydantic,
the Groq adapter: about a second) only for a case that needs a judge call. Their result adds
``errored`` and ``judge`` to the keys above (see ``promptfoo_judge``): a case the judge could not
grade is **errored**, which is neither a score nor not-applicable. ``judge_scored``,
``judge_errored`` and ``judge_detail`` build those shapes, here so that malformed input, which needs
no judge, is reported in the same one.
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


def _judge_not_applicable(reason: str) -> Result:
    """N/A for a judge metric: the shared convention, plus ``errored: false`` (every judge
    component says it; ``judge`` is absent because no judge was asked)."""
    return {**_not_applicable(reason), "errored": False}


def judge_detail(
    metric: str,
    *,
    prompt_version: str | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """The ``judge`` object of a judge component, before the fields that depend on the metric.
    The three optional fields are ``None`` when no judge call was made."""
    return {
        "metric": metric,
        "prompt_version": prompt_version,
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "usage": {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0},
    }


def judge_scored(passed: bool, score: float, reason: str, detail: dict[str, Any]) -> Result:
    """A judge component with a score. ``pass`` marks a perfect one."""
    return {
        "pass": passed,
        "score": score,
        "reason": reason,
        "not_applicable": False,
        "errored": False,
        "judge": {**detail, "error": None},
    }


def judge_errored(detail: dict[str, Any], error: Mapping[str, Any]) -> Result:
    """A judge component for a case the judge could not grade: no score, and not N/A.

    ``score`` 0.0 and ``pass`` false are placeholders so that promptfoo shows the row as failed
    (the way a not-applicable result carries 1.0 so that it never fails); an aggregation reads
    ``errored`` and ``judge.error`` (``kind``, ``is_quota``, ``provider_side``, ``detail``).
    """
    return {
        "pass": False,
        "score": 0.0,
        "reason": f"judge error ({error['kind']}): {error['detail']}",
        "not_applicable": False,
        "errored": True,
        "judge": {**detail, "error": dict(error)},
    }


def _guarded(check: Callable[[Any, Mapping[str, Any]], Result]) -> Assertion:
    @functools.wraps(check)
    def run(output: Any, context: Mapping[str, Any]) -> Result:
        try:
            return check(output, context)
        except _MalformedError as exc:
            return _result(False, 0.0, f"malformed input: {exc}")

    return run


def _guarded_judge(
    metric: str,
) -> Callable[[Callable[[Any, Mapping[str, Any]], Result]], Assertion]:
    """``_guarded`` for a judge metric: malformed input is an *errored* component, not a score of 0
    (the case was not judged), with the kind ``MalformedInput``."""

    def decorate(check: Callable[[Any, Mapping[str, Any]], Result]) -> Assertion:
        @functools.wraps(check)
        def run(output: Any, context: Mapping[str, Any]) -> Result:
            try:
                return check(output, context)
            except _MalformedError as exc:
                error = {
                    "kind": "MalformedInput",
                    "is_quota": False,
                    "provider_side": False,
                    "detail": f"malformed input: {exc}",
                }
                return judge_errored(judge_detail(metric), error)

        return run

    return decorate


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


def _float(source: Mapping[str, Any], key: str) -> float:
    value = source.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _MalformedError(f"{key} is not a number")
    return float(value)


def _is_no_rag(metadata: Mapping[str, Any]) -> bool:
    return _text(metadata, "mode") == NO_RAG


def _run_id(context: Mapping[str, Any]) -> str:
    """The id of this run, which ``promptfoo_tests.build_tests`` put in every test's metadata."""
    test = _mapping(context.get("test"), "test")
    return _text(_mapping(test.get("metadata"), "test metadata"), "run_id")


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


@_guarded_judge("faithfulness")
def faithfulness(output: Any, context: Mapping[str, Any]) -> Result:
    """The share of the answer's claims that the sources they cite support (judge, per claim).

    One judge call per claim, over the chunks that claim cites (``metadata.claims[].chunk_ids``
    resolved through ``metadata.context``). A claim with no valid source is ``NOT_SUPPORTED``
    without a call. N/A: ``no_rag`` (no sources), a refusal and an answer with no claim: such a
    case is excluded from the mean, never 0 and never 1. An errored case has no score.
    """
    metadata = _provider_metadata(context)
    if _is_no_rag(metadata):
        return _judge_not_applicable("no_rag has no sources to judge a claim against")
    if _text(_answer(output), "status") == INSUFFICIENT_CONTEXT:
        return _judge_not_applicable("the model refused, so there is no claim to judge")
    entries = _items(metadata.get("claims"), "claims")
    if not entries:
        return _judge_not_applicable("the answer has no claims")
    run_id = _run_id(context)

    chunks = _context_chunks(metadata)
    contents = {_text(chunk, "label"): _text(chunk, "content") for chunk in chunks.values()}
    parsed: list[tuple[str, float, list[str]]] = []
    for entry in entries:
        claim = _mapping(entry, "a claim")
        labels: list[str] = []
        for chunk_id in _items(claim.get("chunk_ids"), "chunk_ids"):
            chunk = chunks.get(chunk_id) if isinstance(chunk_id, int) else None
            if chunk is None:
                raise _MalformedError(f"cited chunk {chunk_id!r} is not in the recorded context")
            labels.append(_text(chunk, "label"))
        parsed.append((_text(claim, "text"), _float(claim, "confidence"), labels))

    from grounded.evals.judge import cited_sources
    from grounded.evals.promptfoo_judge import ClaimInput, judge_faithfulness

    claims = [
        ClaimInput(text, confidence, cited_sources(labels, contents))
        for text, confidence, labels in parsed
    ]
    return judge_faithfulness(claims, run_id=run_id)


@_guarded_judge("correctness")
def correctness(output: Any, context: Mapping[str, Any]) -> Result:
    """The answer against the golden ``reference_answer``: 1 / 0.5 / 0 (judge, one call).

    Never N/A: it is the one generation metric that compares ``no_rag`` with ``hybrid``, and a
    refusal of an unanswerable question is graded against its reference answer like any other
    (the rubric has a rule for it). The candidate is the answer text, citation markers included
    (rubric rule 2 ignores them).
    """
    reference = _text(_golden(context), "reference_answer")
    question = _text(_mapping(context.get("vars"), "vars"), "question")
    answer = _text(_answer(output), "answer_markdown")
    run_id = _run_id(context)

    from grounded.evals.promptfoo_judge import judge_correctness

    return judge_correctness(
        question=question, reference_answer=reference, answer=answer, run_id=run_id
    )


def _items(value: Any, what: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise _MalformedError(f"{what} is not a list")
    return list(value)  # pyright: ignore[reportUnknownArgumentType]


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
