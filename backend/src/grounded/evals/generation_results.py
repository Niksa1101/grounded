"""Read promptfoo's JSON output (``promptfoo eval -o <file>.json``) into a ``GenerationRun``.

The reading rules are Tech.md §15.3 ("As built in 4.05" and "4.06"), verified on two recorded
outputs (``tests/fixtures/promptfoo/``). The parser records and decides nothing about gating:

- A row with ``failureReason`` 2 is a failed generator call: its tagged error (``error_kind``,
  ``is_quota``, ``skipped`` from the row metadata) becomes the case's ``error``, no metric exists.
  Anything else is graded, and every assertion becomes a metric keyed by ``assertion.metric``.
- A component is *not applicable* (``not_applicable`` true), *errored* (a judge's ``errored``,
  whose ``judge.error`` has the tag; a deterministic assertion that reported ``malformed input``;
  an assertion that raised, which has none of our keys) or *scored*. The ``score`` of the first
  two is a placeholder and is dropped. promptfoo's ``namedScores`` and pass rate are never read:
  they count the placeholders.
- The golden-set digest is read from ``results[].metadata.golden``: promptfoo redacts the copy in
  ``testCase``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from grounded.evals.retrieval_runner import EVAL_DIR
from grounded.schemas.eval import GoldenType
from grounded.schemas.generation_eval import (
    SHA256_PATTERN,
    ErrorInfo,
    ErrorStage,
    GenerationBaselineEntry,
    GenerationCase,
    GenerationConfigResult,
    GenerationRun,
    GenerationRunInfo,
    MetricResult,
)

GENERATION_BASELINE: Final = EVAL_DIR / "baselines" / "generation.json"

_BASELINE_FILE: Final = TypeAdapter(dict[str, GenerationBaselineEntry])
_TAG: Final = re.compile(r"^\[(?P<kind>\w+) quota=(?P<quota>true|false)\] ?")
_PASSED, _FAILED, _ERRORED = 0, 1, 2  # promptfoo's failureReason
_MAX_DETAIL: Final = 500


class PromptfooResultsError(ValueError):
    """The file is not a promptfoo results file this gate can read; the message says why."""


# --- promptfoo's shape, read loosely (unknown keys ignored) --------------------------------------


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Golden(_Wire):
    id: str
    type: GoldenType
    golden_set_version: str
    golden_set_sha256: str = Field(pattern=SHA256_PATTERN)  # "[REDACTED]" is the testCase copy


class _RowMetadata(_Wire):
    """The test's ``golden`` payload merged with the provider's metadata (Tech §15.3)."""

    golden: _Golden
    cold_start: bool = False
    llm_cache_hits: int = 0
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    index_version: str | None = None
    retrieval_config_hash: str | None = None
    latency_ms: dict[str, float] | None = None  # success rows: total, embed, retrieval, llm
    error_kind: str | None = None  # error rows
    error_bases: tuple[str, ...] = ()
    is_quota: bool = False
    skipped: bool = False


class _JudgeError(_Wire):
    kind: str
    is_quota: bool = False
    provider_side: bool | None = None
    detail: str = ""


class _Judge(_Wire):
    prompt_version: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    n_claims: int | None = None
    n_supported: int | None = None
    error: _JudgeError | None = None


class _Assertion(_Wire):
    metric: str


class _Component(_Wire):
    score: float
    reason: str = ""
    not_applicable: bool | None = None  # absent: the assertion raised, promptfoo wrote the result
    errored: bool = False  # judge components only
    assertion: _Assertion
    judge: _Judge | None = None


class _Grading(_Wire):
    component_results: list[_Component] = Field(alias="componentResults")


class _TokenUsage(_Wire):
    prompt: int = 0
    completion: int = 0


class _Provider(_Wire):
    label: str


class _Row(_Wire):
    provider: _Provider
    error: str | None = None
    cost: float | None = None
    failure_reason: int = Field(alias="failureReason")
    grading_result: _Grading | None = Field(default=None, alias="gradingResult")
    token_usage: _TokenUsage | None = Field(default=None, alias="tokenUsage")
    metadata: _RowMetadata


class _ResultsBody(_Wire):
    timestamp: datetime
    results: list[_Row]


class _DocumentMetadata(_Wire):
    promptfoo_version: str = Field(alias="promptfooVersion")


class _Document(_Wire):
    results: _ResultsBody
    metadata: _DocumentMetadata


# --- Parsing -------------------------------------------------------------------------------------


def parse_results(
    raw: bytes | str, *, git_sha: str | None = None, git_dirty: bool | None = None
) -> GenerationRun:
    """promptfoo's JSON output as a ``GenerationRun``. Raises ``PromptfooResultsError`` (a
    ``ValueError``) for anything that is not such a file or has no coherent run in it. The file
    does not record the repo state; ``git_sha`` / ``git_dirty`` are the reader's."""
    try:
        document = _Document.model_validate_json(raw)
        rows = document.results.results
        if not rows:
            raise PromptfooResultsError("the file has no result rows (did the eval run?)")
        golden = {
            (r.metadata.golden.golden_set_version, r.metadata.golden.golden_set_sha256)
            for r in rows
        }
        if len(golden) != 1:
            raise PromptfooResultsError(f"the rows come from {len(golden)} different golden sets")
        [(version, digest)] = golden
        by_config: dict[str, list[_Row]] = {}
        for row in rows:
            by_config.setdefault(row.provider.label, []).append(row)
        judges = [(c.assertion.metric, c.judge) for r in rows for c in _components(r) if c.judge]
        return GenerationRun(
            info=GenerationRunInfo(
                date=document.results.timestamp,
                promptfoo_version=document.metadata.promptfoo_version,
                git_sha=git_sha,
                git_dirty=git_dirty,
                golden_set_version=version,
                golden_set_sha256=digest,
                judge_provider=_agree("judge provider", (j.judge_provider for _, j in judges)),
                judge_model=_agree("judge model", (j.judge_model for _, j in judges)),
                judge_prompt_versions=_judge_prompt_versions(judges),
            ),
            configs={label: _config(label, group) for label, group in by_config.items()},
        )
    except ValidationError as exc:
        raise PromptfooResultsError(
            f"not a promptfoo results file I can read: {_describe(exc)}"
        ) from exc


def read_generation_results(
    path: Path, *, git_sha: str | None = None, git_dirty: bool | None = None
) -> GenerationRun:
    return parse_results(path.read_bytes(), git_sha=git_sha, git_dirty=git_dirty)


def read_generation_baseline(path: Path) -> dict[str, GenerationBaselineEntry]:
    """The committed baseline file, validated: a row missing a required field is an error."""
    return _BASELINE_FILE.validate_json(path.read_bytes())


def _components(row: _Row) -> list[_Component]:
    return row.grading_result.component_results if row.grading_result else []


def _config(label: str, rows: list[_Row]) -> GenerationConfigResult:
    def identity(name: str, values: Iterable[str | None]) -> str | None:
        return _agree(f"{label}: {name}", values)

    return GenerationConfigResult(
        config=label,
        provider=identity("provider", (r.metadata.provider for r in rows)),
        model=identity("model", (r.metadata.model for r in rows)),
        prompt_version=identity("prompt version", (r.metadata.prompt_version for r in rows)),
        index_version=identity("index version", (r.metadata.index_version for r in rows)),
        retrieval_config_hash=identity(
            "retrieval config hash", (r.metadata.retrieval_config_hash for r in rows)
        ),
        cases=sorted((_case(label, row) for row in rows), key=lambda case: case.id),
    )


def _case(label: str, row: _Row) -> GenerationCase:
    md, usage = row.metadata, row.token_usage or _TokenUsage()
    common: dict[str, Any] = {
        "id": md.golden.id,
        "type": md.golden.type,
        "cold_start": md.cold_start,
        "llm_cache_hits": md.llm_cache_hits,
        "input_tokens": usage.prompt,
        "output_tokens": usage.completion,
        "cost_usd": row.cost or 0.0,
    }
    if row.failure_reason == _ERRORED:
        return GenerationCase(**common, error=_generator_error(row))
    if row.failure_reason not in (_PASSED, _FAILED):
        raise PromptfooResultsError(f"{label} {md.golden.id}: unknown failureReason")
    components = _components(row)
    if not components:
        raise PromptfooResultsError(
            f"{label} {md.golden.id}: a graded row has no assertion results"
        )
    metrics: dict[str, MetricResult] = {}
    for component in components:
        name = component.assertion.metric
        if name in metrics:
            raise PromptfooResultsError(f"{label} {md.golden.id}: metric {name} appears twice")
        metrics[name] = _metric(component)
    latency = md.latency_ms.get("total") if md.latency_ms else None
    return GenerationCase(**common, metrics=metrics, latency_ms=latency)


def _generator_error(row: _Row) -> ErrorInfo:
    """The tagged error of a failed generator call. The metadata has the fields; the
    ``[Kind quota=…]`` prefix of ``error`` is the fallback for a row without them."""
    md, text = row.metadata, row.error or ""
    tag = _TAG.match(text)
    kind = md.error_kind or (tag["kind"] if tag else "UnknownError")
    is_quota = md.is_quota if md.error_kind else bool(tag and tag["quota"] == "true")
    return ErrorInfo(
        stage="generator",
        kind=kind,
        bases=md.error_bases,
        is_quota=is_quota,
        skipped=md.skipped,
        detail=_cap(text[tag.end() :] if tag else text),
    )


def _metric(component: _Component) -> MetricResult:
    judge = component.judge
    if component.not_applicable is None:
        return _errored("assertion", "AssertionFailed", component.reason)
    if component.not_applicable:
        return MetricResult(state="na")
    if component.errored:
        failure = judge.error if judge else None
        if failure is None:
            return _errored("judge", "UnknownError", component.reason)
        error = ErrorInfo(
            stage="judge",
            kind=failure.kind,
            is_quota=failure.is_quota,
            provider_side=failure.provider_side,
            skipped=failure.detail.startswith("skipped:"),
            detail=_cap(failure.detail),
        )
        return MetricResult(state="errored", error=error)
    if component.reason.startswith("malformed input"):
        return _errored("assertion", "MalformedInput", component.reason)
    return MetricResult(
        state="scored",
        value=component.score,
        claims=judge.n_claims if judge else None,
        supported=judge.n_supported if judge else None,
    )


def _errored(stage: ErrorStage, kind: str, detail: str) -> MetricResult:
    return MetricResult(
        state="errored", error=ErrorInfo(stage=stage, kind=kind, detail=_cap(detail))
    )


def _judge_prompt_versions(judges: list[tuple[str, _Judge]]) -> dict[str, str]:
    versions = {
        metric: _agree(
            f"{metric} judge prompt version", (j.prompt_version for m, j in judges if m == metric)
        )
        for metric in sorted({metric for metric, _ in judges})
    }
    return {metric: version for metric, version in versions.items() if version is not None}


def _agree(what: str, values: Iterable[str | None]) -> str | None:
    """The one value every row reports, ``None`` if none does. A run is one setup: rows that
    disagree are not one run."""
    found = sorted({v for v in values if v is not None})
    if len(found) > 1:
        raise PromptfooResultsError(f"{what} differs between rows: {', '.join(found)}")
    return found[0] if found else None


def _cap(text: str) -> str:
    return " ".join(text.split())[:_MAX_DETAIL]


def _describe(exc: ValidationError) -> str:
    shown = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:3]]
    more = exc.error_count() - len(shown)
    return "; ".join(shown) + (f" (+{more} more)" if more > 0 else "")
