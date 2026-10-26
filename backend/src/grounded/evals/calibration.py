"""Confidence calibration (Tech.md §9.8, PRD FR-12, ticket 8.04): does the server-side confidence of
a claim track the judge's faithfulness verdict on it? Reads one promptfoo results file; calls no
model and writes nothing (the CLI prints, and writes with ``--out``).

**Buckets** (Tech §9.8): ``low`` is ``confidence < 0.5``, ``mid`` is ``0.5 <= confidence <= 0.8``
(both edges belong to ``mid``) and ``high`` is ``confidence > 0.8``. 0.5 is also where the UI flags
a claim as "Low confidence", so ``low`` is exactly the flagged claims.

**What a claim counts as.** The judge's faithfulness labels are ``SUPPORTED`` and ``NOT_SUPPORTED``
(``schemas/judge.py``); there is no partial grade. Per claim of an answer:

- ``SUPPORTED`` → *supported*; ``NOT_SUPPORTED`` → *not supported*. A ``NOT_SUPPORTED`` decided
  without a call (no usable source, rubric rule 6, ``decided_locally``) is the rubric's verdict and
  counts too; it is also counted in its own column, because such a claim normally has no valid
  citation, so the uncited cap (0.2) puts it in ``low`` mechanically.
- A verdict replayed from the eval cache is the original call's verdict: it counts like the others.
- A claim the judge was asked about but could not grade (its own ``error``, or every claim of an
  errored faithfulness result that has no per-claim record, e.g. a run stopped on a quota) is
  *judge failed*: left out of ``n``, counted on its own. The verdicts of the other claims of a
  partly judged answer stay in: a claim's verdict does not depend on its neighbours.
- A claim of an answer the judge does not grade (faithfulness not applicable: ``no_rag``, which
  cites nothing) is *not judged*: left out of ``n``, counted on its own.

**Per config.** ``no_rag`` has no citations, so its claims are all capped and none is judged; the
headline is ``hybrid``, the config the demo serves. Configs are never pooled.

**Monotonic** means the supported rate never falls from ``low`` to ``mid`` to ``high``, over the
buckets with ``n > 0`` (an empty bucket has no rate); with fewer than two such buckets it cannot be
tested (``None``). Equal rates count as monotonic. Rates are compared as exact fractions.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final, Literal

from pydantic import AliasPath, BaseModel, ConfigDict, Field, ValidationError

from grounded.evals.generation_results import PromptfooResultsError, parse_results
from grounded.evals.retrieval_runner import EVAL_DIR
from grounded.schemas.generation_eval import GenerationConfigResult, GenerationRunInfo

CALIBRATION_DIR: Final = EVAL_DIR / "calibration"
LOW_BELOW: Final = 0.5  # Tech §9.8; the UI's "Low confidence" flag is the same cut
HIGH_ABOVE: Final = 0.8
HEADLINE_CONFIG: Final = "hybrid"

Bucket = Literal["low", "mid", "high"]
BUCKETS: Final[tuple[Bucket, ...]] = ("low", "mid", "high")
Outcome = Literal["supported", "not_supported", "judge_failed", "not_judged"]
# The judge's faithfulness labels (``schemas/judge.py``), mapped explicitly; any other is refused.
_VERDICTS: Final[dict[str, Outcome]] = {"SUPPORTED": "supported", "NOT_SUPPORTED": "not_supported"}
_HAS_VERDICT: Final[frozenset[Outcome]] = frozenset(_VERDICTS.values())


class CalibrationError(ValueError):
    """The input cannot be calibrated as asked; the message says why."""


@dataclass(frozen=True, slots=True)
class ClaimOutcome:
    """One claim: its server-side confidence and what the judge made of it."""

    confidence: float
    outcome: Outcome
    decided_locally: bool = False
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class BucketRow:
    name: str  # a bucket, or "all"
    n: int  # claims with a verdict
    supported: int
    decided_locally: int
    judge_failed: int
    not_judged: int

    @property
    def rate(self) -> float | None:
        """Supported share of the claims with a verdict; ``None`` for an empty bucket."""
        return self.supported / self.n if self.n else None


@dataclass(frozen=True, slots=True)
class Calibration:
    buckets: tuple[BucketRow, ...]  # low, mid, high
    overall: BucketRow
    cache_hits: int  # verdicts replayed from the eval cache
    monotonic: bool | None  # None: fewer than two buckets with n > 0


def bucket_of(confidence: float) -> Bucket:
    """``low`` below 0.5, ``high`` above 0.8, ``mid`` from 0.5 to 0.8 inclusive. Refuses a value
    outside ``[0, 1]`` or not finite: the server never produces one (Tech §9.8)."""
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise CalibrationError(f"confidence {confidence!r} is not in [0, 1]")
    if confidence < LOW_BELOW:
        return "low"
    return "high" if confidence > HIGH_ABOVE else "mid"


def calibrate(claims: Iterable[ClaimOutcome]) -> Calibration:
    """Counts per bucket and overall, the supported rate and whether it is monotonic. Pure; the
    result does not depend on the order of ``claims``."""
    every = list(claims)
    by_bucket = {name: [c for c in every if bucket_of(c.confidence) == name] for name in BUCKETS}
    buckets = tuple(_row(name, by_bucket[name]) for name in BUCKETS)
    cache_hits = sum(c.cache_hit for c in every if c.outcome in _HAS_VERDICT)
    return Calibration(buckets, _row("all", every), cache_hits, is_monotonic(buckets))


def _row(name: str, claims: Sequence[ClaimOutcome]) -> BucketRow:
    judged = [c for c in claims if c.outcome in _HAS_VERDICT]
    return BucketRow(
        name,
        n=len(judged),
        supported=sum(c.outcome == "supported" for c in judged),
        decided_locally=sum(c.decided_locally for c in judged),
        judge_failed=sum(c.outcome == "judge_failed" for c in claims),
        not_judged=sum(c.outcome == "not_judged" for c in claims),
    )


def is_monotonic(buckets: Sequence[BucketRow]) -> bool | None:
    """Is the supported rate non-decreasing over the buckets that have claims, in order?"""
    filled = [row for row in buckets if row.n]
    if len(filled) < 2:
        return None
    # a/b <= c/d  <=>  a*d <= c*b for positive b, d: no float rounding in the comparison
    return all(a.supported * b.n <= b.supported * a.n for a, b in pairwise(filled))


# --- Reading promptfoo's output ------------------------------------------------------------------


class _Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _AnswerClaim(_Wire):
    confidence: float


class _JudgedClaim(_Wire):
    claim_index: int
    confidence: float
    verdict: str | None = None
    decided_locally: bool = False
    cache_hit: bool = False


class _Judge(_Wire):
    claims: list[_JudgedClaim] = []


class _Component(_Wire):
    metric: str = Field(validation_alias=AliasPath("assertion", "metric"))
    errored: bool = False  # not applicable: no judge record, so its claims are "not judged"
    judge: _Judge | None = None


class _Row(_Wire):
    config: str = Field(validation_alias=AliasPath("provider", "label"))
    question_id: str = Field(validation_alias=AliasPath("metadata", "golden", "id"))
    claims: list[_AnswerClaim] = Field(default=[], validation_alias=AliasPath("metadata", "claims"))
    components: list[_Component] = Field(
        default=[], validation_alias=AliasPath("gradingResult", "componentResults")
    )


class _Document(_Wire):
    rows: list[_Row] = Field(validation_alias=AliasPath("results", "results"))


@dataclass(frozen=True, slots=True)
class ConfigCalibration:
    identity: GenerationConfigResult
    answers: int
    calibration: Calibration


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    source_name: str
    source_sha256: str
    git_sha: str | None  # promptfoo's file does not record the commit; the reader supplies it
    info: GenerationRunInfo
    configs: tuple[ConfigCalibration, ...]


def _claim_outcomes(row: _Row) -> list[ClaimOutcome]:
    """Every claim of one answer with its outcome (rules in the module docstring)."""
    where = f"{row.config} {row.question_id}"
    faithfulness = [c for c in row.components if c.metric == "faithfulness"]
    if len(faithfulness) > 1:
        raise CalibrationError(f"{where}: faithfulness appears twice")
    component = faithfulness[0] if faithfulness else None
    judged = component.judge.claims if component and component.judge else []
    if not judged:
        failed = component is not None and component.errored
        outcome: Outcome = "judge_failed" if failed else "not_judged"
        return [ClaimOutcome(c.confidence, outcome) for c in row.claims]
    if len(judged) != len(row.claims):
        raise CalibrationError(
            f"{where}: the judge recorded {len(judged)} claim(s), the answer has {len(row.claims)}"
        )
    outcomes: list[ClaimOutcome] = []
    for entry in judged:
        if entry.verdict is None:
            outcomes.append(ClaimOutcome(entry.confidence, "judge_failed"))
            continue
        if entry.verdict not in _VERDICTS:
            raise CalibrationError(
                f"{where} claim {entry.claim_index}: unknown verdict {entry.verdict!r} "
                f"(the faithfulness labels are {', '.join(_VERDICTS)})"
            )
        outcomes.append(
            ClaimOutcome(
                entry.confidence,
                _VERDICTS[entry.verdict],
                decided_locally=entry.decided_locally,
                cache_hit=entry.cache_hit,
            )
        )
    return outcomes


def build_report(
    raw: bytes,
    *,
    source_name: str,
    configs: Sequence[str] | None = None,
    git_sha: str | None = None,
) -> CalibrationReport:
    """Calibrate each config of a promptfoo results file. ``configs`` picks and orders them;
    default: every config in the file, ``hybrid`` first. Raises ``CalibrationError`` for a file
    that is not a readable run, an unknown config, an unknown verdict or a confidence outside
    ``[0, 1]``."""
    try:
        run = parse_results(raw)
        rows = _Document.model_validate_json(raw).rows
    except (PromptfooResultsError, ValidationError) as exc:
        raise CalibrationError(f"not a promptfoo results file I can read: {exc}") from exc
    names = list(dict.fromkeys(configs)) if configs else _default_order(run.configs)
    unknown = [name for name in names if name not in run.configs]
    if unknown:
        has = ", ".join(sorted(run.configs))
        raise CalibrationError(f"config(s) {', '.join(unknown)} not in the file (it has {has})")
    results: list[ConfigCalibration] = []
    for name in names:
        mine = [row for row in rows if row.config == name]
        outcomes = [outcome for row in mine for outcome in _checked(row)]
        results.append(ConfigCalibration(run.configs[name], len(mine), calibrate(outcomes)))
    return CalibrationReport(
        source_name=source_name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        git_sha=git_sha,
        info=run.info,
        configs=tuple(results),
    )


def _checked(row: _Row) -> list[ClaimOutcome]:
    outcomes = _claim_outcomes(row)
    for index, outcome in enumerate(outcomes):
        try:
            bucket_of(outcome.confidence)
        except CalibrationError as exc:
            raise CalibrationError(f"{row.config} {row.question_id} claim {index}: {exc}") from exc
    return outcomes


def _default_order(configs: Iterable[str]) -> list[str]:
    return sorted(configs, key=lambda name: (name != HEADLINE_CONFIG, name))


# --- Markdown ------------------------------------------------------------------------------------

_RANGES: Final = {"low": "< 0.5", "mid": "0.5 to 0.8", "high": "> 0.8", "all": "0 to 1"}


def render(report: CalibrationReport) -> str:
    """The calibration tables as Markdown, with ``n`` per bucket and the run's identity. Holds no
    question, claim or answer text."""
    info = report.info
    faithfulness_rubric = info.judge_prompt_versions.get("faithfulness", "none")
    git = report.git_sha or "not recorded (promptfoo's file has none; pass --git-sha)"
    lines = [
        f"Confidence calibration of `{report.source_name}` (sha256 {report.source_sha256[:12]}): "
        f"the run of {info.date:%Y-%m-%d %H:%M} UTC, git {git}, golden set "
        f"{info.golden_set_version} (sha256 {info.golden_set_sha256[:8]}), judge "
        f"{info.judge_provider} / {info.judge_model} ({faithfulness_rubric}), promptfoo "
        f"{info.promptfoo_version}.",
        "",
        "Buckets (Tech §9.8): low = confidence < 0.5, mid = 0.5 ≤ confidence ≤ 0.8 (both edges "
        "belong to mid), high = confidence > 0.8. n = claims with a judge verdict; supported "
        "rate = SUPPORTED / n (— when n = 0). NOT_SUPPORTED counts as not supported, including the "
        "verdicts given without a call because the claim had no usable source (decided locally). "
        "A verdict replayed from the eval cache counts like the others. Left out of n and counted "
        "on their own: claims whose judge call failed, and claims of answers the judge does not "
        "grade (faithfulness not applicable). Monotonic: the supported rate never falls from low "
        "to mid to high, over the buckets with n > 0.",
    ]
    for item in report.configs:
        lines += ["", *_config_lines(item)]
    return "\n".join(lines) + "\n"


def _config_lines(item: ConfigCalibration) -> list[str]:
    who, cal = item.identity, item.calibration
    retrieval = (who.retrieval_config_hash or "none")[:8]
    lines = [
        f"**{who.config}**: generator {who.provider} / {who.model}, prompt {who.prompt_version}, "
        f"index {who.index_version}, retrieval config {retrieval}, {item.answers} answers.",
        "",
        "| Bucket | Confidence | n | Supported | Supported rate | Decided locally | Judge failed "
        "| Not judged |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in (*cal.buckets, cal.overall):
        rate = "—" if row.rate is None else f"{row.rate:.3f}"
        lines.append(
            f"| {row.name} | {_RANGES[row.name]} | {row.n} | {row.supported} | {rate} | "
            f"{row.decided_locally} | {row.judge_failed} | {row.not_judged} |"
        )
    filled = [row.name for row in cal.buckets if row.n]
    if cal.monotonic is None:
        verdict = f"not testable ({len(filled)} of 3 buckets have n > 0, it needs two)"
    else:
        verdict = f"{'yes' if cal.monotonic else 'no'} (over {', '.join(filled)})"
    lines += [
        "",
        f"Monotonic: {verdict}. Verdicts replayed from the eval cache: {cal.cache_hits} of "
        f"{cal.overall.n}.",
    ]
    return lines


def published_header(command: str) -> str:
    """The one-line header of a file written with ``--out``."""
    return (
        f"> Generated, do not edit: the output of `{command}` (run in `backend/`; the results file "
        "is gitignored, its name and sha256 are on the first line below). A test fails when the "
        "README block differs from this file."
    )
