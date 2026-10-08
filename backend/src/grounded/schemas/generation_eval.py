"""Generation-eval results and baseline (Tech.md §15.3, §15.7).

``GenerationRun`` is promptfoo's output normalized for the gate: one record per question and config,
every metric in one of three states (scored, not applicable, errored) and every error tagged with
what the file says about it. It is a *record*. Building it (``evals/generation_results.py``) decides
nothing about gating: which errors count toward ``inconclusive``, what ``n`` is and what passes is
the gate's rule (``evals/gate.py``, 4.08). Pydantic is the authority on its shape, like the
retrieval run (AGENTS.md §6.2).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded.schemas.eval import GoldenType

_FROZEN = ConfigDict(frozen=True, extra="forbid")
SHA256_PATTERN = r"^[0-9a-f]{64}$"

MetricState = Literal["scored", "na", "errored"]
# Where a failure happened: the generator call (a provider error row), a judge assertion, or an
# assertion that is neither (a deterministic one given input it cannot read).
ErrorStage = Literal["generator", "judge", "assertion"]


class ErrorInfo(BaseModel):
    """One failure, as the results file records it. The gate classifies by ``kind``."""

    model_config = _FROZEN

    stage: ErrorStage
    kind: str  # the exception's class name; "MalformedInput" / "UnknownError" for the rest
    bases: tuple[str, ...] = ()  # base classes below Exception (generator errors only)
    is_quota: bool = False
    provider_side: bool | None = None  # the judge's own flag; None where the file has none
    skipped: bool = False  # never asked: the run had stopped on an earlier error
    detail: str = ""


class MetricResult(BaseModel):
    """One metric of one case. ``value`` exists exactly when the metric was scored: a not-applicable
    or errored component carries a placeholder score in promptfoo, which is dropped here."""

    model_config = _FROZEN

    state: MetricState
    value: float | None = Field(default=None, ge=0, le=1)
    error: ErrorInfo | None = None
    # Faithfulness only (scored): the claims judged and the ones supported, so that value is
    # supported / claims. The gate averages the per-question values, not these counts.
    claims: int | None = Field(default=None, ge=0)
    supported: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_state(self) -> Self:
        if (self.state == "scored") != (self.value is not None):
            raise ValueError("a value exists exactly when the metric is scored")
        if (self.state == "errored") != (self.error is not None):
            raise ValueError("an error exists exactly when the metric is errored")
        return self


class GenerationCase(BaseModel):
    """One golden question asked of one config.

    Either the generator failed (``error`` set, ``stage="generator"``, no assertion ran, so
    ``metrics`` is empty) or the case was graded and ``metrics`` has a result per assertion, keyed
    by its metric name. A judge error does not error the case: it errors that one metric.
    """

    model_config = _FROZEN

    id: str = Field(pattern=r"^q\d{3}$")
    type: GoldenType
    error: ErrorInfo | None = None
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    cold_start: bool = False  # first call of a worker: pays for the pool and the embedder
    llm_cache_hits: int = Field(default=0, ge=0)  # > 0: a replayed reply, no provider latency
    latency_ms: float | None = None  # the pipeline's own total; graded cases only
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)  # shadow cost (Tech §14), what a failure spent too

    @model_validator(mode="after")
    def _check_error(self) -> Self:
        if self.error is not None and (self.error.stage != "generator" or self.metrics):
            raise ValueError("a generator error means no assertion ran")
        return self


class GenerationConfigResult(BaseModel):
    """Every question run through one config (a promptfoo provider label: ``no_rag``, ``hybrid``).
    The identity fields come from the answers' metadata and are ``None`` when no case got one."""

    model_config = _FROZEN

    config: str
    provider: str | None
    model: str | None
    prompt_version: str | None  # "name@hash" of the answer prompt
    index_version: str | None  # "<fastapi_ref>@<index_config_hash[:8]>"; "none" for no_rag
    retrieval_config_hash: str | None
    cases: list[GenerationCase]

    @model_validator(mode="after")
    def _check_cases(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{self.config}: a question appears twice")
        return self


class GenerationRunInfo(BaseModel):
    """What produced a run. promptfoo's file has no repo state, so ``git_sha`` / ``git_dirty`` are
    whatever the reader supplied (``None`` if it did not)."""

    model_config = _FROZEN

    date: datetime
    promptfoo_version: str
    git_sha: str | None = None
    git_dirty: bool | None = None
    golden_set_version: str
    golden_set_sha256: str = Field(pattern=SHA256_PATTERN)
    judge_provider: str | None = None  # None: no judge call was made
    judge_model: str | None = None
    judge_prompt_versions: dict[str, str] = Field(default_factory=dict)  # metric → "name@hash"


class GenerationRun(BaseModel):
    """One promptfoo eval, normalized: what ``grounded eval gate --suite generation`` reads."""

    model_config = _FROZEN

    suite: Literal["generation"] = "generation"
    info: GenerationRunInfo
    configs: dict[str, GenerationConfigResult]


# --- Baseline (eval/baselines/generation.json, Tech.md §15.7) -----------------------------------


class GenerationThreshold(BaseModel):
    """The gate rule for one metric: ``current >= max(floor, baseline - tolerance)`` over the parts
    that are set.

    ``tolerance`` alone is relative to the baseline (answer correctness 0.08), ``floor`` alone is an
    absolute minimum whatever the baseline says (schema first-try validity 0.95), both together mean
    the higher of the two (faithfulness: floor 0.85, tolerance 0.05). Policy, written by hand in the
    baseline file (AGENTS.md §6.7), never in code.
    """

    model_config = _FROZEN

    tolerance: float | None = Field(default=None, ge=0)
    floor: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _check_rule(self) -> Self:
        if self.tolerance is None and self.floor is None:
            raise ValueError("a threshold needs a tolerance, a floor or both")
        return self


class GenerationBaselineEntry(BaseModel):
    """One config's row in ``eval/baselines/generation.json``. Numbers are copied from a run
    (``grounded eval baseline``, 4.09), never typed; ``thresholds`` is the hand-written part.

    ``metrics`` are the per-question means and ``n`` the questions each was scored on (a metric
    with no scored question has no entry in either). A config with ``thresholds`` is **gated**,
    one without is reported only. Prompt, model and retrieval identity are recorded for the reader;
    the gate compares only what makes numbers incomparable: ``golden_set_version``,
    ``golden_set_sha256`` and ``index_version`` (a prompt or model change is what the gate is for).
    """

    model_config = _FROZEN

    metrics: dict[str, float]
    n: dict[str, int]
    thresholds: dict[str, GenerationThreshold] = Field(default_factory=dict)
    cases: int = Field(ge=1)  # questions the baseline run asked this config
    golden_set_version: str
    golden_set_sha256: str = Field(pattern=SHA256_PATTERN)
    provider: str
    model: str
    prompt_version: str
    index_version: str
    retrieval_config_hash: str | None = None
    judge_provider: str
    judge_model: str
    judge_prompt_versions: dict[str, str]
    promptfoo_version: str
    git_sha: str | None
    git_dirty: bool
    date: datetime
    # Reported, never gated (Tech §15.5), copied from the run's summary (4.09a) so that the eval
    # report can show them next to the quality metrics. A row written without them (or from a run
    # that had no warm, non-cached case) has ``None`` and ``0``: the report prints "—", never a 0.
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    n_latency: int = Field(default=0, ge=0)  # warm, answered, non-cached cases
    cost_per_1k_usd: float | None = Field(default=None, ge=0)
    n_cost: int = Field(default=0, ge=0)  # answered cases

    @model_validator(mode="after")
    def _check_metrics(self) -> Self:
        if set(self.n) != set(self.metrics) or any(count < 1 for count in self.n.values()):
            raise ValueError("n needs a count of at least 1 for exactly the metrics of the row")
        unknown = sorted(set(self.thresholds) - set(self.metrics))
        if unknown:
            raise ValueError(f"thresholds for metrics the row doesn't have: {', '.join(unknown)}")
        return self

    @model_validator(mode="after")
    def _check_reported(self) -> Self:
        if (self.latency_p50_ms is None) != (self.latency_p95_ms is None) or (
            self.latency_p50_ms is None
        ) != (self.n_latency == 0):
            raise ValueError("latency p50, p95 and n_latency exist together, or not at all")
        if (self.cost_per_1k_usd is None) != (self.n_cost == 0):
            raise ValueError("cost_per_1k_usd and n_cost exist together, or not at all")
        return self
