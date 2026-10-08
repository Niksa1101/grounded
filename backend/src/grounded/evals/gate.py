"""The eval gate: compare a run with the committed baseline and say pass or fail (Tech.md §15.5).

**Author-owned** (AGENTS.md §3): ``evaluate_gate`` was written in ticket 2.10 at the Author's
explicit request. The spec tests are in ``tests/unit/test_gate.py``. The generation part is the
second function, ``evaluate_generation_gate``: its contract and spec tests are ticket 4.07
(``tests/unit/test_gate_generation.py``), the implementation is 4.08.

Everything that is not one of the two gate functions is boilerplate that decides nothing: the report
types, the Markdown and the CLI exit code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from grounded.schemas.eval import RetrievalBaselineEntry, RetrievalRun, RetrievalRunInfo
from grounded.schemas.generation_eval import GenerationBaselineEntry, GenerationRun

GateStatus = Literal["pass", "fail", "inconclusive"]


class GateRow(BaseModel):
    """One metric of one config: what the baseline had, what the run got, and the verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: str
    metric: str
    baseline: float | None  # None: the config has no baseline row
    current: float | None  # None: the run has no such config or metric
    delta: float | None  # current - baseline; None when either side is missing
    threshold: float | None  # the lowest ``current`` that passes; None: the metric is not gated
    passed: bool | None  # None: not gated, so reported only
    n: int | None  # questions the run scored for this config; None: not in the run


class GateReport(BaseModel):
    """The gate's whole verdict, rendered by ``render_markdown``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: GateStatus
    rows: list[GateRow]
    # Failures that belong to no single metric (a gated config missing from the run, a setup that
    # differs from the baseline's, nothing gated at all), one human-readable line each.
    reasons: list[str]


def evaluate_gate(run: RetrievalRun, baseline: Mapping[str, RetrievalBaselineEntry]) -> GateReport:
    """Compare ``run`` with ``baseline`` (the rows of ``eval/baselines/retrieval.json``).

    Contract (the spec tests pin each point):

    - **Gated configs.** A config is gated when its baseline row has ``thresholds``; the gated
      metrics are the thresholded ones. Configs without thresholds, and configs of the run that
      have no baseline row, are *reported only*: they get rows with ``threshold`` and ``passed``
      ``None`` and never change the status. A baseline row's own numbers never come from the run.
    - **The rule.** A metric passes iff ``current >= baseline - tolerance``. A drop of exactly the
      tolerance passes; one hair more fails. ``baseline - tolerance`` is floating point:
      ``0.58 - 0.04`` is ``0.5399999999999999``, yet a run that scored ``0.54`` is exactly at the
      tolerance. The row's ``threshold`` is that lowest passing value and ``delta`` is
      ``current - baseline``.
    - **One failing metric fails the suite.** The status is ``fail`` if any gated row fails or
      there is any reason, else ``pass``. Retrieval never calls a provider, so it is never
      ``inconclusive``.
    - **Fail closed, with a reason, never an exception.** Each of these makes the status ``fail``
      and adds one ``reasons`` line that names the config:
        - a gated baseline config is missing from ``run.configs``, or lacks a thresholded metric;
        - ``run.info`` differs from a gated baseline row in ``golden_set_version``,
          ``golden_set_sha256`` or ``index_config_hash`` (numbers scored on different data or
          another index are not comparable, so that config gets no metric rows);
        - no config in the baseline is gated (a gate that checks nothing must not look green).
    - **``n``.** Every row carries the run's ``n`` for its config.
    - **Pure.** No I/O, no clock, no logging: the same inputs give the same report.

    ``baseline`` rows are already validated: ``golden_set_sha256`` and ``git_dirty`` are present.
    """
    rows: list[GateRow] = []
    reasons: list[str] = []
    gated_configs = 0

    # Sorted, so the same inputs always give the same row order.
    for name in sorted(baseline.keys() | run.configs.keys()):
        entry = baseline.get(name)
        result = run.configs.get(name)
        gated = entry is not None and bool(entry.thresholds)
        if gated:
            gated_configs += 1

        if result is None:
            if gated:
                reasons.append(f"{name}: gated config is missing from the run")
            continue

        if entry is not None and gated:
            differences = _setup_differences(run.info, entry)
            if differences:
                reasons.append(
                    f"{name}: run is not comparable with the baseline ({differences}); "
                    "numbers scored on another golden set or index prove nothing"
                )
                continue

        for metric in _metrics_to_report(result.metrics, entry):
            row = _metric_row(name, metric, result.metrics.get(metric), result.n, entry)
            rows.append(row)
            if row.current is None and row.passed is not None:
                reasons.append(f"{name}: gated metric {metric} is missing from the run")

    if gated_configs == 0:
        reasons.append("no config in the baseline is gated: the gate would check nothing")

    failed = bool(reasons) or any(row.passed is False for row in rows)
    return GateReport(status="fail" if failed else "pass", rows=rows, reasons=reasons)


# --- Helpers of evaluate_gate --------------------------------------------------------------------

# A drop of exactly the tolerance passes, but ``baseline - tolerance`` is floating point
# (0.58 - 0.04 == 0.5399999999999999) and so is ``baseline - current``. Metrics are means of a few
# dozen 0/1-ish values: far coarser than this, so it only absorbs rounding, never a real drop.
_EPSILON: Final = 1e-9


def _setup_differences(info: RetrievalRunInfo, entry: RetrievalBaselineEntry) -> str:
    """The setup fields where the run and the baseline row disagree, empty when they match."""
    pairs = (
        ("golden_set_version", info.golden_set_version, entry.golden_set_version),
        ("golden_set_sha256", info.golden_set_sha256, entry.golden_set_sha256),
        ("index_config_hash", info.index_config_hash, entry.index_config_hash),
    )
    return ", ".join(
        f"{field}: baseline {baseline[:12]}, run {current[:12]}"
        for field, current, baseline in pairs
        if current != baseline
    )


def _metrics_to_report(
    current: Mapping[str, float], entry: RetrievalBaselineEntry | None
) -> list[str]:
    """Every metric the run has, plus any gated one it lacks (it must show up as a failure)."""
    missing_gated = [] if entry is None else [m for m in entry.thresholds if m not in current]
    return [*current, *missing_gated]


def _metric_row(
    config: str,
    metric: str,
    current: float | None,
    n: int,
    entry: RetrievalBaselineEntry | None,
) -> GateRow:
    baseline = None if entry is None else entry.metrics.get(metric)
    delta = None if current is None or baseline is None else current - baseline
    rule = None if entry is None else entry.thresholds.get(metric)

    threshold = None
    passed = None
    if rule is not None and baseline is not None:
        threshold = baseline - rule.tolerance
        # ``>=`` is False for NaN, so a NaN current fails closed.
        passed = current is not None and current >= threshold - _EPSILON
    return GateRow(
        config=config,
        metric=metric,
        baseline=baseline,
        current=current,
        delta=delta,
        threshold=threshold,
        passed=passed,
        n=n,
    )


# --- Rendering and exit code (boilerplate: they decide nothing) -----------------------------------

_STATUS_LABEL: Final[dict[GateStatus, str]] = {
    "pass": "✅ pass",
    "fail": "❌ fail",
    "inconclusive": "⚠️ inconclusive",
}
_VERDICT_MARK: Final = {True: "✅", False: "❌", None: "·"}


def exit_code(report: GateReport) -> int:
    """0 for ``pass`` and ``inconclusive``, 1 for ``fail`` (Tech.md §15.5)."""
    return 1 if report.status == "fail" else 0


def render_markdown(report: GateReport) -> str:
    """The report as Markdown: a headline, the metric table, then the reasons."""
    lines = [f"### Retrieval gate: {_STATUS_LABEL[report.status]}", ""]
    if report.rows:
        lines += [
            "| config | metric | baseline | current | Δ | threshold | n | |",
            "|---|---|---:|---:|---:|---:|---:|:-:|",
            *(_row_line(row) for row in report.rows),
            "",
        ]
        if any(row.passed is None for row in report.rows):
            lines += ["· = not gated: reported only.", ""]
    lines += [f"- {reason}" for reason in report.reasons]
    return "\n".join(lines).rstrip() + "\n"


def _row_line(row: GateRow) -> str:
    cells = [
        row.config,
        row.metric,
        _number(row.baseline),
        _number(row.current),
        _number(row.delta, signed=True),
        "not gated" if row.threshold is None else f"≥ {row.threshold:.3f}",
        "—" if row.n is None else str(row.n),
        _VERDICT_MARK[row.passed],
    ]
    return "| " + " | ".join(cells) + " |"


def _number(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


# --- Generation gate (4.07 spec, 4.08 implementation) --------------------------------------------

# Tech.md §15.5: more than this share of a config's cases errored for provider reasons → the config
# is ``inconclusive``. Exactly this share is not.
MAX_PROVIDER_ERROR_RATE: Final = 0.20


class GateCannotRunError(Exception):
    """The results show the gate cannot run (a rejected key, a missing judge configuration): exit
    code 2, not a quality result. ``reasons`` say what was found, one line each."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class ErrorCounts(BaseModel):
    """Cases per error category. A case is counted in every category it has an error for."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: int = 0  # the ones that count toward ``inconclusive``
    generator_bad_output: int = 0  # quality misses: they stay in ``n``
    judge_bad_output: int = 0  # that judge metric is unscored; not counted toward ``inconclusive``
    malformed_input: int = 0  # a bug in the harness: unscored


class ConfigSummary(BaseModel):
    """What the PR comment says about one config besides its metric rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config: str
    gated: bool  # the baseline row has thresholds
    inconclusive: (
        bool  # more than MAX_PROVIDER_ERROR_RATE of its cases errored for provider reasons
    )
    cases: int  # all cases of the config in the run: the denominator of that rate
    n: int  # ``cases`` minus the ones that errored for provider reasons
    n_faithfulness: int  # cases whose faithfulness was scored (not N/A, not errored)
    errors: ErrorCounts
    n_latency: int  # cases in the latency statistics
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    cost_per_1k_usd: float | None


class GenerationGateReport(GateReport):
    """``GateReport`` (status, one row per config and metric with its own ``n``, reasons) plus a
    summary per config in the run, sorted by config name."""

    configs: list[ConfigSummary]


def evaluate_generation_gate(
    run: GenerationRun, baseline: Mapping[str, GenerationBaselineEntry]
) -> GenerationGateReport:
    """Compare ``run`` with ``baseline`` (the rows of ``eval/baselines/generation.json``).

    **Author-owned** (AGENTS.md §3). 4.07 wrote this contract and the spec tests
    (``tests/unit/test_gate_generation.py``, strict ``xfail``); 4.08 implements it. What is marked
    *decided* is the Author's of 2026-10-08. Pure: no I/O, no clock, no logging.

    **Cases and configs.** A case is one golden question asked of one config (``GenerationCase``).
    Everything below is per config; configs never mix. ``cases`` is how many a config has.

    **Gated configs** work as in ``evaluate_gate``: a config is gated when its baseline row has
    ``thresholds``, and the gated metrics are the thresholded ones. Every other config, and every
    config of the run without a baseline row, is *reported only*: rows with ``threshold`` and
    ``passed`` ``None``, and it never changes the status. Baseline numbers never come from the run.

    **Value and ``n`` of a metric** (*decided*: faithfulness is macro). The value is the mean, over
    the cases where the metric is *scored*, of each case's own ``value``: every question weighs the
    same, so an answer with one claim counts as much as one with ten (never sum ``claims`` and
    ``supported`` across cases: that is the micro mean). ``n`` is the number of those cases.

    - *N/A is excluded* (*decided*). A not-applicable metric (faithfulness for ``no_rag``, a refusal
      or an answer with no claim; citation validity and precision for ``no_rag``; citation precision
      for an unanswerable item or an answer with no citation) is out of the value and out of ``n``.
      It is never 0 and never 1. ``n_faithfulness`` is faithfulness's own ``n`` next to ``n``.
    - An *errored* metric (a judge or assertion error) is out of the value and ``n``.
    - A case whose *generator failed* (``error`` set, no assertion ran) has no metric, so it is out
      of every value and ``n``, except a generator ``ProviderBadOutput`` (*decided*: a quality
      miss), which counts as a scored 0.0 for ``schema_first_try`` (the answer was not valid on
      the first try) and stays out of the other metrics.
    - A metric with ``n == 0`` has ``current`` ``None``: no division by zero, never 0 or 1.

    **Provider-errored cases and ``inconclusive``** (*decided*). A case *errored for provider
    reasons* when its generator call failed, or any of its metrics errored, with one of these kinds
    (``error.kind``, or ``"ProviderRateLimited" in error.bases``; generator or judge alike):
    ``ProviderRateLimited`` (a daily quota or a per-minute limit), ``BackoffExhaustedError``,
    ``ProviderUnavailable`` (5xx), ``ProviderTimeout``. A case skipped after a daily quota carries
    that kind and counts. A case counts once, however many errors it has. Not counted:

    - a generator ``ProviderBadOutput`` (a quality miss, above) and a judge ``ProviderBadOutput``
      (that judge metric is unscored; the case stays);
    - ``MalformedInput`` and ``AssertionFailed`` (a bug in the harness): unscored, reported.

    A config is ``inconclusive`` when more than 20% of its cases (``MAX_PROVIDER_ERROR_RATE``)
    errored for provider reasons; exactly 20% is not (compare exactly: 5 of 25 is not, 6 of 25
    is). The denominator is *the config's own cases*, not all configs' (proposed in 4.07, the
    Author may veto: the PR comment needs a verdict per config, and ``no_rag`` and ``hybrid`` fail
    differently). Zero cases, or every case errored, is inconclusive too, without dividing by zero.
    An inconclusive config still gets its rows, with ``current`` and ``n``, but ``passed`` is
    ``None``: nothing of it is gated, and ``ConfigSummary.inconclusive`` says so.

    **The gate could not run.** ``ProviderRequestRejected`` or ``ProviderConfigError`` anywhere in
    the run (generator or judge: a refused key, a missing judge model) raises ``GateCannotRunError``
    with a reason per kind found; the CLI exits 2. Proposed, not in the Author's list: a kind this
    contract does not name (``NoActiveIndexError``, ``IndexMismatchError``, a database error,
    ``UnknownError``) is the setup's fault and raises it too, while ``EmbedderUnavailableError``
    (the embedding API failed) counts as a provider reason.

    **Thresholds** (Tech §15.5, ``GenerationThreshold``): a gated metric passes iff
    ``current >= max(floor, baseline - tolerance)`` over the parts that are set, and the row's
    ``threshold`` is that number. Faithfulness: floor 0.85 and tolerance 0.05; correctness:
    tolerance 0.08; ``refusal_correctness``: tolerance 0.07; ``schema_first_try``: floor 0.95. A
    value exactly at the threshold passes though ``baseline - tolerance`` is floating point
    (``0.9 - 0.07`` is not ``0.83``): absorb rounding as ``evaluate_gate`` does.

    **Rows.** One per config and metric, sorted by config, then metric: every metric seen in any
    case of the config plus every metric of its baseline row (so an all-errored config still lists
    its metrics, as not available). ``n`` is the metric's own, ``delta`` is ``current - baseline``,
    and a metric the baseline lacks has ``baseline`` ``None``.

    **Fail closed, with a reason and never an exception** (but ``GateCannotRunError``). Each of
    these makes the status ``fail`` and adds a ``reasons`` line that names the config:

    - a gated baseline config is missing from ``run.configs``;
    - ``run.info`` differs from a gated baseline row in ``golden_set_version`` or
      ``golden_set_sha256``, or the config's ``index_version`` differs from the row's (numbers
      scored on other data are not comparable, so that config gets no metric rows; its summary is
      still reported). A prompt, model or retrieval-config change is *not* a mismatch: catching
      what it did is the gate's job;
    - a thresholded metric with ``n == 0`` in a config that is not inconclusive: its row has
      ``current`` ``None`` and ``passed`` ``False``;
    - no config in the baseline is gated (a gate that checks nothing must not look green).

    **Status.** ``fail`` if there is any reason or a gated row failed; else ``inconclusive`` if a
    gated config is inconclusive; else ``pass``. A reported-only config, inconclusive or not,
    never changes it.

    **Summaries** (``configs``: one per config of the run, sorted by name): ``cases``, ``n``
    (``cases`` minus the provider-errored ones), ``n_faithfulness``, ``errors`` (cases per
    category), and latency and cost, which are reported and never gated:

    - ``latency_p50_ms`` and ``latency_p95_ms`` are nearest-rank percentiles (the value at rank
      ``ceil(p/100 * m)`` of the ``m`` sorted latencies; proposed in 4.07, the Author may veto)
      over the cases with an answer, a ``latency_ms``, no ``cold_start`` and no cache hit
      (Tech §15.3); ``n_latency`` is ``m``. No such case: ``None``.
    - ``cost_per_1k_usd`` is 1000 times the mean ``cost_usd`` of the cases with an answer (a
      cache hit stays in, Tech §15.3; a failed call's partial spend and the judge's cost do not).
      No such case: ``None``.
    """
    raise NotImplementedError("Author implements in 4.08")


# --- Generation report: Markdown (boilerplate: it decides nothing) -------------------------------


def render_generation_markdown(report: GenerationGateReport) -> str:
    """The report as Markdown: a headline, a notice per inconclusive config, the metric table, the
    per-config summary (cases, ``n``, errors, latency, cost), then the reasons."""
    lines = [f"### Generation gate: {_STATUS_LABEL[report.status]}", ""]
    for summary in report.configs:
        if summary.inconclusive:
            lines += [_inconclusive_notice(summary), ""]
    if report.rows:
        lines += [
            "| config | metric | baseline | current | Δ | threshold | n | |",
            "|---|---|---:|---:|---:|---:|---:|:-:|",
            *(_row_line(row) for row in report.rows),
            "",
        ]
        if any(row.passed is None for row in report.rows):
            lines += [
                "· = not gated: reported only (no thresholds, or the config is inconclusive).",
                "",
            ]
    if report.configs:
        lines += [
            "| config | cases | n | n faithfulness | provider errors | generator bad output "
            "| judge bad output | malformed | latency p50 / p95 (ms) | cost / 1k |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *(_summary_line(summary) for summary in report.configs),
            "",
        ]
    lines += [f"- {reason}" for reason in report.reasons]
    return "\n".join(lines).rstrip() + "\n"


def _inconclusive_notice(summary: ConfigSummary) -> str:
    share = summary.errors.provider / summary.cases if summary.cases else 1.0
    return (
        f"> ⚠️ **{summary.config}: inconclusive.** {summary.errors.provider} of {summary.cases} "
        f"cases ({share:.1%}) errored for provider reasons (limit {MAX_PROVIDER_ERROR_RATE:.0%}: "
        "a quota, a 5xx or a timeout). Its metrics are shown with n but not gated, and an "
        "inconclusive run is not a pass: re-run it."
    )


def _summary_line(summary: ConfigSummary) -> str:
    errors = summary.errors
    latency = (
        "—"
        if summary.latency_p50_ms is None or summary.latency_p95_ms is None
        else f"{summary.latency_p50_ms:.0f} / {summary.latency_p95_ms:.0f} (n={summary.n_latency})"
    )
    cost = "—" if summary.cost_per_1k_usd is None else f"${summary.cost_per_1k_usd:.4f}"
    cells = [
        summary.config,
        str(summary.cases),
        str(summary.n),
        str(summary.n_faithfulness),
        str(errors.provider),
        str(errors.generator_bad_output),
        str(errors.judge_bad_output),
        str(errors.malformed_input),
        latency,
        cost,
    ]
    return "| " + " | ".join(cells) + " |"
