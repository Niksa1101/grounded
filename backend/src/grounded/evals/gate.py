"""The eval gate: compare a run with the committed baseline and say pass or fail (Tech.md §15.5).

**Author-owned** (AGENTS.md §3): ``evaluate_gate`` was written in ticket 2.10 at the Author's
explicit request. The spec tests are in ``tests/unit/test_gate.py``. The generation part is the
second function, ``evaluate_generation_gate``: its contract and spec tests are ticket 4.07
(``tests/unit/test_gate_generation.py``); the Author delegated the implementation to the Agent
("write it", 4.08), who explained it line by line in the PR.

Everything that is not one of the two gate functions is boilerplate that decides nothing: the report
types, the Markdown and the CLI exit code.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from grounded.schemas.eval import RetrievalBaselineEntry, RetrievalRun, RetrievalRunInfo
from grounded.schemas.generation_eval import (
    ErrorInfo,
    GenerationBaselineEntry,
    GenerationCase,
    GenerationConfigResult,
    GenerationRun,
    GenerationRunInfo,
    GenerationThreshold,
)

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
    (``tests/unit/test_gate_generation.py``); 4.08 implements it, at the Author's "write it". What
    is marked *decided* is the Author's of 2026-10-08. Pure: no I/O, no clock, no logging.

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
    An inconclusive config still gets its rows, with ``current`` and ``n``, but ``passed`` and
    ``threshold`` are ``None``: nothing of it is gated, and ``ConfigSummary.inconclusive`` says so.
    This holds for a total outage (every generator call failed for a provider reason): such a
    config has no answer to name its index, and that is not held against it (see *Fail closed*).

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
    its metrics, as not available), plus ``schema_first_try`` when a generator bad output gave it
    a score. ``n`` is the metric's own, ``delta`` is ``current - baseline``, and a metric the
    baseline lacks has ``baseline`` ``None``.

    **Fail closed, with a reason and never an exception** (but ``GateCannotRunError``). Each of
    these makes the status ``fail`` and adds a ``reasons`` line that names the config:

    - a gated baseline config is missing from ``run.configs``;
    - ``run.info`` differs from a gated baseline row in ``golden_set_version`` or
      ``golden_set_sha256``, or the config's ``index_version`` differs from the row's (numbers
      scored on other data are not comparable, so that config gets no metric rows; its summary is
      still reported). A prompt, model or retrieval-config change is *not* a mismatch: catching
      what it did is the gate's job. The ``index_version`` of a config is read from its answers,
      so a config with *no answered case* (every generator call failed) has none: that is
      "unknown", not a difference, and it is not compared, so the inconclusive rule judges the
      config instead of a comparability failure hiding it (a total outage is ``inconclusive``,
      not ``fail``; an outage that is not provider-side, e.g. every case a generator
      ``ProviderBadOutput``, stays a quality result). The golden set is the run's own identity,
      not the answers', so it is compared whatever the config answered. A config that answered
      and still names no index differs;
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
    unrunnable = _cannot_run_reasons(run)
    if unrunnable:
        raise GateCannotRunError(unrunnable)

    rows: list[GateRow] = []
    reasons: list[str] = []
    summaries: list[ConfigSummary] = []
    gated_configs = 0

    # Sorted, so the same inputs always give the same row, summary and reason order.
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

        summary = _summarize(name, result.cases, gated=gated)
        summaries.append(summary)

        if entry is not None and gated:
            differences = _generation_setup_differences(run.info, result, entry)
            if differences:
                reasons.append(
                    f"{name}: run is not comparable with the baseline ({differences}); "
                    "numbers scored on another golden set or index prove nothing"
                )
                continue

        # An inconclusive config is shown but not judged: its rows get no threshold and no verdict.
        enforced = gated and not summary.inconclusive
        for row in _config_rows(name, result.cases, entry, enforced=enforced):
            rows.append(row)
            if row.current is None and row.passed is False:
                reasons.append(f"{name}: gated metric {row.metric} has no scored case")

    if gated_configs == 0:
        reasons.append("no config in the baseline is gated: the gate would check nothing")

    return GenerationGateReport(
        status=_status(rows, reasons, summaries), rows=rows, reasons=reasons, configs=summaries
    )


# --- Helpers of evaluate_generation_gate ---------------------------------------------------------

_FAITHFULNESS: Final = "faithfulness"
_SCHEMA_FIRST_TRY: Final = "schema_first_try"

# Error kinds (``ErrorInfo.kind``, the exception's class name) that count toward ``inconclusive``.
# ``BackoffExhaustedError`` is also found through its base class, ``ProviderRateLimited``.
_PROVIDER_KINDS: Final = frozenset(
    {
        "ProviderRateLimited",
        "BackoffExhaustedError",
        "ProviderUnavailable",
        "ProviderTimeout",
        "EmbedderUnavailableError",
    }
)
# A bug in the harness, not in the system under test: the metric is unscored and reported.
_HARNESS_KINDS: Final = frozenset({"MalformedInput", "AssertionFailed"})

_CANNOT_RUN_HINTS: Final = {
    "ProviderRequestRejected": "a provider refused the request (a rejected key?)",
    "ProviderConfigError": "a provider is not configured (a missing key or judge model?)",
}
_UNNAMED_KIND_HINT: Final = (
    "not a provider failure: the setup is at fault (index, database, harness)"
)

_Category = Literal[
    "provider", "generator_bad_output", "judge_bad_output", "malformed_input", "cannot_run"
]


def _category(error: ErrorInfo) -> _Category:
    """Which bucket an error belongs to. Anything this does not name is ``cannot_run``: an error
    nobody classified must stop the gate, not be counted as a quality result."""
    if error.kind in _PROVIDER_KINDS or "ProviderRateLimited" in error.bases:
        return "provider"
    if error.kind == "ProviderBadOutput":
        return "generator_bad_output" if error.stage == "generator" else "judge_bad_output"
    if error.kind in _HARNESS_KINDS:
        return "malformed_input"
    return "cannot_run"


def _case_errors(case: GenerationCase) -> list[ErrorInfo]:
    """Every error of a case: the generator's (then it has no metrics) or its errored metrics'."""
    errors: list[ErrorInfo] = [] if case.error is None else [case.error]
    errors.extend(result.error for result in case.metrics.values() if result.error is not None)
    return errors


def _is_generator_bad_output(case: GenerationCase) -> bool:
    return case.error is not None and _category(case.error) == "generator_bad_output"


def _cannot_run_reasons(run: GenerationRun) -> list[str]:
    """One line per error kind that shows the gate cannot run, anywhere in the run (every config,
    gated or not): what was found, how many cases, in which configs. Empty when the run is fine."""
    found: Counter[tuple[str, str]] = Counter()  # (kind, config) -> cases
    for name, result in sorted(run.configs.items()):
        for case in result.cases:
            kinds = {e.kind for e in _case_errors(case) if _category(e) == "cannot_run"}
            for kind in kinds:
                found[kind, name] += 1
    return [
        _cannot_run_line(kind, {c: n for (k, c), n in found.items() if k == kind})
        for kind in sorted({kind for kind, _ in found})
    ]


def _cannot_run_line(kind: str, cases_per_config: Mapping[str, int]) -> str:
    where = ", ".join(f"{config} {count}" for config, count in sorted(cases_per_config.items()))
    hint = _CANNOT_RUN_HINTS.get(kind, _UNNAMED_KIND_HINT)
    return f"{kind}: {sum(cases_per_config.values())} case(s) ({where}): {hint}"


def _count_errors(cases: list[GenerationCase]) -> ErrorCounts:
    """Cases per category. A case is counted once per category, however many errors it has there,
    and in every category it has an error for."""
    found = [{_category(error) for error in _case_errors(case)} for case in cases]
    return ErrorCounts(
        provider=sum("provider" in categories for categories in found),
        generator_bad_output=sum("generator_bad_output" in categories for categories in found),
        judge_bad_output=sum("judge_bad_output" in categories for categories in found),
        malformed_input=sum("malformed_input" in categories for categories in found),
    )


def _is_inconclusive(provider_errors: int, cases: int) -> bool:
    """More than ``MAX_PROVIDER_ERROR_RATE`` of the cases errored for provider reasons. Exactly the
    rate is not. No cases is inconclusive too, and nothing divides by it.

    The comparison is between exact fractions, so "exactly the rate" is an exact tie that no
    rounding decides: the float ``0.2`` is not one fifth (it is a hair above), while the fraction
    read from the text ``"0.2"`` is.
    """
    if cases == 0:
        return True
    return Fraction(provider_errors, cases) > Fraction(str(MAX_PROVIDER_ERROR_RATE))


def _mean(values: list[float]) -> float | None:
    """``None`` for no values (never 0 or 1). ``fsum`` is exact, so the order of the cases cannot
    move the last digit."""
    return math.fsum(values) / len(values) if values else None


def _nearest_rank(sorted_values: list[float], percent: int) -> float | None:
    """The value at rank ``ceil(percent / 100 * m)`` of the ``m`` sorted values. ``None`` for no
    values. The ceiling is computed in integers, so no float can move the rank."""
    if not sorted_values:
        return None
    rank = (percent * len(sorted_values) + 99) // 100
    return sorted_values[rank - 1]


def _metric_values(cases: list[GenerationCase], metric: str) -> list[float]:
    """The per-question values of ``metric``, one for each case where it was *scored*. N/A and
    errored metrics, and cases whose generator failed, have none: they are out of the mean and
    out of ``n``. The exception is a generator ``ProviderBadOutput`` (a quality miss, decided):
    the answer was not valid on the first try, so it is a scored 0.0 for ``schema_first_try``."""
    values: list[float] = []
    for case in cases:
        result = case.metrics.get(metric)
        if result is not None and result.value is not None:  # a value exists only when scored
            values.append(result.value)
    if metric == _SCHEMA_FIRST_TRY:
        values.extend(0.0 for case in cases if _is_generator_bad_output(case))
    return values


def _summarize(config: str, cases: list[GenerationCase], *, gated: bool) -> ConfigSummary:
    """Counts, latency and cost of one config (reported, never gated)."""
    errors = _count_errors(cases)
    # Warm, answered cases only: a cold start pays for the pool and the embedder, and a cache hit
    # has no provider latency (Tech §15.3).
    latencies = sorted(
        case.latency_ms
        for case in cases
        if case.error is None
        and case.latency_ms is not None
        and not case.cold_start
        and case.llm_cache_hits == 0
    )
    # Cost keeps cache hits (they replay the original usage) but not a failed call's partial spend.
    mean_cost = _mean([case.cost_usd for case in cases if case.error is None])
    return ConfigSummary(
        config=config,
        gated=gated,
        inconclusive=_is_inconclusive(errors.provider, len(cases)),
        cases=len(cases),
        n=len(cases) - errors.provider,
        n_faithfulness=len(_metric_values(cases, _FAITHFULNESS)),
        errors=errors,
        n_latency=len(latencies),
        latency_p50_ms=_nearest_rank(latencies, 50),
        latency_p95_ms=_nearest_rank(latencies, 95),
        cost_per_1k_usd=None if mean_cost is None else 1000 * mean_cost,
    )


def _generation_setup_differences(
    info: GenerationRunInfo, result: GenerationConfigResult, entry: GenerationBaselineEntry
) -> str:
    """The setup fields where the run and the baseline row disagree, empty when they match. Only
    what makes numbers incomparable: not the prompt, the model or the retrieval config.

    A config's ``index_version`` is read from its answers' metadata and a failed call carries
    none, so a config whose every generator call failed has ``None``. That is "unknown", not
    "different": it is not compared, so the inconclusive rule gets to judge such a config (a total
    outage must not read as a quality fail). The golden set is the run's own identity, not the
    answers', so it is compared whatever the config answered. A config that did answer and still
    names no index is a difference."""
    differences: list[str] = []
    if info.golden_set_version != entry.golden_set_version:
        differences.append(
            f"golden_set_version: baseline {entry.golden_set_version}, "
            f"run {info.golden_set_version}"
        )
    if info.golden_set_sha256 != entry.golden_set_sha256:
        differences.append(
            f"golden_set_sha256: baseline {entry.golden_set_sha256[:12]}, "
            f"run {info.golden_set_sha256[:12]}"
        )
    index_unknown = result.index_version is None and all(
        case.error is not None for case in result.cases
    )
    if not index_unknown and result.index_version != entry.index_version:
        differences.append(
            f"index_version: baseline {entry.index_version}, run {result.index_version}"
        )
    return ", ".join(differences)


def _config_rows(
    config: str,
    cases: list[GenerationCase],
    entry: GenerationBaselineEntry | None,
    *,
    enforced: bool,
) -> list[GateRow]:
    """One row per metric, sorted by name: every metric seen in a case, every metric of the
    baseline row (so a metric nobody scored still shows up, as not available), and
    ``schema_first_try`` when a generator bad output gave it a score."""
    names = {metric for case in cases for metric in case.metrics}
    if entry is not None:
        names.update(entry.metrics)
    if any(_is_generator_bad_output(case) for case in cases):
        names.add(_SCHEMA_FIRST_TRY)
    return [
        _generation_row(config, metric, _metric_values(cases, metric), entry, enforced=enforced)
        for metric in sorted(names)
    ]


def _generation_row(
    config: str,
    metric: str,
    values: list[float],
    entry: GenerationBaselineEntry | None,
    *,
    enforced: bool,
) -> GateRow:
    """The row of one metric. ``enforced`` is true for a gated config that is not inconclusive;
    only then does a thresholded metric get a threshold and a verdict."""
    current = _mean(values)
    baseline = None if entry is None else entry.metrics.get(metric)
    delta = None if current is None or baseline is None else current - baseline
    rule = None if entry is None else entry.thresholds.get(metric)

    threshold = None
    passed = None
    # A thresholded metric always has a baseline value: the baseline row validates that.
    if enforced and rule is not None and baseline is not None:
        threshold = _threshold(rule, baseline)
        # No value (n == 0) fails: a gate that cannot see the metric must not pass it.
        passed = current is not None and current >= threshold - _EPSILON
    return GateRow(
        config=config,
        metric=metric,
        baseline=baseline,
        current=current,
        delta=delta,
        threshold=threshold,
        passed=passed,
        n=len(values),
    )


def _threshold(rule: GenerationThreshold, baseline: float) -> float:
    """The lowest passing value: the higher of the floor and ``baseline - tolerance``, over the
    parts that are set (the rule validates that at least one is)."""
    candidates: list[float] = []
    if rule.floor is not None:
        candidates.append(rule.floor)
    if rule.tolerance is not None:
        candidates.append(baseline - rule.tolerance)
    return max(candidates)


def _status(rows: list[GateRow], reasons: list[str], summaries: list[ConfigSummary]) -> GateStatus:
    """``fail`` beats ``inconclusive`` beats ``pass``: a regression that was seen is not hidden by a
    config that could not be judged. A reported-only config never changes the status."""
    if reasons or any(row.passed is False for row in rows):
        return "fail"
    if any(summary.gated and summary.inconclusive for summary in summaries):
        return "inconclusive"
    return "pass"


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
