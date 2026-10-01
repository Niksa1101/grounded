"""The eval gate: compare a run with the committed baseline and say pass or fail (Tech.md §15.5).

**Author-owned** (AGENTS.md §3): ``evaluate_gate`` is written in ticket 2.10, this file only fixes
the contract. The spec tests are in ``tests/unit/test_gate.py``. The retrieval part lives here in
Phase 2; the generation part and the ``inconclusive`` rule arrive in Phase 4 (4.07-4.08).

Everything below ``evaluate_gate`` is boilerplate that decides nothing: the report's Markdown and
the CLI exit code.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from grounded.schemas.eval import RetrievalBaselineEntry, RetrievalRun

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
    raise NotImplementedError("Author implements the gate in ticket 2.10")


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
