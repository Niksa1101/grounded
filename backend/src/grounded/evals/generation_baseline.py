"""Write rows of ``eval/baselines/generation.json`` from a promptfoo run (Tech.md §15.7, 4.09).

``grounded eval baseline --suite generation --results <promptfoo>.json`` reads the run through the
same parser as the gate (``generation_results.parse_results``) and copies each config's numbers
into the baseline file. Rows come only from a results file (AGENTS.md §7); nothing is typed.

The numbers are the gate's own: this module asks ``evaluate_generation_gate`` for the per-metric
value and ``n`` of a run (macro faithfulness, N/A and unscored cases left out, a generator bad
output kept as a scored zero for ``schema_first_try``) and for the error counts, so a baseline row
and a later gate row of the same results are the same numbers by construction.

**What a baseline may be made of** (decided in 4.09a, the Author may veto any):

- *A fully scored run.* It is refused, and the file left byte-identical, when a config is
  ``inconclusive``, when any case errored for a provider reason (a quota, a 5xx, a timeout:
  generator or judge, a skipped case included), when the judge could not run at all, or when an
  assertion reported a harness bug. A re-run is cheap because the eval LLM cache replays what was
  already paid for. A generator ``ProviderBadOutput`` is a quality miss and a judge
  ``ProviderBadOutput`` leaves one metric unscored: neither blocks a write, and both show in ``n``.
- *One setup per file.* The rows that are kept must come from the same golden set (version and
  bytes), index, generator and judge as the run (``BaselineMismatchError``'s rule in
  ``retrieval_runner``, widened: the RAG value subtracts the correctness of two rows, so they must
  be judged and generated alike). A difference refuses the write; the fix is one run of every
  config, written together.
- *Known git state.* ``git_sha`` is the repository's HEAD when the baseline is written, so write it
  from the checkout that ran the eval, before committing anything else.

**Thresholds are policy, not measurements.** A ``hybrid`` row written for the first time gets
``INITIAL_THRESHOLDS`` (Tech §15.5, approved by the Author in the baseline PR); every other new row
gets none (reported only). A row that already exists keeps the thresholds it has, whatever they
are: the file is the source of truth afterwards, as for retrieval.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from pydantic import TypeAdapter, ValidationError

from grounded.evals.gate import (
    ConfigSummary,
    GateCannotRunError,
    GenerationGateReport,
    evaluate_generation_gate,
)
from grounded.evals.generation_results import read_generation_baseline
from grounded.evals.retrieval_runner import write_json
from grounded.generation.pipeline import NO_RAG_INDEX_VERSION
from grounded.schemas.generation_eval import (
    GenerationBaselineEntry,
    GenerationConfigResult,
    GenerationRun,
    GenerationRunInfo,
    GenerationThreshold,
)

# The config whose row is gated: the system as users meet it. ``no_rag`` is the comparison it is
# measured against and is only reported.
GATED_CONFIG: Final = "hybrid"

# Tech.md §15.5, "Initial thresholds": the rules a first ``hybrid`` row starts with. They are
# written into the baseline file, which is the source of truth from then on (a later write keeps
# the file's), so this table is read exactly once per file. It is a constant and not a ``Settings``
# field because it is copied once into a reviewed baseline PR and never read at run time.
INITIAL_THRESHOLDS: Final[Mapping[str, GenerationThreshold]] = {
    "faithfulness": GenerationThreshold(floor=0.85, tolerance=0.05),
    "correctness": GenerationThreshold(tolerance=0.08),
    "refusal_correctness": GenerationThreshold(tolerance=0.07),
    "schema_first_try": GenerationThreshold(floor=0.95),
}

_FILE: Final = TypeAdapter(dict[str, GenerationBaselineEntry])


class GenerationBaselineError(Exception):
    """The run cannot become baseline rows; nothing was written. The message says why, one problem
    per line."""


@dataclass(frozen=True)
class BaselineUpdate:
    """What a write did: the whole file as it is now, the configs of this run, and which of them
    got the initial thresholds (they need the Author's approval in the baseline PR)."""

    rows: dict[str, GenerationBaselineEntry]
    written: tuple[str, ...]
    seeded: tuple[str, ...]


def update_generation_baseline(path: Path, run: GenerationRun) -> BaselineUpdate:
    """Replace the rows of the configs in ``run`` in the baseline file (created if missing) and
    keep the other rows. Raises ``GenerationBaselineError`` and leaves the file untouched when the
    run cannot be a baseline (see the module docstring); every problem found is in the message.
    """
    if not run.configs:
        raise GenerationBaselineError("the run has no config to write")
    if run.info.git_sha is None or run.info.git_dirty is None:
        # A baseline row must say which code produced it (the gate and the reader need it).
        raise GenerationBaselineError(
            "this run has no git state (not a git checkout), so it can't become a baseline row"
        )
    if run.info.judge_provider is None or run.info.judge_model is None:
        raise GenerationBaselineError(
            "no judge call was made in this run (it has no faithfulness or correctness), so it "
            "can't become a baseline row"
        )
    try:
        # An empty baseline gates nothing; what is wanted is the gate's numbers and error counts.
        report = evaluate_generation_gate(run, {})
    except GateCannotRunError as exc:
        raise GenerationBaselineError(
            "the run shows the eval could not run, so it can't become a baseline row:\n"
            + "\n".join(f"- {reason}" for reason in exc.reasons)
        ) from exc

    unscored = [line for summary in report.configs for line in _run_problems(summary)]
    if unscored:
        raise GenerationBaselineError(
            "the run is not fully scored; a baseline must come from one that is. Re-run the eval "
            "(the eval cache replays what was already paid for):\n"
            + "\n".join(f"- {line}" for line in unscored)
        )

    try:
        existing = read_generation_baseline(path) if path.exists() else {}
    except ValidationError as exc:
        raise GenerationBaselineError(
            f"{path.name} has rows that no longer validate ({exc.error_count()} errors, first: "
            f"{exc.errors()[0]['loc']}). Fix or regenerate them in a baseline PR."
        ) from exc
    kept = {name: row for name, row in existing.items() if name not in run.configs}
    mismatches = _setup_mismatches(kept, run)
    if mismatches:
        raise GenerationBaselineError(
            "\n".join(f"- {line}" for line in mismatches)
            + "\nRows scored on a different setup must not share a file (the RAG value subtracts "
            "one row from another): write every config from one results file."
        )

    rows = dict(kept)
    seeded: list[str] = []
    row_problems: list[str] = []
    for name, result in run.configs.items():
        previous = existing.get(name)
        if previous is not None:
            thresholds = dict(previous.thresholds)
        elif name == GATED_CONFIG:
            thresholds = dict(INITIAL_THRESHOLDS)
            seeded.append(name)
        else:
            thresholds = {}
        entry = _entry(name, result, run.info, report, thresholds, row_problems)
        if entry is not None:
            rows[name] = entry
    if row_problems:
        raise GenerationBaselineError("\n".join(f"- {line}" for line in row_problems))

    _write(path, dict(sorted(rows.items())))
    return BaselineUpdate(rows=rows, written=tuple(run.configs), seeded=tuple(seeded))


# --- Refusals ------------------------------------------------------------------------------------


def _run_problems(summary: ConfigSummary) -> list[str]:
    """Why one config's cases are not fully scored, one line each (empty when they are)."""
    problems: list[str] = []
    errors = summary.errors
    if errors.provider:
        verdict = "inconclusive: " if summary.inconclusive else ""
        problems.append(
            f"{summary.config}: {verdict}{errors.provider} of {summary.cases} case(s) errored for "
            "a provider reason (a quota, a 5xx or a timeout, of the generator or the judge)"
        )
    if errors.malformed_input:
        problems.append(
            f"{summary.config}: {errors.malformed_input} case(s) had an assertion that could not "
            "read its input (a bug in the harness; the metric is unscored)"
        )
    return problems


def _setup_mismatches(kept: Mapping[str, GenerationBaselineEntry], run: GenerationRun) -> list[str]:
    """The kept rows that differ from ``run`` in what makes two rows incomparable, one line each.

    The index is compared only between rows that use one (``no_rag`` has none), the generator
    against any config of the run, and the judge prompt versions only for the metrics both sides
    have (a ``no_rag``-only run has no faithfulness judge).
    """
    info = run.info
    indexes = {
        c.index_version
        for c in run.configs.values()
        if c.index_version not in (None, NO_RAG_INDEX_VERSION)
    }
    generators = {(c.provider, c.model) for c in run.configs.values()}
    lines: list[str] = []
    for name, row in kept.items():
        differences: list[str] = []
        pairs = (
            ("golden_set_version", row.golden_set_version, info.golden_set_version),
            ("golden_set_sha256", _short(row.golden_set_sha256), _short(info.golden_set_sha256)),
            ("judge_provider", row.judge_provider, info.judge_provider),
            ("judge_model", row.judge_model, info.judge_model),
        )
        for field, row_value, run_value in pairs:
            if row_value != run_value:
                differences.append(f"{field}: row {row_value}, run {run_value}")
        uses_index = row.index_version != NO_RAG_INDEX_VERSION
        if uses_index and indexes and row.index_version not in indexes:
            differences.append(
                f"index_version: row {row.index_version}, run {', '.join(sorted(indexes))}"
            )
        if (row.provider, row.model) not in generators:
            run_generators = ", ".join(sorted(f"{p}/{m}" for p, m in generators))
            differences.append(f"generator: row {row.provider}/{row.model}, run {run_generators}")
        for metric, version in sorted(row.judge_prompt_versions.items()):
            run_version = info.judge_prompt_versions.get(metric)
            if run_version is not None and run_version != version:
                differences.append(f"{metric} judge prompt: row {version}, run {run_version}")
        if differences:
            lines.append(f"row {name!r} differs from the run in " + "; ".join(differences))
    return lines


def _short(digest: str) -> str:
    return digest[:8]


# --- Rows ----------------------------------------------------------------------------------------


def _entry(
    name: str,
    result: GenerationConfigResult,
    info: GenerationRunInfo,
    report: GenerationGateReport,
    thresholds: dict[str, GenerationThreshold],
    problems: list[str],
) -> GenerationBaselineEntry | None:
    """The row of one config, or ``None`` after adding to ``problems`` what it lacks."""
    scored = [row for row in report.rows if row.config == name and row.current is not None]
    unscored = sorted(set(thresholds) - {row.metric for row in scored})
    if unscored:
        problems.append(
            f"{name}: no scored case for {', '.join(unscored)}, which the row has a threshold on"
        )
    identity = (result.provider, result.model, result.prompt_version, result.index_version)
    if None in identity:
        problems.append(
            f"{name}: the answers do not report their provider, model, prompt version and index "
            "version (did every call fail?)"
        )
    if unscored or None in identity:
        return None
    summary = next(s for s in report.configs if s.config == name)
    answered = sum(1 for case in result.cases if case.error is None)
    # Validated as a dict, not built from keywords: the identity, git and judge fields were
    # checked above and by the caller, and Pydantic is the authority on the row (AGENTS.md §6.2).
    return GenerationBaselineEntry.model_validate(
        {
            "metrics": {row.metric: row.current for row in scored},
            "n": {row.metric: row.n for row in scored},
            "thresholds": thresholds,
            "cases": summary.cases,
            "golden_set_version": info.golden_set_version,
            "golden_set_sha256": info.golden_set_sha256,
            "provider": result.provider,
            "model": result.model,
            "prompt_version": result.prompt_version,
            "index_version": result.index_version,
            "retrieval_config_hash": result.retrieval_config_hash,
            "judge_provider": info.judge_provider,
            "judge_model": info.judge_model,
            "judge_prompt_versions": info.judge_prompt_versions,
            "promptfoo_version": info.promptfoo_version,
            "git_sha": info.git_sha,
            "git_dirty": info.git_dirty,
            "date": info.date,
            "latency_p50_ms": summary.latency_p50_ms,
            "latency_p95_ms": summary.latency_p95_ms,
            "n_latency": summary.n_latency,
            "cost_per_1k_usd": summary.cost_per_1k_usd,
            "n_cost": 0 if summary.cost_per_1k_usd is None else answered,
        }
    )


def check_against_itself(run: GenerationRun, update: BaselineUpdate) -> GenerationGateReport | None:
    """The gate's verdict on the rows just written, against the run they came from (the 4.09
    check "the gate on the new baseline against itself"). ``None`` when none of the written rows
    is gated. It is a warning light and never undoes a write: a seeded threshold the run itself
    misses (a floor above the measured value) is the Author's call in the baseline PR.

    Only the written rows are compared, so a rewrite of ``no_rag`` alone is not reported as a
    missing ``hybrid``."""
    gated = {name: update.rows[name] for name in update.written if update.rows[name].thresholds}
    return evaluate_generation_gate(run, gated) if gated else None


def _write(path: Path, rows: Mapping[str, GenerationBaselineEntry]) -> None:
    data: dict[str, Any] = _FILE.dump_python(dict(rows), mode="json")
    for row in data.values():
        # A threshold rule is a tolerance, a floor or both: write only the parts that are set, as
        # the hand-written form in Tech §15.7 has them.
        row["thresholds"] = {
            metric: {k: v for k, v in rule.items() if v is not None}
            for metric, rule in row["thresholds"].items()
        }
    write_json(path, data)
