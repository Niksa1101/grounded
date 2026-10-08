"""Record a generation eval run in ``eval_runs`` (DB.md §4, Tech.md §15.7; ticket 4.10c).

``grounded eval record --suite generation --results <promptfoo>.json`` turns a run into one
``eval_runs`` row per config, for the dashboard (Phase 8). CI calls it on a push to ``main`` only,
with the owner connection (``DATABASE_URL_DIRECT``, DB.md §5), and only once the Author has set the
repository variable ``EVAL_RECORD_RUNS``: it is a write to production.

**What a row holds, and what it never holds.** Aggregates only (AGENTS.md §6.13): the config, the
commit and branch, the golden-set version, the prompt version, the models, the gate's verdict, the
metric values with their own ``n``, the case counts and a link to the CI run. No question text, no
question id, no answer. The numbers are the gate's own (``evaluate_generation_gate``), so a row and
a PR comment of the same results agree by construction; nothing is typed by hand (AGENTS.md §7).

**Decisions of 4.10c** (the Author may veto any; DB.md §4 says the same):

- ``status`` is the verdict of the whole run (the gate's: ``fail`` beats ``inconclusive`` beats
  ``pass``), written on every row of it. A reported-only config like ``no_rag`` has no verdict of
  its own, and the dashboard reads "what did the gate say about this run".
- ``metrics`` is flat: ``{"<metric>": value, "n_<metric>": n, ..., "n": cases that did not error for
  a provider reason, "latency_p50_ms", "latency_p95_ms", "n_latency", "cost_per_1k_usd"}``. A metric
  with no scored case is absent, never 0.
- ``errored_case_count`` is the cases that errored for a provider reason (a quota, a 5xx, a
  timeout: the ones ``inconclusive`` counts), so ``errored_case_count / case_count`` is the share
  the 20% rule looks at.
- ``index_config_hash`` is the full ``index_versions.config_hash`` of the index the run was made
  against, read from the CI database; the results carry only the 8-character label. ``none`` for a
  config that retrieves nothing (``no_rag``).
- One insert per call, no natural key: DB.md defines none. A re-run of a CI job is a new run and a
  new row. All rows of a call go in one transaction.
- Nothing is written without ``--write``; the CLI prints the rows it would write instead.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field

from grounded.evals.gate import GateStatus, evaluate_generation_gate
from grounded.generation.pipeline import NO_RAG_INDEX_VERSION
from grounded.schemas.generation_eval import GenerationBaselineEntry, GenerationRun

SUITE: Final = "generation"
# The index of a config that retrieves nothing: the word the baseline rows use as ``index_version``.
NO_INDEX: Final = NO_RAG_INDEX_VERSION
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class EvalRecordError(Exception):
    """The run cannot be recorded; nothing was written."""


class EvalRunRow(BaseModel):
    """One row of ``eval_runs`` without ``id`` and ``created_at`` (the database sets those)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suite: Literal["generation"]
    config_name: str
    git_sha: str = Field(pattern=_GIT_SHA.pattern)
    branch: str = Field(min_length=1)
    golden_set_version: str
    prompt_version: str | None
    index_config_hash: str
    generator_model: str | None
    judge_model: str | None
    status: GateStatus
    metrics: dict[str, float | int]
    case_count: int = Field(ge=0)
    errored_case_count: int = Field(ge=0)
    report_url: str | None


@dataclass(frozen=True)
class ActiveIndex:
    """The index the CI database was ingested with: what a hybrid run retrieved from."""

    git_ref: str
    config_hash: str

    @property
    def label(self) -> str:
        """``<fastapi_ref>@<config_hash[:8]>``: how the results name the index."""
        return f"{self.git_ref}@{self.config_hash[:8]}"


def build_rows(
    run: GenerationRun,
    baseline: Mapping[str, GenerationBaselineEntry],
    *,
    git_sha: str,
    branch: str,
    report_url: str | None,
    active_index: ActiveIndex | None,
) -> list[EvalRunRow]:
    """One row per config of ``run``, sorted by config name. Pure: no I/O, no clock.

    The verdict comes from the gate against ``baseline``; the numbers come from the same function
    without a baseline, which always has rows for every config (a run on another golden set or
    index gets ``fail`` from the first and still has its own numbers in the second). Raises
    ``GateCannotRunError`` as the gate does, and ``EvalRecordError`` when a config's index cannot
    be named.
    """
    verdict = evaluate_generation_gate(run, baseline)
    numbers = evaluate_generation_gate(run, {})
    rows: list[EvalRunRow] = []
    for name in sorted(run.configs):
        result = run.configs[name]
        summary = next(s for s in numbers.configs if s.config == name)
        metrics: dict[str, float | int] = {}
        for row in numbers.rows:
            if row.config == name and row.current is not None and row.n is not None:
                metrics[row.metric] = row.current
                metrics[f"n_{row.metric}"] = row.n
        metrics["n"] = summary.n
        if summary.latency_p50_ms is not None and summary.latency_p95_ms is not None:
            metrics["latency_p50_ms"] = summary.latency_p50_ms
            metrics["latency_p95_ms"] = summary.latency_p95_ms
            metrics["n_latency"] = summary.n_latency
        if summary.cost_per_1k_usd is not None:
            metrics["cost_per_1k_usd"] = summary.cost_per_1k_usd
        rows.append(
            EvalRunRow(
                suite=SUITE,
                config_name=name,
                git_sha=git_sha,
                branch=branch,
                golden_set_version=run.info.golden_set_version,
                prompt_version=result.prompt_version,
                index_config_hash=_index_hash(name, result.index_version, active_index),
                generator_model=result.model,
                judge_model=run.info.judge_model,
                status=verdict.status,
                metrics=metrics,
                case_count=summary.cases,
                errored_case_count=summary.errors.provider,
                report_url=report_url,
            )
        )
    return rows


def _index_hash(config: str, label: str | None, active: ActiveIndex | None) -> str:
    if config == "no_rag" or label == NO_INDEX:
        return NO_INDEX
    if active is None:
        raise EvalRecordError(f"{config}: the active index of the CI database is needed")
    # A label means the answers named the index they were made with: it has to be this one. No label
    # (every call of the config failed) leaves the CI database's only active index as the answer.
    if label is not None and label != active.label:
        raise EvalRecordError(
            f"{config}: the run used index {label}, but the active index of this database is "
            f"{active.label}; record from the job that ingested and ran the eval"
        )
    return active.config_hash


_ACTIVE_INDEX = "SELECT git_ref, config_hash FROM index_versions WHERE is_active"


def read_active_index(conninfo: str) -> ActiveIndex:
    """The active index version of the database at ``conninfo`` (read only)."""
    with psycopg.connect(conninfo) as conn:
        row = conn.execute(_ACTIVE_INDEX).fetchone()
    if row is None:
        raise EvalRecordError("no active index version in the database: run `grounded ingest`")
    return ActiveIndex(git_ref=row[0], config_hash=row[1])


_INSERT = """
INSERT INTO eval_runs (
    suite, config_name, git_sha, branch, golden_set_version, prompt_version, index_config_hash,
    generator_model, judge_model, status, metrics, case_count, errored_case_count, report_url
) VALUES (
    %(suite)s, %(config_name)s, %(git_sha)s, %(branch)s, %(golden_set_version)s, %(prompt_version)s,
    %(index_config_hash)s, %(generator_model)s, %(judge_model)s, %(status)s, %(metrics)s,
    %(case_count)s, %(errored_case_count)s, %(report_url)s
)
RETURNING id
"""


def insert_rows(conninfo: str, rows: Sequence[EvalRunRow]) -> list[UUID]:
    """Insert ``rows`` in one transaction and return their ids; any failure inserts none."""
    ids: list[UUID] = []
    with psycopg.connect(conninfo) as conn:  # commits on exit, rolls back on an exception
        for row in rows:
            params: dict[str, Any] = row.model_dump()
            params["metrics"] = Jsonb(row.metrics)
            fetched = conn.execute(_INSERT, params).fetchone()
            if fetched is None:  # RETURNING yields the inserted row; unreachable unless it broke
                raise EvalRecordError(f"{row.config_name}: the insert returned no id")
            ids.append(fetched[0])
    return ids
