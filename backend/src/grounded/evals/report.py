"""``grounded eval report``: the eval tables of the README, rendered from the committed baselines.

Only ``eval/baselines/retrieval.json`` and ``eval/baselines/generation.json`` are read, so the
numbers in the README are pasted from this output and never typed (AGENTS.md §7). The output is a
pure function of the two files: fixed order, fixed rounding (3 decimals), no clock. A test checks
that the block of the README between its markers is exactly this output.

Every metric is printed with its ``n`` (the questions it was scored on; not-applicable and unscored
questions are left out). A missing number is ``—``, never a 0. Met / not met compares the
**unrounded** value with the target of PRD §8, so a value that prints as ``0.750`` can still be
below a target of 0.75.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from grounded.evals.generation_results import read_generation_baseline
from grounded.evals.retrieval_runner import metric_names, read_baseline
from grounded.generation.pipeline import NO_RAG_INDEX_VERSION
from grounded.schemas.eval import RetrievalBaselineEntry
from grounded.schemas.generation_eval import GenerationBaselineEntry

RETRIEVAL_FILE: Final = "eval/baselines/retrieval.json"
GENERATION_FILE: Final = "eval/baselines/generation.json"

# Same absorption as the gates: a mean of a few dozen values is far coarser than this, so it only
# keeps float noise (0.1 + 0.2) from deciding a verdict.
_EPSILON: Final = 1e-9
_MISSING: Final = "—"
_GATED_CONFIG: Final = "hybrid"  # the PRD §8 targets are about the system as users meet it


@dataclass(frozen=True)
class Target:
    """A PRD §8 target: at least ``value``, or strictly more than it."""

    value: float
    strict: bool = False

    def met(self, measured: float) -> bool:
        if self.strict:
            return measured > self.value + _EPSILON
        return measured >= self.value - _EPSILON

    @property
    def label(self) -> str:
        number = f"{self.value:.2f}" if self.value else "0"
        return f"{'>' if self.strict else '≥'} {number}"


# PRD §8 "Initial targets", the numbers in one place. They are about ``hybrid``. Rerank lift
# (Phase 6) and judge-human agreement (4.11) are not baseline rows and arrive with their tickets.
RETRIEVAL_TARGETS: Final[Mapping[str, Target]] = {
    "recall@5": Target(0.80),
    "mrr": Target(0.60),
    "ndcg@5": Target(0.65),
}
GENERATION_TARGETS: Final[Mapping[str, Target]] = {
    "faithfulness": Target(0.90),
    "correctness": Target(0.75),
    "refusal_correctness": Target(0.90),
    "schema_first_try": Target(0.97),
}
RAG_VALUE_TARGET: Final = Target(0.0, strict=True)

_RETRIEVAL_ORDER: Final = ("dense", "fts", "hybrid")
_GENERATION_ORDER: Final = ("no_rag", "hybrid")
_RETRIEVAL_LABELS: Final = {
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "mrr": "MRR",
    "ndcg@5": "nDCG@5",
    "ndcg@10": "nDCG@10",
}
_GENERATION_METRICS: Final = (
    ("faithfulness", "Faithfulness"),
    ("correctness", "Answer correctness"),
    ("refusal_correctness", "Refusal accuracy"),
    ("schema_first_try", "Schema first-try validity"),
    ("citation_validity", "Citation validity"),
    ("citation_precision", "Citation precision"),
)

_LEGEND: Final = (
    f"`{_MISSING}`: not applicable or not scored. `n` is the number of questions a metric was "
    "scored on (not-applicable and unscored questions are left out). Met / not met compares the "
    "unrounded value with the target of PRD §8."
)
_K_NOTE: Final = (
    "`k` is the length of the list a retrieval mode returns (`K_DENSE`, `K_FTS` or `K_FUSED`); "
    "MRR runs over it."
)


def build_report(retrieval_path: Path, generation_path: Path) -> str:
    """Read the baseline files that exist and render them; a missing file is said so in the text."""
    retrieval = read_baseline(retrieval_path) if retrieval_path.exists() else None
    generation = read_generation_baseline(generation_path) if generation_path.exists() else None
    return render_report(retrieval, generation)


def render_report(
    retrieval: Mapping[str, RetrievalBaselineEntry] | None,
    generation: Mapping[str, GenerationBaselineEntry] | None,
) -> str:
    """The retrieval ablation and the generation table as Markdown, ending in a newline. ``None``
    (or no rows) for a baseline means it is not committed yet."""
    sections: list[str] = []
    if retrieval:
        sections.append(_retrieval_section(retrieval))
    else:
        sections.append(f"**Retrieval baseline:** not committed yet (`{RETRIEVAL_FILE}`).")
    if generation:
        sections.append(_generation_section(generation))
    else:
        sections.append(f"**Generation baseline:** not committed yet (`{GENERATION_FILE}`).")
    if retrieval or generation:
        sections.append(f"{_LEGEND} {_K_NOTE}" if retrieval else _LEGEND)
    return "\n\n".join(sections) + "\n"


# --- Retrieval -----------------------------------------------------------------------------------


def _retrieval_section(rows: Mapping[str, RetrievalBaselineEntry]) -> str:
    names = _ordered(rows, _RETRIEVAL_ORDER)
    metrics = metric_names()
    values = list(rows.values())
    indexes = _unique(f"{r.fastapi_ref}@{r.index_config_hash[:8]}" for r in values)
    embeddings = _unique(f"{r.embedding_model} ({r.embedding_dim} dims)" for r in values)
    provenance = (
        f"golden set {_unique(r.golden_set_version for r in values)} "
        f"(sha256 {_unique(r.golden_set_sha256[:8] for r in values)}), index {indexes}, "
        f"embedding {embeddings}, {_git_and_date(values)}"
    )
    table = [
        ["Config", "n", "k", *(_RETRIEVAL_LABELS.get(m, m) for m in metrics)],
        *(
            [
                name,
                str(rows[name].n),
                str(rows[name].k),
                *(_number(rows[name].metrics.get(m)) for m in metrics),
            ]
            for name in names
        ),
    ]
    hybrid = rows.get(_GATED_CONFIG)
    if hybrid is not None:
        table.append(
            [
                f"*PRD §8 target ({_GATED_CONFIG})*",
                "",
                "",
                *(_target_label(RETRIEVAL_TARGETS.get(m)) for m in metrics),
            ]
        )
        table.append(
            [
                f"*{_GATED_CONFIG} vs target*",
                "",
                "",
                *(_verdict(RETRIEVAL_TARGETS.get(m), hybrid.metrics.get(m)) for m in metrics),
            ]
        )
    return "\n".join(
        [
            f"**Retrieval ablation** (`{RETRIEVAL_FILE}`; retrieval only, no LLM): {provenance}.",
            "",
            *_markdown_table(table),
        ]
    )


# --- Generation ----------------------------------------------------------------------------------


def _generation_section(rows: Mapping[str, GenerationBaselineEntry]) -> str:
    names = _ordered(rows, _GENERATION_ORDER)
    values = list(rows.values())
    indexes = [r.index_version for r in values if r.index_version != NO_RAG_INDEX_VERSION]
    prompts = ", ".join(f"{name} {rows[name].prompt_version}" for name in names)
    judge_prompts = ", ".join(sorted({v for r in values for v in r.judge_prompt_versions.values()}))
    provenance = (
        f"golden set {_unique(r.golden_set_version for r in values)} "
        f"(sha256 {_unique(r.golden_set_sha256[:8] for r in values)}), "
        f"index {_unique(indexes) if indexes else 'none'}, "
        f"generator {_unique(f'{r.provider} / {r.model}' for r in values)} (prompts: {prompts}), "
        f"judge {_unique(f'{r.judge_provider} / {r.judge_model}' for r in values)} "
        f"({judge_prompts or 'no prompts recorded'}), "
        f"promptfoo {_unique(r.promptfoo_version for r in values)}, {_git_and_date(values)}"
    )

    metrics = list(_GENERATION_METRICS)
    known = {key for key, _ in metrics}
    extra = sorted({key for r in values for key in r.metrics} - known)
    metrics += [(key, key) for key in extra]

    hybrid = rows.get(_GATED_CONFIG)
    header = ["Metric", *names, "Target (PRD §8)", f"{_GATED_CONFIG} vs target"]
    table = [header, ["Questions asked", *(str(rows[name].cases) for name in names), "", ""]]
    for key, label in metrics:
        target = GENERATION_TARGETS.get(key)
        table.append(
            [
                label,
                *(_number(rows[name].metrics.get(key), rows[name].n.get(key)) for name in names),
                _target_label(target),
                _verdict(target, hybrid.metrics.get(key) if hybrid else None),
            ]
        )
    table.append(_rag_value_row(rows, names))
    table.append(
        [
            "Latency p50 / p95, ms (warm)",
            *(_latency(rows[name]) for name in names),
            "",
            "",
        ]
    )
    table.append(
        [
            "Shadow cost per 1k questions, USD",
            *(_cost(rows[name]) for name in names),
            "",
            "",
        ]
    )
    return "\n".join(
        [
            f"**Generation baseline** (`{GENERATION_FILE}`): {provenance}.",
            "",
            *_markdown_table(table),
        ]
    )


def _rag_value_row(rows: Mapping[str, GenerationBaselineEntry], names: Sequence[str]) -> list[str]:
    """correctness(hybrid) - correctness(no_rag), in the hybrid column (PRD §8 "RAG value")."""
    label = "RAG value: correctness(hybrid) - correctness(no_rag)"
    with_rag, without = rows.get(_GATED_CONFIG), rows.get("no_rag")
    value = None
    cell = _MISSING
    if with_rag is not None and without is not None:
        a, b = with_rag.metrics.get("correctness"), without.metrics.get("correctness")
        n_a, n_b = with_rag.n.get("correctness"), without.n.get("correctness")
        if a is not None and b is not None:
            value = a - b
            cell = f"{value:+.3f} (n={n_a} vs {n_b})"
    return [
        label,
        *(cell if name == _GATED_CONFIG else _MISSING for name in names),
        RAG_VALUE_TARGET.label,
        _verdict(RAG_VALUE_TARGET, value),
    ]


def _latency(row: GenerationBaselineEntry) -> str:
    if row.latency_p50_ms is None or row.latency_p95_ms is None:
        return _MISSING
    return f"{row.latency_p50_ms:.0f} / {row.latency_p95_ms:.0f} (n={row.n_latency})"


def _cost(row: GenerationBaselineEntry) -> str:
    if row.cost_per_1k_usd is None:
        return _MISSING
    return f"${row.cost_per_1k_usd:.4f} (n={row.n_cost})"


# --- Shared pieces -------------------------------------------------------------------------------


def _ordered[T](rows: Mapping[str, T], first: Sequence[str]) -> list[str]:
    """The configs of ``first`` that exist, in that order, then the others by name."""
    return [name for name in first if name in rows] + sorted(set(rows) - set(first))


def _number(value: float | None, n: int | None = None) -> str:
    """A metric as ``0.912`` (retrieval: ``n`` is a column of its own) or ``0.912 (n=24)``."""
    if value is None:
        return _MISSING
    return f"{value:.3f}" if n is None else f"{value:.3f} (n={n})"


def _target_label(target: Target | None) -> str:
    return _MISSING if target is None else target.label


def _verdict(target: Target | None, measured: float | None) -> str:
    if target is None or measured is None:
        return _MISSING
    return "met" if target.met(measured) else "not met"


def _unique(values: Iterable[str]) -> str:
    """The distinct values, sorted and joined: one value when every row agrees, all of them (a
    sign that the rows come from different setups) when they do not."""
    return " | ".join(sorted(set(values)))


def _git_and_date(rows: Sequence[RetrievalBaselineEntry | GenerationBaselineEntry]) -> str:
    git = _unique(
        f"{'unknown' if r.git_sha is None else r.git_sha[:7]}{' (dirty)' if r.git_dirty else ''}"
        for r in rows
    )
    return f"git {git}, {_unique(_day(r.date) for r in rows)}"


def _day(moment: datetime) -> str:
    return moment.astimezone(UTC).date().isoformat()


def _markdown_table(rows: Sequence[Sequence[str]]) -> list[str]:
    """Header row first. The first column is left-aligned, the others right-aligned."""
    width = len(rows[0])
    lines = ["| " + " | ".join(rows[0]) + " |", "|---|" + "---:|" * (width - 1)]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return lines
