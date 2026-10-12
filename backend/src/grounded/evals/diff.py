"""``grounded eval diff``: two retrieval configs compared question by question (ticket 6.05).

Reads the results files of ``grounded eval retrieval`` (a baseline has no per-question rows, so it
cannot be diffed) and renders, as Markdown, what the lift analysis of Phase 6 needs: the mean
metrics of both configs with ``n``, the mean paired delta, how many questions each config wins on
(by nDCG@5), and for every question the rank of each grade-2 label in both configs. Nothing is
rescored: the per-question metrics are the ones the run stored, so the table and the results file
cannot disagree. Pure functions over the parsed files; the only inputs besides them are the golden
set (for the grades, which the results file does not keep) and its sha256.

**It refuses to compare what is not comparable** (``DiffError``, every reason listed), like the
baseline writers do: runs on another golden-set file or another index, other retrieval settings,
questions that differ, numbers that do not add up (an edited file), and, for ``hybrid`` against
``hybrid_rerank``, a reranked list that is not the same candidates in another order or that is the
hybrid order for every question (a reranker that did not run). A comparison that quietly mixed
setups would read as a lift or a loss that is not one (AGENTS.md §7).

It prints no clock and no path, so the same files always give the same text. It computes no cost
and no latency: the rerank call counts and the live latency are printed only when the results file
carries them (``RerankRunStats``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from grounded.evals.retrieval_runner import metric_names
from grounded.schemas.eval import (
    GoldenItem,
    RerankRunStats,
    RetrievalConfigResult,
    RetrievalRun,
    RetrievalRunInfo,
)

PLAIN_CONFIG: Final = "hybrid"
RERANKED_CONFIG: Final = "hybrid_rerank"
# The PRD §8 lift is nDCG@5(hybrid_rerank) - nDCG@5(hybrid): the metric that decides win, loss
# and tie.
HEADLINE_METRIC: Final = "ndcg@5"

# A tie is float noise and nothing else: two nDCG@5 values closer than this are the same number
# (the means of the reports use the same absorption). Any real difference, however small, is a win
# or a loss, and the honesty note says when it is below one question's worth. The Markdown
# names this value as 1e-9: change both together.
_EPSILON: Final = 1e-9

# What the search itself is made of. Two configs may differ in the mode and in the reranker, and in
# nothing else: other K or RRF values would change which candidates exist.
_NOT_SEARCH_SETTINGS: Final = frozenset({"mode", "rerank_provider", "rerank_model"})

_DOCS_PREFIX: Final = "docs/en/docs/"
_LABELS: Final = {
    "recall@5": "Recall@5",
    "recall@10": "Recall@10",
    "mrr": "MRR",
    "ndcg@5": "nDCG@5",
    "ndcg@10": "nDCG@10",
}


class DiffError(Exception):
    """The runs cannot be compared. ``problems`` lists every reason found, not only the first."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class LabelRank:
    """A grade-2 label and its rank (1-based, ``None`` = not retrieved) in each config."""

    label: str
    base: int | None
    other: int | None


@dataclass(frozen=True)
class QuestionDiff:
    id: str
    type: str
    answers: tuple[LabelRank, ...]  # the grade-2 labels, in the golden set's order
    deltas: Mapping[str, float]  # metric -> other - base, for every metric of ``metric_names``


@dataclass(frozen=True)
class Diff:
    """Everything the Markdown shows, already computed: ``render`` only formats it."""

    base: str  # config names; every delta is ``other - base``
    other: str
    infos: tuple[RetrievalRunInfo, ...]  # one per results file (one or two)
    n: int
    skipped_unanswerable: int
    k_base: int
    k_other: int
    base_means: Mapping[str, float]
    other_means: Mapping[str, float]
    mean_deltas: Mapping[str, float]  # mean of the paired deltas
    wins: int  # questions where ``other`` has the higher nDCG@5
    losses: int
    ties: int
    questions: tuple[QuestionDiff, ...]  # losses first: by headline delta, then id
    rerank: tuple[tuple[str, RerankRunStats | None], ...]  # a config that reranked, and its stats


def compare(
    runs: Sequence[RetrievalRun],
    configs: Sequence[str],
    *,
    golden: Sequence[GoldenItem],
    golden_sha256: str,
) -> Diff:
    """Diff ``configs[1]`` against ``configs[0]``.

    One run holds both configs (``eval retrieval --config hybrid --config hybrid_rerank``); two
    runs hold one each, the first config looked up in the first run and the second in the second.
    ``golden`` and its ``golden_sha256`` give the grade of every label; the hash must be the one the
    runs recorded, or the grades could belong to another set of questions.
    """
    if len(configs) != 2 or configs[0] == configs[1]:
        raise DiffError([f"two different configs are needed, got {list(configs)}"])
    if len(runs) not in (1, 2):
        raise DiffError(
            [
                f"give one results file holding both configs, or two files (one per config), "
                f"got {len(runs)}"
            ]
        )
    sources = [runs[0], runs[0]] if len(runs) == 1 else [runs[0], runs[1]]
    found: list[RetrievalConfigResult] = []
    missing: list[str] = []
    for position, (name, run) in enumerate(zip(configs, sources, strict=True), start=1):
        result = run.configs.get(name)
        if result is None:
            where = "the results file" if len(runs) == 1 else f"results file {position}"
            holds = ", ".join(run.configs) or "no config"
            missing.append(f"config {name!r} is not in {where} (it has: {holds})")
        else:
            found.append(result)
    if missing:
        raise DiffError(missing)
    [base, other] = found
    infos = tuple(run.info for run in runs)

    problems = _problems(base, other, infos, golden, golden_sha256)
    if problems:
        raise DiffError(problems)
    return _build(base, other, infos, golden)


# --- Refusals ------------------------------------------------------------------------------------


def _problems(
    base: RetrievalConfigResult,
    other: RetrievalConfigResult,
    infos: Sequence[RetrievalRunInfo],
    golden: Sequence[GoldenItem],
    golden_sha256: str,
) -> list[str]:
    problems: list[str] = []
    problems += _setup_problems(infos, golden_sha256)
    for result in (base, other):
        problems += _own_problems(result)
    problems += _pair_problems(base, other)
    problems += _golden_problems(base, other, golden)
    pair = {base.config, other.config}
    if pair == {PLAIN_CONFIG, RERANKED_CONFIG}:
        problems += _rerank_problems(base, other)
    return problems


def _setup_problems(infos: Sequence[RetrievalRunInfo], golden_sha256: str) -> list[str]:
    problems: list[str] = []
    if len(infos) == 2:
        first, second = infos
        for field, a, b in (
            ("golden_set_version", first.golden_set_version, second.golden_set_version),
            ("golden_set_sha256", first.golden_set_sha256, second.golden_set_sha256),
            ("index_config_hash", first.index_config_hash, second.index_config_hash),
        ):
            if a != b:
                problems.append(
                    f"the two files were scored on different data: {field} is {_short(a)} in "
                    f"file 1 and {_short(b)} in file 2"
                )
    for sha in sorted({info.golden_set_sha256 for info in infos} - {golden_sha256}):
        problems.append(
            f"the golden-set file (sha256 {_short(golden_sha256)}) is not the one the run scored "
            f"(sha256 {_short(sha)}); pass that file with --golden"
        )
    return problems


def _own_problems(result: RetrievalConfigResult) -> list[str]:
    """A results file is read back as data: check that its rows add up before trusting them."""
    name = result.config
    problems: list[str] = []
    ids = [q.id for q in result.questions]
    if len(set(ids)) != len(ids):
        problems.append(f"{name}: a question id appears more than once")
    if result.n != len(result.questions):
        problems.append(f"{name}: n is {result.n} but the file has {len(result.questions)} rows")
    wanted = metric_names()
    incomplete = sorted(q.id for q in result.questions if not set(wanted) <= set(q.metrics))
    if incomplete:
        problems.append(f"{name}: questions without every metric: {_some(incomplete)}")
        return problems
    for metric in wanted:
        if not result.questions:
            break
        mean = math.fsum(q.metrics[metric] for q in result.questions) / len(result.questions)
        stored = result.metrics.get(metric)
        if stored is None or abs(mean - stored) > _EPSILON:
            problems.append(
                f"{name}: the mean {metric} in the file ({stored}) is not the mean of its "
                f"per-question rows ({mean:.6f}); the file was edited or is damaged"
            )
    return problems


def _pair_problems(base: RetrievalConfigResult, other: RetrievalConfigResult) -> list[str]:
    problems: list[str] = []
    settings_base = base.retrieval_config.model_dump(exclude=set(_NOT_SEARCH_SETTINGS))
    settings_other = other.retrieval_config.model_dump(exclude=set(_NOT_SEARCH_SETTINGS))
    differing = sorted(key for key in settings_base if settings_base[key] != settings_other[key])
    if differing:
        parts = ", ".join(f"{k} {settings_base[k]} vs {settings_other[k]}" for k in differing)
        problems.append(f"the retrieval settings differ ({parts}): the candidates are not the same")
    ids_base = {q.id for q in base.questions}
    ids_other = {q.id for q in other.questions}
    if ids_base != ids_other:
        problems.append(
            f"the questions differ: only in {base.config}: {_some(sorted(ids_base - ids_other))}; "
            f"only in {other.config}: {_some(sorted(ids_other - ids_base))}"
        )
    return problems


def _golden_problems(
    base: RetrievalConfigResult, other: RetrievalConfigResult, golden: Sequence[GoldenItem]
) -> list[str]:
    """Every question is an answerable item of the golden set, and both results have all its
    labels (the grades come from the golden set, the ranks from the files)."""
    by_id = {item.id: item for item in golden}
    problems: list[str] = [
        f"{question.id} is not an answerable question of the golden set"
        for question in base.questions
        if (item := by_id.get(question.id)) is None or not item.answerable
    ]
    for result in (base, other):
        for question in result.questions:
            item = by_id.get(question.id)
            if item is None or not item.answerable:
                continue
            absent = sorted(label for label in item.relevant if label not in question.ranks)
            if absent:
                problems.append(
                    f"{result.config} {question.id}: labels of the golden set missing in the "
                    f"file: {absent}"
                )
    return problems


def _rerank_problems(base: RetrievalConfigResult, other: RetrievalConfigResult) -> list[str]:
    """``hybrid`` against ``hybrid_rerank``: the second must be the first, reordered by a reranker.

    A reranker error ends the eval run without a results file (Tech §8), so a degraded run cannot
    come out of the tool; these checks catch a file that did not come from it (an edited file, a
    reranked list that is not the hybrid candidates, a run with no reranker behind it).
    """
    reranked, plain = (base, other) if base.config == RERANKED_CONFIG else (other, base)
    problems: list[str] = []
    if plain.retrieval_config.rerank_provider != "none":
        problems.append(f"{plain.config}: the config says it used a reranker, so it is not plain")
    if reranked.k != plain.k:
        problems.append(f"k differs ({plain.config} {plain.k}, {reranked.config} {reranked.k})")
    stats = reranked.rerank
    if stats is not None:
        if stats.live_calls + stats.cache_hits == 0:
            problems.append(f"{reranked.config}: the run recorded no rerank call at all")
        if stats.model != reranked.retrieval_config.rerank_model:
            problems.append(
                f"{reranked.config}: the rerank model of the calls ({stats.model}) is not the one "
                f"of its config ({reranked.retrieval_config.rerank_model})"
            )
    by_id = {q.id: q for q in plain.questions}
    changed = 0
    different: list[str] = []
    for question in reranked.questions:
        before = by_id.get(question.id)
        if before is None:
            continue  # reported by _pair_problems
        if Counter(before.retrieved) != Counter(question.retrieved):
            different.append(question.id)
        if before.retrieved != question.retrieved:
            changed += 1
    if different:
        problems.append(
            f"{reranked.config} does not rank the same candidates as {plain.config} "
            f"({_some(sorted(different))}): a reranker only reorders them"
        )
    if len(reranked.questions) > 1 and not different and changed == 0:
        problems.append(
            f"{reranked.config} has the {plain.config} order for every question: the reranker "
            "did not run, or its answer was not applied (a degraded run is not a result)"
        )
    return problems


# --- The numbers ---------------------------------------------------------------------------------


def _build(
    base: RetrievalConfigResult,
    other: RetrievalConfigResult,
    infos: tuple[RetrievalRunInfo, ...],
    golden: Sequence[GoldenItem],
) -> Diff:
    by_id = {item.id: item for item in golden}
    other_by_id = {q.id: q for q in other.questions}
    names = metric_names()
    questions: list[QuestionDiff] = []
    for before in base.questions:
        after = other_by_id[before.id]
        item = by_id[before.id]
        questions.append(
            QuestionDiff(
                id=before.id,
                type=before.type,
                answers=tuple(
                    LabelRank(
                        section.section, before.ranks[section.section], after.ranks[section.section]
                    )
                    for section in item.relevant_sections
                    if section.grade == 2
                ),
                deltas={m: after.metrics[m] - before.metrics[m] for m in names},
            )
        )
    questions.sort(key=lambda q: (q.deltas[HEADLINE_METRIC], q.id))
    n = len(questions)
    headline = [q.deltas[HEADLINE_METRIC] for q in questions]
    return Diff(
        base=base.config,
        other=other.config,
        infos=infos,
        n=n,
        skipped_unanswerable=base.skipped_unanswerable,
        k_base=base.k,
        k_other=other.k,
        base_means={m: base.metrics[m] for m in names},
        other_means={m: other.metrics[m] for m in names},
        mean_deltas={m: math.fsum(q.deltas[m] for q in questions) / n for m in names},
        wins=sum(1 for d in headline if d > _EPSILON),
        losses=sum(1 for d in headline if d < -_EPSILON),
        ties=sum(1 for d in headline if abs(d) <= _EPSILON),
        questions=tuple(questions),
        rerank=tuple(
            (result.config, result.rerank)
            for result in (base, other)
            if result.rerank is not None or result.retrieval_config.rerank_provider != "none"
        ),
    )


# --- Markdown ------------------------------------------------------------------------------------


def render(diff: Diff) -> str:
    """The comparison as Markdown, ending in a newline: a pure function of ``diff``."""
    base, other = diff.base, diff.other
    sections = [
        f"# `{other}` against `{base}`",
        _intro(diff),
        _means_section(diff),
        _outcome_section(diff),
        _rerank_section(diff),
        _questions_section(diff),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


def _intro(diff: Diff) -> str:
    infos = diff.infos
    golden = _unique(
        f"{i.golden_set_version} (sha256 {_short(i.golden_set_sha256)})" for i in infos
    )
    index = _unique(f"{i.fastapi_ref}@{_short(i.index_config_hash)}" for i in infos)
    embedding = _unique(f"{i.embedding_model} ({i.embedding_dim} dims)" for i in infos)
    git = _unique(
        f"{'unknown' if i.git_sha is None else i.git_sha[:7]}{' (dirty)' if i.git_dirty else ''}"
        for i in infos
    )
    k = (
        f"k = {diff.k_base}"
        if diff.k_base == diff.k_other
        else f"k = {diff.k_base} ({diff.base}), {diff.k_other} ({diff.other})"
    )
    files = "one results file" if len(infos) == 1 else "two results files"
    return (
        f"Paired over the same {diff.n} answerable questions ({diff.skipped_unanswerable} "
        f"unanswerable ones are not scored), from {files}: golden set {golden}, index {index}, "
        f"embedding {embedding}, {k}, git {git}. Every delta is `{diff.other}` minus "
        f"`{diff.base}`, computed from the unrounded numbers of the results file."
    )


def _means_section(diff: Diff) -> str:
    rows = [["Metric", "n", diff.base, diff.other, "Delta"]]
    for metric in metric_names():
        rows.append(
            [
                _LABELS.get(metric, metric),
                str(diff.n),
                _number(diff.base_means[metric]),
                _number(diff.other_means[metric]),
                _signed(diff.mean_deltas[metric]),
            ]
        )
    return "\n".join(["## Mean metrics", "", *_table(rows)])


def _outcome_section(diff: Diff) -> str:
    one = 1 / diff.n  # the most one question can move a mean of values in [0, 1] (AGENTS.md §7)
    delta = diff.mean_deltas[HEADLINE_METRIC]
    size = "smaller than" if abs(delta) < one - _EPSILON else "at least"
    label = _LABELS[HEADLINE_METRIC]
    return "\n".join(
        [
            f"## Per-question outcome by {label}",
            "",
            f"`{diff.other}` is better on {diff.wins} questions, `{diff.base}` is better on "
            f"{diff.losses}, and they tie on {diff.ties} (n = {diff.n}). A tie is a difference "
            f"of at most 1e-9 in {label} (float noise); any other difference is a win or a "
            "loss, however small.",
            "",
            f"**One question's worth.** With n = {diff.n}, one question moves a mean metric by at "
            f"most 1/{diff.n} = {one:.3f} (a per-question metric lies in [0, 1]). A difference of "
            "means below that is not reported as a lift or a loss (AGENTS.md §7). This is a paired "
            f"comparison on {diff.n} questions, not a significance test. The {label} delta, "
            f"{_signed(delta)}, is {size} one question's worth.",
        ]
    )


def _rerank_section(diff: Diff) -> str:
    lines: list[str] = []
    for name, stats in diff.rerank:
        if stats is None:
            lines.append(f"`{name}`: the results file has no rerank call counts or latency.")
            continue
        latency = (
            f"live latency p50 {stats.latency_p50_ms:.0f} / p95 {stats.latency_p95_ms:.0f} ms"
            if stats.latency_p50_ms is not None and stats.latency_p95_ms is not None
            else "no live call, so no latency"
        )
        lines.append(
            f"`{name}` reranked with {stats.model}: {stats.live_calls} live calls, "
            f"{stats.cache_hits} answered from the rerank cache; {latency}."
        )
    if not lines:
        return ""
    return "\n".join(["## Rerank calls", "", *lines])


def _questions_section(diff: Diff) -> str:
    base, other = diff.base, diff.other
    head = [
        "Question",
        "Type",
        *(f"Δ {_LABELS.get(m, m)}" for m in metric_names()),
        f"Grade-2 label ranks, {base} -> {other}",
    ]
    rows = [head]
    for question in diff.questions:
        rows.append(
            [
                question.id,
                question.type,
                *(_signed(question.deltas[m]) for m in metric_names()),
                "; ".join(
                    f"`{_short_label(a.label)}`: {_rank(a.base)} -> {_rank(a.other)}"
                    for a in question.answers
                ),
            ]
        )
    return "\n".join(
        [
            "## Per question",
            "",
            f"Sorted by the {_LABELS[HEADLINE_METRIC]} delta (losses first), then by question id. "
            "A rank is the 1-based position of the first chunk that matches the label; `null` "
            "means the label was not retrieved. Labels are listed in the golden set's order.",
            "",
            *_table(rows, left={0, 1, len(head) - 1}),
        ]
    )


def _table(rows: Sequence[Sequence[str]], left: Collection[int] = (0,)) -> list[str]:
    """Header row first; the columns in ``left`` are left-aligned (text), the others right-aligned
    (numbers)."""
    divider = "|" + "".join("---|" if i in left else "---:|" for i in range(len(rows[0])))
    lines = ["| " + " | ".join(rows[0]) + " |", divider]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return lines


def _number(value: float) -> str:
    return f"{value:.3f}"


def _signed(value: float) -> str:
    """``+0.060``, ``-0.040`` and ``0.000`` for a tie (so float noise never prints a sign)."""
    return "0.000" if abs(value) <= _EPSILON else f"{value:+.3f}"


def _rank(rank: int | None) -> str:
    return "null" if rank is None else str(rank)


def _short_label(label: str) -> str:
    return label.removeprefix(_DOCS_PREFIX)


def _short(value: str) -> str:
    return value[:8] if len(value) > 12 else value


def _unique(values: Iterable[str]) -> str:
    """The distinct values, sorted and joined: one when the files agree."""
    return " | ".join(sorted(set(values)))


def _some(ids: Sequence[str], limit: int = 5) -> str:
    """Up to ``limit`` ids, then how many more: a message names the culprits, not all of them."""
    if not ids:
        return "none"
    shown = ", ".join(ids[:limit])
    return shown if len(ids) <= limit else f"{shown} and {len(ids) - limit} more"
