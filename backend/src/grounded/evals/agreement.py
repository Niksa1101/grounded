"""Judge-human agreement (Tech.md §15.4, PRD FR-22 and §8, ticket 4.11): the labeling sheet's file
format, Cohen's kappa and the report ``grounded eval agreement`` prints.

Everything here is a pure function of text and numbers; the files are read by the CLI. The sheet is
the CSV the Author fills in (``human_label``); the key (``schemas/judge_agreement.py``) says what
each ``item_id`` is and what the judge said. The two are joined by ``item_id``, and the join
**refuses** a sheet with a missing, invalid or unknown label instead of dropping the row: a
silently smaller ``n`` would change the number it reports.

**Cohen's kappa** (unweighted) is ``(p_o - p_e) / (1 - p_e)``: ``p_o`` the share of items on which
the two raters gave the same label, ``p_e`` the agreement expected if each rater picked labels at
random with its own label frequencies, ``sum_k (r_k * c_k) / n^2`` for ``r_k`` and ``c_k`` the
number of items each rater gave label ``k``. It has no value when ``p_e`` is 1 (both raters used
one and the same label for every item): the formula divides by zero and the data say nothing about
agreement beyond chance. That case is reported as undefined with its reason, never as 0 or 1. The
arithmetic is done in exact fractions, so "``p_e`` is 1" is an exact test, not a float comparison.
"""

from __future__ import annotations

import csv
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from grounded.evals.retrieval_runner import EVAL_DIR, RESULTS_DIR
from grounded.schemas.judge_agreement import ALLOWED_LABELS, AgreementKey, ItemSource, Kind

# --- The sheet (eval/judge_agreement/v1.csv) -----------------------------------------------------

AGREEMENT_DIR: Final = EVAL_DIR / "judge_agreement"
DEFAULT_SHEET: Final = AGREEMENT_DIR / "v1.csv"
DEFAULT_VIEW: Final = AGREEMENT_DIR / "v1.md"
# The key is not committed with the sheet, so that the labels are made blind; 4.11b copies it next
# to the labels once they are in.
DEFAULT_KEY: Final = RESULTS_DIR / "judge-agreement-v1.key.json"

SHEET_COLUMNS: Final = (
    "item_id",
    "kind",
    "question_id",
    "question",
    "claim",
    "sources",
    "reference_answer",
    "candidate_answer",
    "allowed_labels",
    "human_label",
)

# PRD §8: judge-human agreement on at least 10 verdicts, 0.8 desired.
MIN_ITEMS: Final = 10
TARGET: Final = 0.8

_BOM = b"\xef\xbb\xbf"  # Excel opens a UTF-8 CSV as UTF-8 only with this mark
_DELIMITERS = ",;\t"  # a comma-separated file saved by Excel in a Serbian locale uses ";"


class SheetError(ValueError):
    """The file is not a labeling sheet this module can read; the message says why."""


class LabelError(ValueError):
    """The labels cannot be joined to the key. ``problems`` has one line per item, all of them."""

    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = tuple(problems)


def render_sheet(rows: Sequence[Mapping[str, str]]) -> bytes:
    """UTF-8 with a BOM, LF row ends, a quoted cell wherever the text has a comma, a quote or a
    newline (the evidence has all three). Deterministic: the same rows are the same bytes."""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {c: row.get(c, "").replace("\r\n", "\n").replace("\r", "\n") for c in SHEET_COLUMNS}
        )
    return _BOM + out.getvalue().encode("utf-8")


def parse_sheet(data: bytes) -> list[dict[str, str]]:
    """The rows of a sheet, with or without the BOM, comma-, semicolon- or tab-separated (read off
    the header line). Blank rows are ignored. Only the labels' columns have to survive a round trip
    through a spreadsheet program: ``item_id``, ``kind`` and ``human_label``."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SheetError("the file is not UTF-8 (in Excel: Save As, 'CSV UTF-8')") from exc
    delimiter = max(_DELIMITERS, key=text.partition("\n")[0].count)
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
        missing = [c for c in SHEET_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise SheetError(f"the header lacks the columns {', '.join(missing)}")
        rows = [{c: row.get(c) or "" for c in SHEET_COLUMNS} for row in reader]
    except csv.Error as exc:
        raise SheetError(f"not a readable CSV: {exc}") from exc
    return [row for row in rows if any(value.strip() for value in row.values())]


# --- Cohen's kappa -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Kappa:
    value: float | None  # None: undefined, ``reason`` says why
    reason: str | None
    observed: float  # p_o
    expected: float  # p_e


def cohens_kappa(human: Sequence[str], judge: Sequence[str]) -> Kappa:
    """Unweighted Cohen's kappa of two raters' labels for the same items, in the same order."""
    if len(human) != len(judge) or not human:
        raise ValueError("kappa needs the same, non-zero number of labels from both raters")
    n = len(human)
    observed = Fraction(sum(h == j for h, j in zip(human, judge, strict=True)), n)
    human_counts, judge_counts = Counter(human), Counter(judge)
    expected = sum(Fraction(human_counts[k] * judge_counts[k], n * n) for k in human_counts)
    if expected == 1:
        (label,) = human_counts
        return Kappa(
            None,
            f"both raters gave every item the label {label}, so agreement by chance is 100%",
            float(observed),
            1.0,
        )
    return Kappa(
        float((observed - expected) / (1 - expected)), None, float(observed), float(expected)
    )


# --- Joining the labels to the key ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rating:
    """One item, rated by both."""

    item_id: str
    kind: Kind
    source: ItemSource
    human: str
    judge: str


def normalize_label(text: str) -> str:
    """``" not supported "`` is ``NOT_SUPPORTED``: case, surrounding space and the separator a
    person types are not mistakes worth refusing a sheet for."""
    return "_".join(text.upper().replace("-", " ").split())


def rate_items(sheet: Sequence[Mapping[str, str]], key: AgreementKey) -> list[Rating]:
    """Every key item with its human label and its judge verdict, in key order.

    Raises ``LabelError`` listing every item that has no row, no label, a label its kind does not
    allow, a kind that differs from the key's, or no judge verdict in the key; and every row the key
    does not know. Nothing is skipped.
    """
    problems: list[str] = []
    rows: dict[str, Mapping[str, str]] = {}
    for row in sheet:
        item_id = row["item_id"].strip()
        if item_id in rows:
            problems.append(f"{item_id}: appears twice in the sheet")
        rows[item_id] = row
    problems += [
        f"{i}: is in the sheet but not in the key"
        for i in sorted(set(rows) - {x.item_id for x in key.items})
    ]

    ratings: list[Rating] = []
    for item in key.items:
        row = rows.get(item.item_id)
        if row is None:
            problems.append(f"{item.item_id}: has no row in the sheet")
            continue
        if row["kind"].strip() != item.kind:
            problems.append(
                f"{item.item_id}: the sheet says kind {row['kind']!r}, the key {item.kind!r}"
            )
        label = normalize_label(row["human_label"])
        allowed = ALLOWED_LABELS[item.kind]
        if not label:
            problems.append(f"{item.item_id}: no human_label")
        elif label not in allowed:
            problems.append(
                f"{item.item_id}: {row['human_label']!r} is not one of {', '.join(allowed)}"
            )
        if item.judge is None:
            problems.append(
                f"{item.item_id}: the key has no judge verdict yet (re-run export-verdicts)"
            )
        if label in allowed and item.judge is not None:
            ratings.append(Rating(item.item_id, item.kind, item.source, label, item.judge.verdict))
    if problems:
        raise LabelError(problems)
    return ratings


# --- Summaries and the report --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Summary:
    n: int
    agree: int
    kappa: Kappa | None  # None: no items
    confusion: Mapping[tuple[str, str], int]  # (human label, judge label) -> items
    labels: tuple[str, ...]

    @property
    def rate(self) -> float | None:
        return self.agree / self.n if self.n else None


def summarize(ratings: Sequence[Rating], labels: Sequence[str]) -> Summary:
    human, judge = [r.human for r in ratings], [r.judge for r in ratings]
    return Summary(
        n=len(ratings),
        agree=sum(h == j for h, j in zip(human, judge, strict=True)),
        kappa=cohens_kappa(human, judge) if ratings else None,
        confusion=dict(Counter(zip(human, judge, strict=True))),
        labels=tuple(labels),
    )


def render_report(ratings: Sequence[Rating], key: AgreementKey) -> str:
    """The Markdown ``grounded eval agreement`` prints: every number with its ``n``, kappa as
    undefined (with the reason) where it has no value, the PRD target next to the numbers. It
    states whether the target is met and decides nothing else."""
    faith, corr = ALLOWED_LABELS["faithfulness"], ALLOWED_LABELS["correctness"]

    def of(kind: Kind | None = None, source: ItemSource | None = None) -> list[Rating]:
        return [r for r in ratings if kind in (None, r.kind) and source in (None, r.source)]

    scopes = {
        "all": ("All items (labels of both kinds pooled)", summarize(of(), faith + corr)),
        "faithfulness": ("Faithfulness", summarize(of("faithfulness"), faith)),
        "correctness": ("Correctness", summarize(of("correctness"), corr)),
        "real": ("Real items only (pooled)", summarize(of(source="real"), faith + corr)),
        "faithfulness_real": (
            "Faithfulness, real items only",
            summarize(of("faithfulness", "real"), faith),
        ),
        "controls": ("Controls only (synthetic)", summarize(of(source="control"), faith)),
    }
    sample = key.sample
    lines = [
        "# Judge-human agreement",
        "",
        f"Sample: seed {sample.seed}, {len(ratings)} items from the run {sample.results_file} "
        f"(sha256 {sample.results_sha256[:12]}, golden set {sample.golden_set_version}). "
        f"Judge: {sample.judge_provider} {sample.judge_model}, rubrics "
        + ", ".join(f"{v}" for _, v in sorted(sample.judge_prompt_versions.items()))
        + ".",
        "",
        "| Scope | n | Exact agreement | Cohen's kappa |",
        "|---|---:|---:|---:|",
    ]
    notes: list[str] = []
    for name, s in scopes.values():
        if s.n == 0:
            lines.append(f"| {name} | 0 | n/a | n/a |")
            continue
        kappa = "undefined" if s.kappa is None or s.kappa.value is None else f"{s.kappa.value:.3f}"
        if s.kappa is not None and s.kappa.value is None:
            notes.append(f"Kappa of '{name}' is undefined: {s.kappa.reason}.")
        lines.append(f"| {name} | {s.n} | {100 * s.agree / s.n:.1f}% ({s.agree}/{s.n}) | {kappa} |")
    lines += ["", *notes] if notes else []

    lines += ["", "## PRD §8 target", ""]
    all_items, real = scopes["all"][1], scopes["real"][1]
    lines += [
        f"PRD §8 asks for agreement on at least {MIN_ITEMS} verdicts, {TARGET} desired. It does "
        "not say whether that is the exact agreement or kappa, so both are shown against it.",
        "",
        "| Measure | Value | n | >= 0.8 |",
        "|---|---:|---:|---|",
    ]
    for name, s in (("all items", all_items), ("real items only", real)):
        if s.n == 0 or s.rate is None:
            continue
        lines.append(f"| Exact agreement, {name} | {s.rate:.3f} | {s.n} | {_met(s.rate)} |")
        if s.kappa is not None and s.kappa.value is not None:
            lines.append(
                f"| Cohen's kappa, {name} | {s.kappa.value:.3f} | {s.n} | {_met(s.kappa.value)} |"
            )
        else:
            lines.append(f"| Cohen's kappa, {name} | undefined | {s.n} | n/a |")
    lines += [
        "",
        f"At least {MIN_ITEMS} items labeled: {'yes' if all_items.n >= MIN_ITEMS else 'no'} "
        f"(n = {all_items.n}).",
    ]

    lines += ["", "## Confusion matrices (rows: human, columns: judge)"]
    for scope in ("faithfulness", "correctness", "faithfulness_real"):
        name, s = scopes[scope]
        if s.n:
            lines += ["", f"### {name} (n = {s.n})", "", *_matrix(s)]

    wrong = [r for r in ratings if r.human != r.judge]
    lines += ["", f"## Disagreements (n = {len(wrong)} of {len(ratings)})", ""]
    lines += [
        f"- {r.item_id} ({r.kind}, {r.source}): human {r.human}, judge {r.judge}" for r in wrong
    ] or ["None."]
    return "\n".join(lines) + "\n"


def _met(value: float) -> str:
    return "met" if value >= TARGET else "not met"


def _matrix(summary: Summary) -> list[str]:
    header = "| human \\ judge | " + " | ".join(summary.labels) + " | total |"
    rows = [header, "|---|" + "---:|" * (len(summary.labels) + 1)]
    for human in summary.labels:
        counts = [summary.confusion.get((human, judge), 0) for judge in summary.labels]
        rows.append(f"| {human} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    totals = [
        sum(summary.confusion.get((h, judge), 0) for h in summary.labels)
        for judge in summary.labels
    ]
    rows.append("| total | " + " | ".join(map(str, totals)) + f" | {summary.n} |")
    return rows
