"""Retrieval metrics: Recall@k, MRR, nDCG@k (Tech.md §15.2). **Author-owned module** (AGENTS.md §3),
written by the Agent at the Author's explicit request (2026-09-26).

Spec tests: ``tests/unit/test_metrics.py``.

Pure functions over one golden-set question. The runner (``retrieval_runner.py``) calls
``label_ranks`` once per question, feeds the result to the three metrics and averages them with
``n``. Labels are ``"<source_path>"`` (the whole page) or ``"<source_path>#<anchor>"``; ``relevant``
maps each label of the question to its grade: 2 = contains the answer, 1 = useful context.

Rules shared by the metrics:

- A label's **rank** is the 1-based position of the first *chunk* that matches it, so every chunk
  takes a position, including a second part of a section that is already ranked. Unmatched labels
  are absent from ``ranks`` and count as not retrieved.
- ``k`` larger than the retrieved list is fine: the missing positions are simply not relevant.
- ``relevant`` must hold at least one grade-2 label (unanswerable questions are skipped by the
  runner, not scored as 0), grades must be 1 or 2, and ``k`` must be ≥ 1. Otherwise:
  ``ValueError``.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

_GRADES = frozenset({1, 2})


class ChunkRef(Protocol):
    """What the metrics need from a retrieved chunk (``RetrievedChunk`` satisfies it)."""

    @property
    def section_id(self) -> str: ...  # "<source_path>#<deepest anchor>" ("...md#" for an intro)

    @property
    def anchor_path(self) -> Sequence[str]: ...  # outermost first; () for a page intro


def section_matches(label: str, source_path: str, anchor_path: Sequence[str]) -> bool:
    """Whether a chunk of ``source_path`` with ``anchor_path`` matches ``label`` (Tech.md §15.1).

    ``"path#anchor"`` matches iff the paths are equal and ``anchor`` is one of the chunk's anchors,
    so an H2 label also matches the chunks of its H3 subsections (not the other way around).
    ``"path"`` matches every chunk of that page. Anchors compare as whole strings, never as
    substrings. A label with an empty anchor (``"path#"``) is malformed: ``ValueError``.
    """
    path, has_anchor, anchor = label.partition("#")
    if has_anchor and not anchor:
        raise ValueError(f"label {label!r} has an empty anchor: use the page path alone instead")
    if path != source_path:
        return False
    return not has_anchor or anchor in anchor_path


def label_ranks(retrieved: Sequence[ChunkRef], labels: Collection[str]) -> dict[str, int]:
    """Map each label to the 1-based rank of the first chunk in ``retrieved`` that matches it.

    A chunk's source path is its ``section_id`` up to the ``#``. Labels no chunk matches are left
    out. Nested labels (one an ancestor of the other: a page and a section on it, an H2 and one of
    its H3s) are not allowed in the golden set, so a single chunk matching two labels means bad
    input: ``ValueError``. That also guarantees distinct ranks, which keeps nDCG ≤ 1.
    """
    ranks: dict[str, int] = {}
    for rank, chunk in enumerate(retrieved, start=1):
        source_path = chunk.section_id.partition("#")[0]
        matched = [
            label for label in labels if section_matches(label, source_path, chunk.anchor_path)
        ]
        if len(matched) > 1:
            raise ValueError(
                f"nested labels {sorted(matched)} all match chunk {chunk.section_id!r}"
            )
        for label in matched:
            ranks.setdefault(label, rank)  # only the first matching chunk counts
    return ranks


def recall_at_k(ranks: Mapping[str, int], relevant: Mapping[str, int], k: int) -> float:
    """Share of the grade-2 labels ranked within the top ``k``. Grade-1 labels don't count."""
    _check_relevant(relevant)
    _check_k(k)
    answers = _grade_2(relevant)
    found = sum(1 for label in answers if ranks.get(label, k + 1) <= k)
    return found / len(answers)


def mrr(ranks: Mapping[str, int], relevant: Mapping[str, int]) -> float:
    """Reciprocal rank of the best-ranked grade-2 label; 0.0 if none was retrieved.

    No cutoff: the whole retrieved list counts (``K_DENSE`` for dense, ``K_FUSED`` for hybrid).
    """
    _check_relevant(relevant)
    best = min((ranks[label] for label in _grade_2(relevant) if label in ranks), default=None)
    return 0.0 if best is None else 1.0 / best


def ndcg_at_k(ranks: Mapping[str, int], relevant: Mapping[str, int], k: int) -> float:
    """Normalized DCG over the labels ranked within the top ``k``.

    - gain of a label = ``2**grade - 1`` (grade 2 → 3, grade 1 → 1), discounted by
      ``log2(rank + 1)``;
    - ideal DCG: all labels of ``relevant`` (both grades) sorted by grade, best first, at ranks
      1, 2, …, keeping only the first ``min(k, len(relevant))``;
    - result = DCG / ideal DCG, in [0, 1]. ``ranks`` with a repeated value: ``ValueError``.
    """
    _check_relevant(relevant)
    _check_k(k)
    if len(set(ranks.values())) != len(ranks):
        raise ValueError(f"every label needs its own rank, got {sorted(ranks.values())}")
    # fsum is exactly rounded, so the result doesn't depend on the order of ``ranks``.
    dcg = math.fsum(
        _gain(relevant[label]) / math.log2(rank + 1)
        for label, rank in ranks.items()
        if rank <= k and label in relevant
    )
    best_first = sorted(relevant.values(), reverse=True)[:k]
    ideal = math.fsum(
        _gain(grade) / math.log2(rank + 1) for rank, grade in enumerate(best_first, start=1)
    )
    return dcg / ideal


def _grade_2(relevant: Mapping[str, int]) -> list[str]:
    return [label for label, grade in relevant.items() if grade == 2]


def _gain(grade: int) -> float:
    return 2.0**grade - 1.0


def _check_relevant(relevant: Mapping[str, int]) -> None:
    if bad := sorted(label for label, grade in relevant.items() if grade not in _GRADES):
        raise ValueError(f"grades must be 1 or 2, got {[relevant[label] for label in bad]}")
    if not _grade_2(relevant):
        raise ValueError("a scored question needs at least one grade-2 label")


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
