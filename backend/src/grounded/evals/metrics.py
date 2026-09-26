"""Retrieval metrics: Recall@k, MRR, nDCG@k (Tech.md §15.2). **Author-owned module** (AGENTS.md §3).

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
  runner, not scored as 0), and ``k`` must be ≥ 1. Otherwise: ``ValueError``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Protocol


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
    raise NotImplementedError("metrics are Author-owned: see the spec tests")


def label_ranks(retrieved: Sequence[ChunkRef], labels: Collection[str]) -> dict[str, int]:
    """Map each label to the 1-based rank of the first chunk in ``retrieved`` that matches it.

    A chunk's source path is its ``section_id`` up to the ``#``. Labels no chunk matches are left
    out. Nested labels (one an ancestor of the other: a page and a section on it, an H2 and one of
    its H3s) are not allowed in the golden set, so a single chunk matching two labels means bad
    input: ``ValueError``. That also guarantees distinct ranks, which keeps nDCG ≤ 1.
    """
    raise NotImplementedError("metrics are Author-owned: see the spec tests")


def recall_at_k(ranks: Mapping[str, int], relevant: Mapping[str, int], k: int) -> float:
    """Share of the grade-2 labels ranked within the top ``k``. Grade-1 labels don't count."""
    raise NotImplementedError("metrics are Author-owned: see the spec tests")


def mrr(ranks: Mapping[str, int], relevant: Mapping[str, int]) -> float:
    """Reciprocal rank of the best-ranked grade-2 label; 0.0 if none was retrieved.

    No cutoff: the whole retrieved list counts (``K_DENSE`` for dense, ``K_FUSED`` for hybrid).
    """
    raise NotImplementedError("metrics are Author-owned: see the spec tests")


def ndcg_at_k(ranks: Mapping[str, int], relevant: Mapping[str, int], k: int) -> float:
    """Normalized DCG over the labels ranked within the top ``k``.

    - gain of a label = ``2**grade - 1`` (grade 2 → 3, grade 1 → 1), discounted by
      ``log2(rank + 1)``;
    - ideal DCG: all labels of ``relevant`` (both grades) sorted by grade, best first, at ranks
      1, 2, …, keeping only the first ``min(k, len(relevant))``;
    - result = DCG / ideal DCG, in [0, 1]. ``ranks`` with a repeated value: ``ValueError``.
    """
    raise NotImplementedError("metrics are Author-owned: see the spec tests")
