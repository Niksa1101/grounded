"""Spec for the retrieval metrics (Tech.md §15.2; contract in ``grounded.evals.metrics``).

Written before the implementation (Author-owned). Expected values are computed by hand: the
nDCG ones are written as the formula (``log2`` of the rank + 1) with the decimal in a comment.
Gains: grade 2 → 2**2 - 1 = 3, grade 1 → 2**1 - 1 = 1.

Each ``ValueError`` test matches one word the message must contain ("anchor", "nested", "grade-2",
"rank", "k"), so the error says which rule was broken.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import log2

import pytest

from grounded.evals.metrics import label_ranks, mrr, ndcg_at_k, recall_at_k, section_matches

P = "docs/en/docs/tutorial/page.md"
Q = "docs/en/docs/tutorial/other.md"
R = "docs/en/docs/advanced/elsewhere.md"


@dataclass(frozen=True)
class C:
    """A retrieved chunk as the metrics see it (satisfies ``ChunkRef``)."""

    section_id: str
    anchor_path: tuple[str, ...]


def chunk(source_path: str, *anchors: str) -> C:
    """``chunk(P, "alpha", "beta")`` is the H3 chunk ``P#beta`` under H2 ``alpha``."""
    return C(section_id=f"{source_path}#{anchors[-1] if anchors else ''}", anchor_path=anchors)


# --- section_matches ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "source_path", "anchor_path", "expected"),
    [
        (f"{P}#alpha", P, ("alpha",), True),
        (f"{P}#alpha", P, ("alpha", "beta"), True),  # H2 label matches its H3 sub-chunk
        (f"{P}#beta", P, ("alpha", "beta"), True),
        (f"{P}#beta", P, ("alpha",), False),  # H3 label doesn't match its parent's chunk
        (f"{P}#alpha", P, (), False),  # nor the page intro
        (f"{P}#alpha", Q, ("alpha",), False),  # same anchor on another page
        (f"{P}#alpha", P, ("alpha-extra",), False),  # whole anchors, not substrings
        (f"{P}#alpha-extra", P, ("alpha",), False),
        (P, P, (), True),  # a page label matches the intro...
        (P, P, ("alpha", "beta"), True),  # ...and every section of the page
        (P, Q, (), False),
    ],
)
def test_section_matches(
    label: str, source_path: str, anchor_path: tuple[str, ...], expected: bool
) -> None:
    assert section_matches(label, source_path, anchor_path) is expected


def test_label_with_an_empty_anchor_is_malformed() -> None:
    with pytest.raises(ValueError, match="anchor"):
        section_matches(f"{P}#", P, ())


# --- label_ranks --------------------------------------------------------------------------------


def test_rank_is_the_position_of_the_first_matching_chunk() -> None:
    retrieved = [
        chunk(R, "z"),  # 1
        chunk(P, "alpha"),  # 2: first match of P#alpha
        chunk(P, "alpha"),  # 3: a second part of the same section still takes a position
        chunk(P, "gamma"),  # 4
    ]
    ranks = label_ranks(retrieved, [f"{P}#alpha", f"{P}#gamma", f"{P}#delta"])
    assert ranks == {f"{P}#alpha": 2, f"{P}#gamma": 4}  # delta never matched: absent


def test_h2_label_is_ranked_by_its_h3_chunk() -> None:
    retrieved = [chunk(R, "z"), chunk(P, "alpha", "beta")]
    assert label_ranks(retrieved, [f"{P}#alpha"]) == {f"{P}#alpha": 2}


def test_page_label_is_ranked_by_any_chunk_of_the_page() -> None:
    retrieved = [chunk(R, "z"), chunk(Q), chunk(P), chunk(P, "alpha")]
    assert label_ranks(retrieved, [P]) == {P: 3}  # the intro chunk "P#"


def test_nothing_retrieved_ranks_nothing() -> None:
    assert label_ranks([], [f"{P}#alpha"]) == {}


@pytest.mark.parametrize(
    "labels",
    [
        [f"{P}#alpha", f"{P}#beta"],  # H2 and one of its H3s
        [P, f"{P}#beta"],  # a page and a section on it
    ],
)
def test_one_chunk_matching_two_labels_is_bad_input(labels: list[str]) -> None:
    with pytest.raises(ValueError, match="nested"):
        label_ranks([chunk(P, "alpha", "beta")], labels)


# --- recall_at_k --------------------------------------------------------------------------------

A, B, G1 = f"{P}#a", f"{P}#b", f"{P}#g1"
RELEVANT = {A: 2, B: 2, G1: 1}


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (1, 1 / 2),  # A only
        (5, 1 / 2),  # G1 at rank 2 is grade 1: doesn't count
        (7, 2 / 2),
        (50, 2 / 2),  # k beyond the list is fine
    ],
)
def test_recall_at_k(k: int, expected: float) -> None:
    ranks = {A: 1, G1: 2, B: 7}
    assert recall_at_k(ranks, RELEVANT, k) == pytest.approx(expected)


def test_recall_keeps_its_denominator_when_labels_are_missing() -> None:
    assert recall_at_k({A: 3}, RELEVANT, 50) == pytest.approx(1 / 2)  # B never retrieved
    assert recall_at_k({}, RELEVANT, 5) == 0.0


# --- mrr ----------------------------------------------------------------------------------------


def test_mrr_uses_the_best_grade_2_rank() -> None:
    # G1 at rank 1 is grade 1 and doesn't count; B (4) beats A (6).
    assert mrr({G1: 1, B: 4, A: 6}, RELEVANT) == pytest.approx(1 / 4)


def test_mrr_is_zero_when_no_grade_2_label_is_retrieved() -> None:
    assert mrr({G1: 1}, RELEVANT) == 0.0
    assert mrr({}, RELEVANT) == 0.0


# --- ndcg_at_k ----------------------------------------------------------------------------------


def test_ndcg_perfect_order_is_one() -> None:
    assert ndcg_at_k({A: 1, G1: 2}, {A: 2, G1: 1}, 5) == pytest.approx(1.0)


def test_ndcg_swapped_order() -> None:
    # DCG = 1/log2(2) + 3/log2(3) = 2.8928; ideal = 3/log2(2) + 1/log2(3) = 3.6309 → 0.7967
    expected = (1 + 3 / log2(3)) / (3 + 1 / log2(3))
    assert ndcg_at_k({G1: 1, A: 2}, {A: 2, G1: 1}, 5) == pytest.approx(expected)


def test_ndcg_with_gaps_and_a_label_below_k() -> None:
    # Within k=5: A at 2 and G1 at 4 (B at 9 is cut off).
    # DCG = 3/log2(3) + 1/log2(5) = 2.3235
    # ideal over min(5, 3) = 3 labels, grades [2, 2, 1]: 3/1 + 3/log2(3) + 1/log2(4) = 5.3928
    # → 0.4308
    expected = (3 / log2(3) + 1 / log2(5)) / (3 + 3 / log2(3) + 1 / log2(4))
    assert ndcg_at_k({A: 2, G1: 4, B: 9}, RELEVANT, 5) == pytest.approx(expected)


def test_ideal_dcg_is_cut_at_k() -> None:
    # k=2 < 3 labels: the ideal keeps only the two grade-2 labels (3 + 3/log2(3)), which is
    # exactly what ranks A=1, B=2 achieve. Without the cut the result would be < 1.
    assert ndcg_at_k({A: 1, B: 2, G1: 3}, RELEVANT, 2) == pytest.approx(1.0)


def test_ndcg_with_k_beyond_the_list() -> None:
    # ideal over min(10, 1) = 1 label.
    assert ndcg_at_k({A: 1}, {A: 2}, 10) == pytest.approx(1.0)


def test_ndcg_counts_grade_1_labels() -> None:
    # Only G1 retrieved, at rank 3: DCG = 1/log2(4) = 0.5. ideal (k=5, grades [2, 1]):
    # 3/1 + 1/log2(3) = 3.6309 → 0.1377
    expected = (1 / log2(4)) / (3 + 1 / log2(3))
    assert ndcg_at_k({G1: 3}, {A: 2, G1: 1}, 5) == pytest.approx(expected)


def test_ndcg_is_zero_when_nothing_relevant_is_in_the_top_k() -> None:
    assert ndcg_at_k({A: 6}, RELEVANT, 5) == 0.0
    assert ndcg_at_k({}, RELEVANT, 5) == 0.0


def test_ndcg_rejects_repeated_ranks() -> None:
    with pytest.raises(ValueError, match="rank"):
        ndcg_at_k({A: 1, G1: 1}, RELEVANT, 5)


# --- Input errors shared by the metrics ---------------------------------------------------------


@pytest.mark.parametrize(
    "metric",
    [
        lambda rel: recall_at_k({}, rel, 5),
        lambda rel: mrr({}, rel),
        lambda rel: ndcg_at_k({}, rel, 5),
    ],
    ids=["recall", "mrr", "ndcg"],
)
@pytest.mark.parametrize("relevant", [{}, {G1: 1}], ids=["no-labels", "grade-1-only"])
def test_a_question_needs_a_grade_2_label(
    metric: Callable[[dict[str, int]], float], relevant: dict[str, int]
) -> None:
    with pytest.raises(ValueError, match="grade-2"):
        metric(relevant)


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_positive(k: int) -> None:
    with pytest.raises(ValueError, match="k"):
        recall_at_k({A: 1}, RELEVANT, k)
    with pytest.raises(ValueError, match="k"):
        ndcg_at_k({A: 1}, RELEVANT, k)


# --- Worked example: one question end to end ----------------------------------------------------


def test_worked_example() -> None:
    labels = {f"{P}#alpha": 2, f"{P}#gamma": 2, Q: 1}
    retrieved = [
        chunk(R, "z"),  # 1
        chunk(P, "alpha", "beta"),  # 2: P#alpha (H2 label via its H3 chunk)
        chunk(P, "alpha", "beta"),  # 3: same section again, already ranked
        chunk(Q),  # 4: Q (page label, grade 1), via the intro
        chunk(Q, "y"),  # 5: Q again
        chunk(P, "gamma-extra"),  # 6: not P#gamma
        chunk(R, "z"),  # 7
        chunk(R, "w"),  # 8
        chunk(P, "gamma"),  # 9: P#gamma
        chunk(R, "v"),  # 10
    ]
    ranks = label_ranks(retrieved, labels)
    assert ranks == {f"{P}#alpha": 2, Q: 4, f"{P}#gamma": 9}

    assert recall_at_k(ranks, labels, 5) == pytest.approx(1 / 2)
    assert recall_at_k(ranks, labels, 10) == pytest.approx(1.0)
    assert mrr(ranks, labels) == pytest.approx(1 / 2)
    ideal = 3 + 3 / log2(3) + 1 / log2(4)  # 5.3928
    assert ndcg_at_k(ranks, labels, 5) == pytest.approx((3 / log2(3) + 1 / log2(5)) / ideal)
    assert ndcg_at_k(ranks, labels, 10) == pytest.approx(
        (3 / log2(3) + 1 / log2(5) + 3 / log2(10)) / ideal  # 0.5983
    )
