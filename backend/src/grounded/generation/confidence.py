"""Per-claim confidence, computed on the server (Tech.md §9.8, PRD FR-12 and D15).

**Author-owned** (AGENTS.md §3). The contract below is from 3.09. The heuristic is from 3.10, which
the Author delegated to the agent ("write it") with a line-by-line explanation in the PR; the Author
reviews it and may rewrite it. The spec tests are in ``tests/unit/test_confidence.py``.

**Why it exists.** The model's own ``self_confidence`` is poorly calibrated, so it is never shown
as "confidence" (AGENTS.md §6.6). What the reader sees is a number the *server* derives from
evidence it can check: how well retrieval ranked the cited chunks, whether the dense and the lexical
search agreed on them, how many sources back the claim, and, weakly, what the model said about
itself. Phase 8 measures whether the number is calibrated (buckets against the judge's "supported"
rate), so the heuristic must stay simple enough to explain and every number it uses must be a
``Settings`` field.

**Inputs.**

- ``claims``: the model's claims as validated by ``LLMClaim``, so their citations are still *labels*
  (``c1..c9``) and ``self_confidence`` is available.
- ``label_map``: this request's labels, label -> chunk (``BuiltContext.labels``, Tech §9.3). A
  *valid* citation of a claim is a label of ``citation_ids`` that is a key of ``label_map``. Labels
  that are not keys are ignored here (``citations.map_citations`` counts them, §9.6); the same label
  twice in one claim is one citation.
- ``retrieved``: the whole fused list as ``hybrid_search`` returned it, best first. Every chunk of
  ``label_map`` is in it (same ``chunk_id``). It is the context for judging one chunk's standing
  (its position, the scale of the scores), nothing requires the heuristic to use it.
- ``config``: the weights and the cap, a plain value object built from ``Settings`` (see
  ``ConfidenceConfig``) so this stays a pure function.

**The signals** a chunk carries are the ``RetrievedChunk`` fields ``dense_rank`` and
``dense_distance``, ``fts_rank`` and ``fts_score``, ``rrf_score`` and ``rerank_score``. Each is
``None`` when the chunk was not found by that list (dense-only or lexical-only retrieval, rerank
off). ``rerank_score``, when present, is a relevance score in ``[0, 1]`` (Phase 6). ``fts_score`` is
``ts_rank_cd``: unbounded and only comparable within one query. ``dense_distance`` is a cosine
distance in ``[0, 2]``, lower is better.

**Output.** One ``ClaimScore`` per input claim, in the same order (``[]`` for no claims); a claim is
scored on its own, never by what the other claims say. ``ClaimScore.components`` is returned in the
API for transparency, with exactly the keys ``COMPONENT_KEYS``, in every case (also for an uncited
claim, and with rerank off), so the response has a stable shape:

- ``retrieval``: how strongly retrieval ranked what the claim cites.
- ``agreement``: whether the dense and the lexical search both found what the claim cites.
- ``citations``: how many distinct valid sources back the claim.
- ``self_confidence``: the model's self-report, as given.
- ``rerank``: the rerank relevance of what the claim cites; ``0.0`` when rerank is off.

Every component is a finite float in ``[0, 1]``. A component is a *signal*, not the final number:
how they are combined, and whether the weights are normalised, is the heuristic's decision and is
documented in the README (Tech §9.8). The Author may rename or add keys (the optional lexical
overlap between a claim and its chunks is the usual candidate) by changing ``COMPONENT_KEYS`` and
the config.

**Invariants** (the spec tests pin each one; none of them says how to compute the number):

1. *Pure and deterministic.* The result depends only on the arguments: no clock, no randomness, no
   I/O, and no argument is modified. Same input, same output.
2. *Range.* ``confidence`` is a finite float in ``[0, 1]`` for every input, including extreme
   signals (a zero ``rrf_score``, a ``fts_score`` far above 1, a distance of 2).
3. *No valid citation is capped.* A claim with no valid citation (no ``citation_ids``, or only
   labels that are not in ``label_map``) has ``confidence <= config.uncited_cap``, whatever its
   ``self_confidence`` and however good the retrieval was. ``Settings`` keeps the cap at most
   ``0.2``.
4. *Self-report cannot carry a claim.* With ``self_confidence = 1.0`` and the weakest possible
   retrieval support (one valid citation, to a chunk at the bottom of ``retrieved`` that only one
   list found, at its last rank, with no rerank score), ``confidence <= 0.6``. This holds for the
   default config; ``Settings`` caps ``w_self`` at ``0.6`` as a guard.
5. *Monotonic in support.* Changing one thing at a time, none of these may lower ``confidence``:
   a better ``dense_rank`` or ``fts_rank`` (a smaller number), a chunk found by both lists instead
   of one, a higher ``rrf_score``, a smaller ``dense_distance``, a higher ``fts_score``, a higher
   ``rerank_score`` (both present), a higher ``self_confidence``, and **one more distinct valid
   citation, even to a chunk weaker than the ones already cited** (so an average over the cited
   chunks would break this, because a weak extra source would drag it down). A repeated label is
   not an extra citation: the score does not change.
6. *Rerank is optional.* ``rerank_score is None`` (rerank off) is a normal input, not an error.
   When it is present it may add support (Phase 6).

``min_confidence`` of the whole answer is the minimum over the claims and is the caller's job (the
pipeline, 3.10).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from grounded.retrieval.types import RetrievedChunk
from grounded.schemas.llm import LLMClaim
from grounded.settings import Settings

# The keys of ``ClaimScore.components``, in the order the API shows them. Documented above; the
# spec tests read the key set from here, so a new key is added here (and to the docstring).
COMPONENT_KEYS: Final = ("retrieval", "agreement", "citations", "self_confidence", "rerank")


@dataclass(frozen=True, slots=True)
class ConfidenceConfig:
    """The numbers the heuristic may use, taken from ``Settings`` (AGENTS.md §6.7).

    The weights are relative importances (they are *not* required to sum to 1: the heuristic decides
    whether to normalise). ``uncited_cap`` is the ceiling of a claim with no valid citation.
    """

    w_retrieval: float
    w_agreement: float
    w_citations: float
    w_self: float
    w_rerank: float
    uncited_cap: float

    @classmethod
    def from_settings(cls, settings: Settings) -> ConfidenceConfig:
        return cls(
            w_retrieval=settings.confidence_w_retrieval,
            w_agreement=settings.confidence_w_agreement,
            w_citations=settings.confidence_w_citations,
            w_self=settings.confidence_w_self,
            w_rerank=settings.confidence_w_rerank,
            uncited_cap=settings.confidence_uncited_cap,
        )


@dataclass(frozen=True, slots=True)
class ClaimScore:
    """The server-computed confidence of one claim and the components it was built from."""

    confidence: float  # in [0, 1]
    components: dict[str, float]  # keys are COMPONENT_KEYS, values in [0, 1]


# --- The heuristic (3.10) ------------------------------------------------------------------------
#
# Per cited chunk, three support values in [0, 1] are derived from its retrieval signals. A claim
# takes the *best* of each over its valid citations (a saturating aggregation: adding a citation
# can only raise it, an average could lower it). The five components are blended with the config
# weights into a weighted mean, and a claim with no valid citation is capped. ``score_claims``
# argues why each invariant holds for every input.

# A cosine distance lies in [0, 2] by definition (see above), so ``1 - d / 2`` maps it onto [0, 1].
# This is the width of the metric's range, not a tuned number.
_COSINE_DISTANCE_SPAN: Final = 2.0

# ``citations = n / (n + PRIOR_DOUBT)``: each distinct valid source is one unit of evidence against
# one unit of prior doubt, so one source gives 1/2, two give 2/3, and the value approaches 1
# without reaching it. It rises with every extra source however weak (invariant 5) and needs
# neither k nor the label count. It is the one free constant of the heuristic; the Phase 8
# calibration may revisit it.
_PRIOR_DOUBT: Final = 1.0


@dataclass(frozen=True, slots=True)
class _ListScale:
    """The scale of the signals that have none of their own, read off this request's list.

    A rank is a position in a list whose length is not given, and ``fts_score`` / ``rrf_score`` are
    only comparable within one query, so each is judged against the extreme of this list.
    """

    dense_depth: int  # the largest ``dense_rank`` seen (0 when no chunk has one)
    fts_depth: int
    max_fts_score: float  # the largest finite ``fts_score`` (0.0 when none is positive)
    max_rrf_score: float


@dataclass(frozen=True, slots=True)
class _ChunkSupport:
    """What retrieval says about one chunk; each value is in [0, 1]."""

    retrieval: float
    agreement: float
    rerank: float


def _unit(x: float) -> float:
    """Clamp to [0, 1]; a NaN or an infinity (the DB never produces one) becomes 0.0."""
    return min(1.0, max(0.0, x)) if math.isfinite(x) else 0.0


def _scale_of(chunks: Iterable[RetrievedChunk]) -> _ListScale:
    seen = list(chunks)
    return _ListScale(
        dense_depth=max((c.dense_rank for c in seen if c.dense_rank is not None), default=0),
        fts_depth=max((c.fts_rank for c in seen if c.fts_rank is not None), default=0),
        max_fts_score=max(
            (c.fts_score for c in seen if c.fts_score is not None and math.isfinite(c.fts_score)),
            default=0.0,
        ),
        max_rrf_score=max(
            (c.rrf_score for c in seen if c.rrf_score is not None and math.isfinite(c.rrf_score)),
            default=0.0,
        ),
    )


def _rank_strength(rank: int, depth: int) -> float:
    """``(depth - rank + 1) / depth``: the share of the list from this rank to its end.

    1.0 at rank 1 and ``1 / depth`` at the last rank; ``depth`` is forced to be at least ``rank``,
    so the value is in (0, 1]. It never rises as ``rank`` grows, also when ``rank`` itself is the
    deepest rank (then ``depth`` grows with it): with ``M`` the deepest *other* rank it is
    ``(M - rank + 1) / M`` for ``rank <= M`` and ``1 / rank`` beyond, and the two meet at ``1 / M``.
    """
    rank = max(rank, 1)
    depth = max(depth, rank)
    return (depth - rank + 1) / depth


def _ratio(value: float | None, top: float) -> float:
    """``value / top`` in [0, 1]; 0.0 when the signal is absent or the list has no positive top."""
    if value is None or top <= 0.0:
        return 0.0
    return _unit(value / top)


def _list_strength(rank: int | None, depth: int, value_part: float) -> float:
    """One list's verdict on a chunk: 0.0 when the list did not find it (no rank), else the mean
    of the chunk's rank in it and its raw score there, both in [0, 1]."""
    return 0.0 if rank is None else (_rank_strength(rank, depth) + value_part) / 2


def _chunk_support(chunk: RetrievedChunk, scale: _ListScale) -> _ChunkSupport:
    dense_closeness = (
        0.0
        if chunk.dense_distance is None
        else _unit(1.0 - chunk.dense_distance / _COSINE_DISTANCE_SPAN)
    )
    dense = _list_strength(chunk.dense_rank, scale.dense_depth, dense_closeness)
    lexical = _list_strength(
        chunk.fts_rank, scale.fts_depth, _ratio(chunk.fts_score, scale.max_fts_score)
    )
    fused = _ratio(chunk.rrf_score, scale.max_rrf_score)
    return _ChunkSupport(
        # Three fixed slots: an absent signal adds 0 and a present one adds >= 0, so a chunk found
        # by one more list never scores lower. A mean over only the *present* signals would.
        retrieval=(dense + lexical + fused) / 3,
        # Both lists must like the chunk: the weaker verdict, 0.0 unless both found it.
        agreement=min(dense, lexical),
        rerank=0.0 if chunk.rerank_score is None else _unit(chunk.rerank_score),
    )


def score_claims(
    claims: Sequence[LLMClaim],
    label_map: Mapping[str, RetrievedChunk],
    retrieved: Sequence[RetrievedChunk],
    *,
    config: ConfidenceConfig,
) -> list[ClaimScore]:
    """Score every claim: ``confidence`` and ``components``, one ``ClaimScore`` per claim, in order.

    The contract, the meaning of the arguments, the components and the invariants are in the module
    docstring; the spec tests in ``tests/unit/test_confidence.py`` pin each invariant.

    **Components** (each in [0, 1]; ``n`` is the number of distinct valid citations and the maxima
    run over the cited chunks, see ``_chunk_support``):

    - ``retrieval``: the maximum of the mean of three slots, the dense list's verdict, the lexical
      list's verdict and ``rrf_score`` relative to the best ``rrf_score`` of the list.
    - ``agreement``: the maximum of ``min(dense verdict, lexical verdict)``.
    - ``citations``: ``n / (n + 1)``.
    - ``self_confidence``: the model's value, clamped.
    - ``rerank``: the maximum ``rerank_score`` (``0.0`` when no cited chunk has one).

    **Confidence** is ``sum(w_k * component_k) / sum(w_k)``: the weights are relative, they need not
    sum to 1. A claim with no valid citation additionally gets ``min(.., uncited_cap)``. The
    weights are assumed non-negative (``Settings`` enforces it); with all zero it is 0.0.

    **Why each invariant holds for every input** (``w`` are the weights, ``W`` their sum):

    1. *Pure.* Nothing is read but the arguments and nothing is modified: ``retrieved`` and
       ``label_map`` are only iterated and the result is built from new objects.
    2. *Range.* Each component goes through ``_unit`` and the weights are non-negative, so the
       weighted mean is in [0, 1]; ``_unit`` is applied once more to the result to absorb the
       last-bit rounding error (a sum that comes out as ``1.0000000000000002``).
    3. *Uncited cap.* Without a valid citation ``n = 0`` and the ``min`` with ``uncited_cap`` is the
       last step, after the weighted mean, so nothing (the self-report, the retrieval of chunks the
       claim does not cite) can lift the result above the cap. ``Settings`` keeps the cap <= 0.2.
    4. *Self-report cannot carry.* With the default weights, one valid citation (``citations =
       1/2``), no rerank and a chunk found by one list only (``agreement = 0`` because
       ``min(x, 0) = 0``, and ``retrieval <= 2/3`` because the missing list's slot is 0), the
       confidence is at most ``0.15*1 + 0.40*(2/3) + 0.20*(1/2) = 0.517 <= 0.6``. That bound uses
       the best values the one list could have, so it covers the weakest support and everything
       above it, not only the tested cases.
    5. *Monotone.* A weighted mean with non-negative weights does not fall when a component rises,
       so it is enough that every component is non-decreasing in every signal. ``_rank_strength``
       does not rise with the rank (proof there). ``dense_distance`` enters as ``1 - d / 2``.
       ``fts_score`` and ``rrf_score`` enter as ``x / top`` where ``top`` is the best of the list
       *including the chunk itself*: raising the chunk's own value raises the ratio, or the chunk
       becomes the top and the ratio is 1.0, never less than before. A list that finds the chunk
       turns its slot from 0 into a value >= 0, and ``min`` of two non-negative values is >= 0.
       ``rerank`` and ``self_confidence`` enter directly. A ``max`` over the cited chunks does not
       fall when a chunk is added, and ``n / (n + 1)`` rises with ``n``, so one more distinct valid
       citation, however weak, never lowers any component. A repeated label is removed by
       ``dict.fromkeys`` before counting, so it changes nothing. Floating-point sums of
       non-negative terms and a division by the same positive ``W`` are monotone under
       round-to-nearest, so rounding cannot reverse an order.
    6. *Rerank optional.* ``None`` becomes the component 0.0, which adds nothing; with rerank on, a
       higher score adds more. With ``w_rerank > 0`` and rerank off the best possible confidence
       drops by that weight's share, so keep ``CONFIDENCE_W_RERANK`` at 0 until rerank is on.
    """
    # The scale is read off the retrieved list *and* the labelled chunks, so a labelled chunk that
    # (against the contract) is missing from ``retrieved`` still lies inside its own scale.
    scale = _scale_of([*retrieved, *label_map.values()])
    support = {label: _chunk_support(chunk, scale) for label, chunk in label_map.items()}
    weights = {
        "retrieval": config.w_retrieval,
        "agreement": config.w_agreement,
        "citations": config.w_citations,
        "self_confidence": config.w_self,
        "rerank": config.w_rerank,
    }
    total_weight = sum(weights.values())

    scores: list[ClaimScore] = []
    for claim in claims:
        cited = [support[label] for label in dict.fromkeys(claim.citation_ids) if label in support]
        n = len(cited)
        raw = {
            "retrieval": max((s.retrieval for s in cited), default=0.0),
            "agreement": max((s.agreement for s in cited), default=0.0),
            "citations": n / (n + _PRIOR_DOUBT),
            "self_confidence": claim.self_confidence,
            "rerank": max((s.rerank for s in cited), default=0.0),
        }
        components = {key: _unit(raw[key]) for key in COMPONENT_KEYS}
        blended = (
            sum(weights[key] * components[key] for key in COMPONENT_KEYS) / total_weight
            if total_weight > 0.0
            else 0.0
        )
        confidence = _unit(blended)
        if n == 0:
            confidence = min(confidence, config.uncited_cap)
        scores.append(ClaimScore(confidence=confidence, components=components))
    return scores
