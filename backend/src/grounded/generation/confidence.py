"""Per-claim confidence, computed on the server (Tech.md §9.8, PRD FR-12 and D15).

**Author-owned** (AGENTS.md §3): the heuristic is written in ticket 3.10, this file only fixes the
contract. The spec tests are in ``tests/unit/test_confidence.py``.

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

from collections.abc import Mapping, Sequence
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
    """
    raise NotImplementedError("Author implements the confidence heuristic in ticket 3.10")
