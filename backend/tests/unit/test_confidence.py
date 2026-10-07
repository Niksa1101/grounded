"""Spec tests of the confidence heuristic (Tech.md §9.8, ticket 3.09).

They state *invariants* (a range, a cap, an ordering, determinism), never the value of a score, so
they describe what any acceptable heuristic must satisfy without being the heuristic. They were
strict ``xfail`` until ``score_claims`` landed in ticket 3.10, which removed the markers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError

from grounded.generation.confidence import (
    COMPONENT_KEYS,
    ClaimScore,
    ConfidenceConfig,
    score_claims,
)
from grounded.retrieval.types import RetrievedChunk
from grounded.schemas.llm import LLMClaim
from tests.support import make_settings

CONFIG = ConfidenceConfig.from_settings(make_settings())
# A non-default config that ``Settings`` accepts, close to its invariant-4 ceiling (0.594 of 0.6):
# the bound must hold for every accepted config, not only the defaults (Phase 3 review #4).
NEAR_CEILING = ConfidenceConfig.from_settings(
    make_settings(
        confidence_w_self=0.3,
        confidence_w_retrieval=0.3,
        confidence_w_agreement=0.21,
        confidence_w_citations=0.2,
        confidence_w_rerank=0.0,
    )
)

# --- Builders ------------------------------------------------------------------------------------


def chunk(chunk_id: int, **signals: Any) -> RetrievedChunk:
    """A chunk with only the given retrieval signals set; the others stay ``None``."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        section_id=f"docs/page.md#s{chunk_id}",
        anchor_path=(f"s{chunk_id}",),
        breadcrumb_text=f"Page > Section {chunk_id}",
        url=f"https://fastapi.tiangolo.com/page/#section-{chunk_id}",
        content="body text",
        token_count=10,
        content_hash=f"hash{chunk_id}",
        **signals,
    )


def claim(*labels: str, self_confidence: float = 0.5) -> LLMClaim:
    return LLMClaim(text="A claim.", citation_ids=list(labels), self_confidence=self_confidence)


# A realistic list around the chunks under test: strong chunks ahead of them, weak ones behind.
STRONG = tuple(
    chunk(
        800 + i,
        dense_rank=i + 1,
        dense_distance=0.1,
        fts_rank=i + 1,
        fts_score=2.0,
        rrf_score=2 / (61 + i),
    )
    for i in range(3)
)
WEAK = tuple(
    chunk(900 + i, dense_rank=15 + i, dense_distance=0.7, rrf_score=1 / (75 + i)) for i in range(3)
)


def score(
    claims: list[LLMClaim],
    labels: Mapping[str, RetrievedChunk],
    *,
    bottom: bool = False,
    config: ConfidenceConfig = CONFIG,
) -> list[ClaimScore]:
    """Run ``score_claims`` with the labelled chunks at the top of ``retrieved``, or the bottom."""
    cited = list(labels.values())
    retrieved = [*STRONG, *WEAK, *cited] if bottom else [*cited, *WEAK]
    return score_claims(claims, labels, retrieved, config=config)


def one(
    ids: tuple[str, ...],
    labels: Mapping[str, RetrievedChunk],
    *,
    self_confidence: float = 0.5,
    bottom: bool = False,
    config: ConfidenceConfig = CONFIG,
) -> ClaimScore:
    [result] = score(
        [claim(*ids, self_confidence=self_confidence)], labels, bottom=bottom, config=config
    )
    return result


# --- Plain tests: types and settings --------------------------------------------------------------


def test_claim_score_is_frozen() -> None:
    result = ClaimScore(confidence=0.5, components=dict.fromkeys(COMPONENT_KEYS, 0.5))
    with pytest.raises(FrozenInstanceError):
        result.confidence = 0.9  # type: ignore[misc]


def test_component_keys_are_distinct_and_named() -> None:
    assert len(set(COMPONENT_KEYS)) == len(COMPONENT_KEYS) > 0
    assert all(key and key == key.strip() for key in COMPONENT_KEYS)


def test_config_is_built_from_the_settings_fields() -> None:
    settings = make_settings(
        confidence_w_retrieval=0.1,
        confidence_w_agreement=0.2,
        confidence_w_citations=0.3,
        confidence_w_self=0.4,
        confidence_w_rerank=0.5,
        confidence_uncited_cap=0.05,
    )
    assert ConfidenceConfig.from_settings(settings) == ConfidenceConfig(
        w_retrieval=0.1,
        w_agreement=0.2,
        w_citations=0.3,
        w_self=0.4,
        w_rerank=0.5,
        uncited_cap=0.05,
    )


def test_default_weights_are_shares_and_the_cap_is_the_documented_ceiling() -> None:
    s = make_settings()
    weights = (
        s.confidence_w_retrieval,
        s.confidence_w_agreement,
        s.confidence_w_citations,
        s.confidence_w_self,
        s.confidence_w_rerank,
    )
    assert math.isclose(sum(weights), 1.0)
    assert s.confidence_uncited_cap == 0.2


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("confidence_uncited_cap", 0.21),  # Tech §9.8: an uncited claim is capped at 0.2
        ("confidence_uncited_cap", -0.1),
        ("confidence_w_self", 0.61),  # Tech §9.8: self-report alone can't pass 0.6
        ("confidence_w_self", -0.1),
        ("confidence_w_retrieval", -0.1),
        ("confidence_w_agreement", 1.1),
        ("confidence_w_citations", -0.1),
        ("confidence_w_rerank", -0.1),
    ],
)
def test_settings_reject_values_that_break_the_invariants(name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{name: value})


# --- Spec tests: the invariants -------------------------------------------------------------------

# Chunk profiles from "nothing known" to the extremes of every signal.
PROFILES: dict[str, dict[str, Any]] = {
    "no_signals": {},
    "dense_only": {"dense_rank": 3, "dense_distance": 0.3, "rrf_score": 1 / 63},
    "fts_only": {"fts_rank": 2, "fts_score": 1.4, "rrf_score": 1 / 62},
    "both_lists": {
        "dense_rank": 1,
        "dense_distance": 0.05,
        "fts_rank": 1,
        "fts_score": 3.0,
        "rrf_score": 2 / 61,
    },
    "reranked": {"dense_rank": 2, "rrf_score": 1 / 62, "rerank_score": 0.93},
    "rerank_zero": {"dense_rank": 2, "rrf_score": 1 / 62, "rerank_score": 0.0},
    "extreme_low": {
        "dense_rank": 40,
        "dense_distance": 2.0,
        "fts_rank": 40,
        "fts_score": 0.0,
        "rrf_score": 0.0,
        "rerank_score": 0.0,
    },
    "extreme_high": {
        "dense_rank": 1,
        "dense_distance": 0.0,
        "fts_rank": 1,
        "fts_score": 12.0,
        "rrf_score": 1.0,
        "rerank_score": 1.0,
    },
}
SELF_VALUES = (0.0, 0.3, 1.0)


def labels_of(profile: Mapping[str, Any], *others: Mapping[str, Any]) -> dict[str, RetrievedChunk]:
    return {f"c{n}": chunk(n, **p) for n, p in enumerate([profile, *others], start=1)}


@pytest.mark.parametrize("name", PROFILES)
@pytest.mark.parametrize("self_confidence", SELF_VALUES)
@pytest.mark.parametrize(
    "ids",
    [
        (),
        ("c9",),
        ("c1",),
        ("c1", "c2"),
        ("c1", "c1"),
        ("c1", "c2", "c3", "c4", "c5"),
        ("c1", "c9"),
    ],
    ids=["none", "unknown", "one", "two", "repeated", "five", "valid_and_unknown"],
)
def test_confidence_and_components_are_in_range_with_stable_keys(
    name: str, self_confidence: float, ids: tuple[str, ...]
) -> None:
    labels = labels_of(PROFILES[name], PROFILES["dense_only"], {}, PROFILES["both_lists"], {})
    result = one(ids, labels, self_confidence=self_confidence)
    assert math.isfinite(result.confidence)
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.components) == set(COMPONENT_KEYS)
    assert all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in result.components.values())


def test_confidence_is_in_range_over_a_grid_of_signals() -> None:
    grid = {
        "dense_rank": (None, 1, 40),
        "dense_distance": (None, 0.0, 2.0),
        "fts_rank": (None, 1, 40),
        "fts_score": (None, 0.0, 12.0),
        "rrf_score": (None, 0.0, 0.0328),
        "rerank_score": (None, 0.0, 1.0),
    }
    for values in product(*grid.values()):
        signals = dict(zip(grid, values, strict=True))
        for self_confidence in (0.0, 1.0):
            result = one(("c1",), {"c1": chunk(1, **signals)}, self_confidence=self_confidence)
            assert 0.0 <= result.confidence <= 1.0, signals


def test_no_claims_gives_no_scores() -> None:
    assert score([], labels_of(PROFILES["both_lists"])) == []


@pytest.mark.parametrize(
    "config",
    [
        CONFIG,
        replace(CONFIG, uncited_cap=0.05),
        replace(CONFIG, uncited_cap=0.0),
        replace(CONFIG, w_self=0.6, w_rerank=0.3),
    ],
    ids=["default", "cap_0.05", "cap_0", "heavy_self_and_rerank"],
)
@pytest.mark.parametrize("self_confidence", [0.0, 1.0])
@pytest.mark.parametrize("ids", [(), ("c7",), ("c7", "c8")], ids=["no_ids", "unknown", "unknowns"])
def test_a_claim_without_a_valid_citation_is_capped(
    config: ConfidenceConfig, self_confidence: float, ids: tuple[str, ...]
) -> None:
    # The best retrieval there is, in the label map and at the top: it must not rescue the claim,
    # because the claim does not cite any of it.
    labels = labels_of(PROFILES["extreme_high"], PROFILES["both_lists"])
    result = one(ids, labels, self_confidence=self_confidence, config=config)
    assert result.confidence <= config.uncited_cap
    assert set(result.components) == set(COMPONENT_KEYS)


@pytest.mark.parametrize(
    "weakest",
    [
        {},
        {"dense_rank": 20, "dense_distance": 0.9, "rrf_score": 1 / 80},
        {"fts_rank": 20, "fts_score": 0.0, "rrf_score": 1 / 80},
        {"dense_rank": 40, "dense_distance": 2.0, "rrf_score": 0.0},
    ],
    ids=["no_signals", "dense_last", "fts_last", "extreme_low"],
)
@pytest.mark.parametrize("config", [CONFIG, NEAR_CEILING], ids=["defaults", "near-ceiling"])
def test_self_confidence_alone_cannot_pass_point_six(
    weakest: dict[str, Any], config: ConfidenceConfig
) -> None:
    # One citation to a chunk at the bottom of the list, one list only, no rerank.
    result = one(
        ("c1",), {"c1": chunk(1, **weakest)}, self_confidence=1.0, bottom=True, config=config
    )
    assert result.confidence <= 0.6


# One side of each pair is weaker retrieval support for the same cited chunk, the other stronger.
SUPPORT_PAIRS = {
    "dense_rank_better": ({"dense_rank": 6}, {"dense_rank": 1}),
    "fts_rank_better": ({"fts_rank": 9, "fts_score": 1.0}, {"fts_rank": 2, "fts_score": 1.0}),
    "rrf_higher": (
        {"dense_rank": 4, "rrf_score": 0.012},
        {"dense_rank": 4, "rrf_score": 0.0328},
    ),
    "dense_distance_smaller": (
        {"dense_rank": 4, "dense_distance": 0.6},
        {"dense_rank": 4, "dense_distance": 0.2},
    ),
    "fts_score_higher": (
        {"fts_rank": 4, "fts_score": 0.5},
        {"fts_rank": 4, "fts_score": 3.5},
    ),
    "fts_joins_dense": (
        {"dense_rank": 4, "rrf_score": 1 / 64},
        {"dense_rank": 4, "rrf_score": 1 / 64, "fts_rank": 4, "fts_score": 1.0},
    ),
    "dense_joins_fts": (
        {"fts_rank": 4, "fts_score": 1.0, "rrf_score": 1 / 64},
        {"fts_rank": 4, "fts_score": 1.0, "rrf_score": 1 / 64, "dense_rank": 4},
    ),
    "rerank_higher": (
        {"dense_rank": 4, "rerank_score": 0.2},
        {"dense_rank": 4, "rerank_score": 0.9},
    ),
}


@pytest.mark.parametrize("self_confidence", SELF_VALUES)
@pytest.mark.parametrize("pair", SUPPORT_PAIRS)
def test_better_retrieval_support_never_lowers_confidence(
    pair: str, self_confidence: float
) -> None:
    weaker, stronger = SUPPORT_PAIRS[pair]
    low = one(("c1",), {"c1": chunk(1, **weaker)}, self_confidence=self_confidence)
    high = one(("c1",), {"c1": chunk(1, **stronger)}, self_confidence=self_confidence)
    assert high.confidence >= low.confidence


def test_a_higher_self_confidence_never_lowers_confidence() -> None:
    labels = {"c1": chunk(1, **PROFILES["dense_only"])}
    values = [one(("c1",), labels, self_confidence=s).confidence for s in (0.0, 0.2, 0.5, 0.9, 1.0)]
    assert values == sorted(values)


@pytest.mark.parametrize("first", ["strong", "weak"])
@pytest.mark.parametrize("added", ["stronger", "equal", "weaker"])
@pytest.mark.parametrize("self_confidence", SELF_VALUES)
def test_one_more_valid_citation_never_lowers_confidence(
    first: str, added: str, self_confidence: float
) -> None:
    levels = {
        "strong": PROFILES["both_lists"],
        "weak": {"dense_rank": 12, "dense_distance": 0.8, "rrf_score": 1 / 72},
    }
    # The added chunk is the first one's twin, better or worse by a clear margin.
    extra = {
        "stronger": PROFILES["extreme_high"],
        "equal": levels[first],
        "weaker": PROFILES["extreme_low"],
    }[added]
    labels = {"c1": chunk(1, **levels[first]), "c2": chunk(2, **extra)}
    alone = one(("c1",), labels, self_confidence=self_confidence)
    both = one(("c1", "c2"), labels, self_confidence=self_confidence)
    assert both.confidence >= alone.confidence


@pytest.mark.parametrize("name", ["dense_only", "both_lists", "reranked"])
def test_a_repeated_label_is_one_citation(name: str) -> None:
    labels = labels_of(PROFILES[name])
    assert one(("c1", "c1", "c1"), labels) == one(("c1",), labels)


def test_unknown_labels_are_not_citations() -> None:
    # They are counted elsewhere (citations.py); here they must not change what the valid ones earn.
    labels = labels_of(PROFILES["both_lists"])
    assert one(("c1", "c8"), labels) == one(("c1",), labels)


@pytest.mark.parametrize("bottom", [False, True])
def test_rerank_off_is_handled(bottom: bool) -> None:
    labels = labels_of({"dense_rank": 2, "rrf_score": 1 / 62})
    assert labels["c1"].rerank_score is None
    result = one(("c1",), labels, bottom=bottom)
    assert 0.0 <= result.confidence <= 1.0
    assert set(result.components) == set(COMPONENT_KEYS)


def test_same_input_gives_the_same_output_and_inputs_are_untouched() -> None:
    labels = labels_of(PROFILES["reranked"], PROFILES["dense_only"])
    claims = [claim("c1", self_confidence=0.7), claim("c2", "c1"), claim(), claim("c5")]
    snapshot = (dict(labels), list(claims))
    first = score(claims, labels)
    second = score(claims, labels)
    assert first == second
    assert (dict(labels), list(claims)) == snapshot


def test_each_claim_is_scored_on_its_own_and_in_order() -> None:
    labels = labels_of(PROFILES["both_lists"], PROFILES["dense_only"], PROFILES["extreme_low"])
    claims = [claim("c1", self_confidence=0.9), claim(), claim("c2", "c3"), claim("c3")]
    together = score(claims, labels)
    assert len(together) == len(claims)
    assert together == [score([c], labels)[0] for c in claims]
    assert score(claims[::-1], labels) == together[::-1]


# --- Tests added with the implementation (3.10): gaps the spec tests leave -------------------

# Weakest to strongest value of each signal, for the monotonicity sweep below.
SIGNAL_ORDER: dict[str, tuple[float, ...]] = {
    "dense_rank": (40, 10, 1),
    "dense_distance": (2.0, 0.5, 0.0),
    "fts_rank": (40, 10, 1),
    "fts_score": (0.0, 1.0, 12.0),
    "rrf_score": (0.0, 0.01, 0.0328),
    "rerank_score": (0.0, 0.5, 1.0),
}


@pytest.mark.parametrize("self_confidence", [0.0, 1.0])
@pytest.mark.parametrize("key", SIGNAL_ORDER)
def test_improving_any_one_signal_never_lowers_confidence_in_any_context(
    key: str, self_confidence: float
) -> None:
    # The spec pairs fix one context; this sweeps every combination of the other signals being
    # absent or present, so a non-monotone interaction between two signals cannot hide.
    others = [k for k in SIGNAL_ORDER if k != key]
    for present in product((False, True), repeat=len(others)):
        base = {k: SIGNAL_ORDER[k][1] for k, on in zip(others, present, strict=True) if on}
        values = [
            one(
                ("c1",),
                {"c1": chunk(1, **base, **{key: v})},
                self_confidence=self_confidence,
            ).confidence
            for v in SIGNAL_ORDER[key]
        ]
        assert values == sorted(values), (key, base, values)


@pytest.mark.parametrize("config", [CONFIG, NEAR_CEILING], ids=["defaults", "near-ceiling"])
@pytest.mark.parametrize("bottom", [False, True])
@pytest.mark.parametrize("found_by", ["dense", "fts"])
def test_a_chunk_found_by_one_list_cannot_pass_point_six_even_with_perfect_signals(
    found_by: str, bottom: bool, config: ConfidenceConfig
) -> None:
    # The self-report bound must not depend on the signals being weak: one list alone cannot
    # corroborate the chunk, so even its best values stay under the line.
    perfect = (
        {"dense_rank": 1, "dense_distance": 0.0, "rrf_score": 1.0}
        if found_by == "dense"
        else {"fts_rank": 1, "fts_score": 12.0, "rrf_score": 1.0}
    )
    result = one(
        ("c1",), {"c1": chunk(1, **perfect)}, self_confidence=1.0, bottom=bottom, config=config
    )
    assert result.confidence <= 0.6
    assert result.components["agreement"] == 0.0


def test_the_confidence_is_the_weighted_mean_of_its_components() -> None:
    labels = labels_of(PROFILES["both_lists"], PROFILES["dense_only"])
    config = replace(CONFIG, w_rerank=0.3)
    result = one(("c1", "c2"), labels, self_confidence=0.7, config=config)
    weights = {
        "retrieval": config.w_retrieval,
        "agreement": config.w_agreement,
        "citations": config.w_citations,
        "self_confidence": config.w_self,
        "rerank": config.w_rerank,
    }
    expected = sum(weights[k] * result.components[k] for k in COMPONENT_KEYS) / sum(
        weights.values()
    )
    assert math.isclose(result.confidence, expected)
    assert result.components["self_confidence"] == 0.7
    assert result.components["rerank"] == 0.0  # rerank is off: the component is 0, not an error


def test_an_uncited_claim_has_zero_support_components() -> None:
    result = one(("c9",), labels_of(PROFILES["both_lists"]), self_confidence=0.8)
    assert result.components == {
        "retrieval": 0.0,
        "agreement": 0.0,
        "citations": 0.0,
        "self_confidence": 0.8,
        "rerank": 0.0,
    }


def test_all_weights_zero_gives_zero_not_a_division_error() -> None:
    config = ConfidenceConfig(
        w_retrieval=0.0, w_agreement=0.0, w_citations=0.0, w_self=0.0, w_rerank=0.0, uncited_cap=0.2
    )
    result = one(("c1",), labels_of(PROFILES["both_lists"]), config=config)
    assert result.confidence == 0.0
