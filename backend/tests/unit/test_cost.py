"""Shadow cost (Tech §14): hand-computed values, and the startup refusals."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from grounded.observability.cost import Pricing, PricingError, load_pricing
from grounded.settings import Settings

GEMINI = "gemini-x"
EMBED = "embed-x"
RERANK = "rerank-x"


def pricing(**overrides: Any) -> Pricing:
    table: dict[str, Any] = {
        "as_of": "2026-10-06",
        "source": "test",
        "models": {GEMINI: {"input_per_mtok": 0.30, "output_per_mtok": 2.50}},
        "embeddings": {EMBED: {"input_per_mtok": 0.15}},
        "rerank": {RERANK: {"per_1k_searches": 2.0}},
    }
    return Pricing.model_validate({**table, **overrides})


def test_generation_cost_by_hand() -> None:
    # 10,000 in * 0.30 / 1e6 = 0.003; 2,000 out * 2.50 / 1e6 = 0.005.
    cost = pricing().shadow_cost(
        provider="gemini", model=GEMINI, input_tokens=10_000, output_tokens=2_000
    )
    assert cost == Decimal("0.00800000")


def test_embedding_and_rerank_terms_by_hand() -> None:
    # 4,000 embed tokens * 0.15 / 1e6 = 0.0006; 3 rerank calls * 2.0 / 1000 = 0.006.
    cost = pricing().shadow_cost(
        embedding_model=EMBED, embedding_tokens=4_000, rerank_model=RERANK, rerank_calls=3
    )
    assert cost == Decimal("0.00660000")


def test_all_terms_add_up() -> None:
    cost = pricing().shadow_cost(
        provider="gemini",
        model=GEMINI,
        input_tokens=1_000,  # 0.0003
        output_tokens=500,  # 0.00125 (thinking tokens are inside this count)
        embedding_model=EMBED,
        embedding_tokens=28,  # 0.0000042
    )
    assert cost == Decimal("0.00155420")


def test_the_cost_is_rounded_half_up_to_the_eight_places_of_the_column() -> None:
    p = pricing()
    assert p.shadow_cost(provider="gemini", model=GEMINI, input_tokens=1) == Decimal("0.00000030")
    assert p.shadow_cost(embedding_model=EMBED, embedding_tokens=1) == Decimal("0.00000015")
    # 1 token at 0.005 per million is 5e-9, exactly half of the last place: it rounds up.
    half = pricing(embeddings={EMBED: {"input_per_mtok": 0.005}})
    assert half.shadow_cost(embedding_model=EMBED, embedding_tokens=1) == Decimal("0.00000001")
    # 1 token at 0.004 per million is 4e-9: below half, it rounds down.
    below = pricing(embeddings={EMBED: {"input_per_mtok": 0.004}})
    assert below.shadow_cost(embedding_model=EMBED, embedding_tokens=1) == Decimal("0.00000000")


def test_nothing_ran_costs_zero() -> None:
    assert pricing().shadow_cost() == Decimal("0.00000000")


def test_the_fake_provider_is_free_whatever_its_model() -> None:
    cost = pricing().shadow_cost(
        provider="fake", model="not-in-the-file", input_tokens=10**6, output_tokens=10**6
    )
    assert cost == 0


def test_an_unknown_model_is_an_error_not_zero() -> None:
    with pytest.raises(PricingError, match="no price for model 'gemini-y'"):
        pricing().shadow_cost(provider="gemini", model="gemini-y", input_tokens=1)
    with pytest.raises(PricingError, match="embedding model 'embed-y'"):
        pricing().shadow_cost(embedding_model="embed-y", embedding_tokens=1)
    with pytest.raises(PricingError, match="rerank model 'rerank-y'"):
        pricing().shadow_cost(rerank_model="rerank-y", rerank_calls=1)


def test_startup_checks_refuse_an_unknown_model() -> None:
    p = pricing()
    p.require_generator("gemini", GEMINI)
    p.require_generator("fake", "stub")  # exempt
    p.require_embedding(EMBED)
    with pytest.raises(PricingError, match="gemini-y"):
        p.require_generator("gemini", "gemini-y")
    with pytest.raises(PricingError, match="embed-y"):
        p.require_embedding("embed-y")


def test_an_unverified_price_is_unknown_never_zero() -> None:
    p = pricing(models={GEMINI: {"status": "unverified"}})
    with pytest.raises(PricingError, match="unverified"):
        p.require_generator("gemini", GEMINI)
    with pytest.raises(PricingError, match="unverified"):
        p.shadow_cost(provider="gemini", model=GEMINI)


def test_a_verified_row_without_prices_is_invalid() -> None:
    with pytest.raises(ValueError, match="needs input_per_mtok and output_per_mtok"):
        pricing(models={GEMINI: {"input_per_mtok": 0.3}})


def test_the_committed_pricing_file_loads_and_prices_the_defaults() -> None:
    loaded = load_pricing()
    defaults = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    loaded.require_embedding(defaults.embedding_model)
    loaded.require_generator("gemini", "gemini-3.5-flash-lite")


def test_a_missing_or_invalid_file_is_a_pricing_error(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="cannot load"):
        load_pricing(tmp_path / "missing.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text(
        'as_of = "2026-10-06"\nsource = "x"\n[models.m]\nbogus = 1\n', encoding="utf-8"
    )
    with pytest.raises(PricingError, match="cannot load"):
        load_pricing(broken)
