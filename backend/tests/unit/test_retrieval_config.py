"""``RetrievalConfig``: built from Settings, and a hash that is a stable function of its value."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from grounded.retrieval.config import RetrievalConfig
from tests.support import make_settings

FIELDS: dict[str, Any] = {
    "mode": "hybrid",
    "k_dense": 20,
    "k_fts": 20,
    "k_fused": 40,
    "k_context": 5,
    "rrf_k": 60,
    "rerank_provider": "none",
    "rerank_model": None,
}


def make(**overrides: Any) -> RetrievalConfig:
    return RetrievalConfig.model_validate(FIELDS | overrides)


def test_from_settings_takes_the_retrieval_fields_and_the_mode() -> None:
    settings = make_settings(k_dense=11, k_fts=12, k_fused=13, k_context=3, rrf_k=14)
    assert RetrievalConfig.from_settings(settings, "fts") == RetrievalConfig(
        mode="fts", k_dense=11, k_fts=12, k_fused=13, k_context=3, rrf_k=14
    )


def test_the_defaults_are_those_in_tech_md() -> None:
    assert RetrievalConfig.from_settings(make_settings(), "hybrid") == make()


def test_canonical_json_is_sorted_and_compact() -> None:
    # The exact text is the contract: cache keys and baseline rows carry hashes of it, so a
    # renamed field or a changed separator must show up here.
    assert make().canonical_json() == (
        '{"k_context":5,"k_dense":20,"k_fts":20,"k_fused":40,"mode":"hybrid",'
        '"rerank_model":null,"rerank_provider":"none","rrf_k":60}'
    )


def test_config_hash_is_the_sha256_of_the_canonical_json() -> None:
    config = make()
    expected = hashlib.sha256(config.canonical_json().encode("utf-8")).hexdigest()
    assert config.config_hash == expected
    assert len(config.config_hash) == 64


def test_the_same_inputs_give_the_same_hash() -> None:
    assert make().config_hash == make().config_hash
    assert (
        make().config_hash == RetrievalConfig.from_settings(make_settings(), "hybrid").config_hash
    )


CHANGES: dict[str, Any] = {
    "mode": "dense",
    "k_dense": 21,
    "k_fts": 21,
    "k_fused": 41,
    "k_context": 6,
    "rrf_k": 61,
    "rerank_provider": "cohere",
}


def test_every_field_is_covered_by_a_change() -> None:
    # A new field must get a row in CHANGES (rerank_model needs a provider, so it has its own test).
    assert set(CHANGES) | {"rerank_model"} == set(FIELDS)


@pytest.mark.parametrize("field", list(CHANGES))
def test_changing_any_field_changes_the_hash(field: str) -> None:
    assert make(**{field: CHANGES[field]}).config_hash != make().config_hash


def test_changing_the_rerank_model_changes_the_hash() -> None:
    on = make(rerank_provider="cohere", rerank_model="model-a")
    assert on.config_hash != make(rerank_provider="cohere", rerank_model="model-b").config_hash
    assert on.config_hash != make(rerank_provider="cohere").config_hash


def test_key_order_does_not_matter() -> None:
    forward = json.dumps(FIELDS)
    backward = json.dumps(dict(reversed(FIELDS.items())))
    assert forward != backward
    a = RetrievalConfig.model_validate_json(forward)
    b = RetrievalConfig.model_validate_json(backward)
    assert a.canonical_json() == b.canonical_json()
    assert a.config_hash == b.config_hash


def test_a_config_round_trips_through_json_with_the_same_hash() -> None:
    config = make(rerank_provider="cohere", rerank_model="model-a")
    again = RetrievalConfig.model_validate_json(config.model_dump_json())
    assert again == config
    assert again.config_hash == config.config_hash


def test_a_config_is_frozen_and_closed() -> None:
    config = make()
    with pytest.raises(ValidationError):
        config.k_dense = 1  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValidationError):
        make(unknown=1)


@pytest.mark.parametrize("field", ["k_dense", "k_fts", "k_fused", "k_context", "rrf_k"])
def test_ks_must_be_positive(field: str) -> None:
    with pytest.raises(ValidationError):
        make(**{field: 0})


def test_a_rerank_model_needs_a_provider() -> None:
    with pytest.raises(ValidationError, match="rerank_provider"):
        make(rerank_model="model-a")


def test_from_settings_takes_rerank_only_when_it_is_on() -> None:
    # Rerank off: a stray RERANK_MODEL is not applied, so it must not change the hash.
    off = RetrievalConfig.from_settings(make_settings(rerank_model="model-a"), "hybrid")
    assert (off.rerank_provider, off.rerank_model) == ("none", None)
    assert off.config_hash == make().config_hash

    on = RetrievalConfig.from_settings(
        make_settings(rerank_provider="cohere", rerank_model="model-a"), "hybrid"
    )
    assert (on.rerank_provider, on.rerank_model) == ("cohere", "model-a")
    assert on.config_hash != off.config_hash
