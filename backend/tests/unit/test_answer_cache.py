"""The answer cache key and its failure policy (3.13, Tech §11): no database needed."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from typing import Any, cast
from uuid import UUID

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from grounded.generation.confidence import ConfidenceConfig
from grounded.infra.answer_cache import (
    MAX_TTL_DAYS,
    AnswerCache,
    CacheKey,
    confidence_config_hash,
    from_stored,
    to_stored,
)
from grounded.infra.hashing import normalize_question
from grounded.schemas.api import AskResponse, Meta
from tests.support import make_settings

CONFIDENCE = ConfidenceConfig.from_settings(make_settings())


def key(question: str = "How do I run a task?", **overrides: Any) -> CacheKey:
    parts: dict[str, Any] = {
        "prompt_version": "answer_v1@aaaaaaaa",
        "index_version_id": 1,
        "retrieval_config_hash": "r" * 64,
        "generator_model": "model-a",
        "confidence": CONFIDENCE,
    }
    return CacheKey.build(question, **{**parts, **overrides})


@pytest.mark.parametrize(
    "variant",
    [
        "how do i run a task",
        "How do I run a task?",
        "  HOW   do I\trun a task ??",
        "How do I run a task…",
    ],
)
def test_questions_that_normalize_alike_share_a_key(variant: str) -> None:
    assert key(variant).digest == key("How do I run a task?").digest


@pytest.mark.parametrize(
    "other",
    [
        "How do I run a test?",
        "What is C#?",  # not "what is c": only sentence punctuation is stripped
    ],
)
def test_different_questions_have_different_keys(other: str) -> None:
    assert key(other).digest != key("How do I run a task?").digest
    assert key("What is C?").digest != key("What is C#?").digest


def test_the_key_uses_the_shared_normalization() -> None:
    # One definition for the cache and request_logs.question_hash (infra/hashing.py).
    assert key("Why   NOT?!").question == normalize_question("Why   NOT?!") == "why not"


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_version": "answer_v1@bbbbbbbb"},
        {"index_version_id": 2},
        {"retrieval_config_hash": "s" * 64},
        {"generator_model": "model-b"},
        {"confidence": dataclasses.replace(CONFIDENCE, w_self=0.3)},
        {"confidence": dataclasses.replace(CONFIDENCE, uncited_cap=0.1)},
    ],
    ids=["prompt", "index", "retrieval", "model", "confidence weight", "confidence cap"],
)
def test_every_part_of_the_key_changes_it(change: dict[str, Any]) -> None:
    assert key(**change).digest != key().digest


def test_the_key_is_stable_and_is_a_sha256() -> None:
    assert key().digest == key().digest
    assert len(key().digest) == 64
    assert (
        key().digest
        == hashlib.sha256(
            json.dumps(
                [
                    "how do i run a task",
                    "answer_v1@aaaaaaaa",
                    1,
                    "r" * 64,
                    "model-a",
                    confidence_config_hash(CONFIDENCE),
                ]
            ).encode()
        ).hexdigest()
    )


def test_a_separator_inside_a_question_cannot_shift_a_field() -> None:
    # With a plain "|" join, these two would be the same string.
    forged = key("what is it|answer_v1@aaaaaaaa", prompt_version="1", index_version_id=7)
    honest = key("what is it", prompt_version="answer_v1@aaaaaaaa", index_version_id=1)
    assert forged.digest != honest.digest


def test_the_confidence_hash_follows_the_values() -> None:
    same = ConfidenceConfig.from_settings(make_settings())
    assert confidence_config_hash(same) == confidence_config_hash(CONFIDENCE)
    other = ConfidenceConfig.from_settings(make_settings(confidence_w_retrieval=0.5))
    assert confidence_config_hash(other) != confidence_config_hash(CONFIDENCE)


def response() -> AskResponse:
    return AskResponse(
        status="answered",
        answer_markdown="Run it. [1]",
        claims=[],
        citations=[],
        follow_up_questions=[],
        min_confidence=None,
        meta=Meta(
            request_id=UUID(int=1),
            provider="p",
            model="m",
            fallback_used=False,
            cache_hit=False,
            rerank_used=False,
            prompt_version="answer_v1@aaaaaaaa",
            index_version="0.0.1@abcdef12",
            retrieval_config_hash="r" * 64,
            latency_ms={"total": 5},
            tokens={"input": 1, "output": 2},
            shadow_cost_usd=0.5,
        ),
    )


def test_the_stored_response_has_no_per_request_meta_and_round_trips() -> None:
    original = response()
    stored = to_stored(original)
    assert "meta" not in stored
    json.dumps(stored)  # plain JSON: it goes into a jsonb column

    fresh = original.meta.model_copy(update={"request_id": UUID(int=2), "cache_hit": True})
    rebuilt = from_stored(stored, fresh)
    assert rebuilt.meta == fresh
    assert rebuilt.model_dump(exclude={"meta"}) == original.model_dump(exclude={"meta"})


def test_the_ttl_is_clamped_to_the_retention_limit() -> None:
    assert AnswerCache(cast(AsyncConnectionPool, None), ttl_days=90)._ttl_days == MAX_TTL_DAYS  # pyright: ignore[reportPrivateUsage]


# --- a database error is a miss or a skipped write, logged without the question -----------------


class _DownPool:
    def connection(self) -> Any:
        raise psycopg.OperationalError("connection failed: DETAIL zebra-secret-question")


async def test_a_database_error_on_lookup_is_a_logged_miss(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = AnswerCache(cast(AsyncConnectionPool, _DownPool()), ttl_days=30)
    with caplog.at_level(logging.WARNING):
        assert await cache.get(key("zebra-secret-question")) is None
    [record] = [r for r in caplog.records if r.name == "grounded.infra.answer_cache"]
    assert record.getMessage() == "answer cache lookup failed"
    assert record.__dict__["error"] == "OperationalError"
    assert "zebra" not in caplog.text


async def test_a_database_error_on_write_is_logged_and_does_not_raise(
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = AnswerCache(cast(AsyncConnectionPool, _DownPool()), ttl_days=30)
    with caplog.at_level(logging.WARNING):
        await cache.put(key("zebra-secret-question"), response())
    assert "answer cache write failed" in caplog.text
    assert "zebra" not in caplog.text
