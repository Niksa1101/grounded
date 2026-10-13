"""``POST /v1/ask`` protection: the 4 KB body limit, the question rules and the rate limit through
the real app and a real database (5.04, Tech §12 and §13).

The limiter's window arithmetic is in ``test_ratelimit.py``; what only the route can show is here:
the order of the checks, the 429 and its header, the log row, and the eval switch.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

from grounded.infra.ratelimit import RateLimiter
from grounded.schemas.api import MAX_BODY_BYTES
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import (
    EVAL_SETTINGS,
    WallClock,
    app_client,
    insert_index_version,
    make_settings,
)

pytestmark = pytest.mark.integration

QUESTION = "Where does the quokka sleep?"
PEER = "198.51.100.9"
IP_KEY = "ip-hash-key-for-tests"


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute(
                "TRUNCATE index_versions, request_logs, rate_limit_counters "
                "RESTART IDENTITY CASCADE"
            )

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    yield test_database_url
    wipe()


def rows(url: str) -> list[dict[str, Any]]:
    with psycopg.connect(url) as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM request_logs ORDER BY created_at, id")
        return cur.fetchall()


def clock() -> WallClock:
    return WallClock(datetime(2026, 3, 14, 10, 0, 30, tzinfo=UTC))


async def ask(client: httpx.AsyncClient, question: str = QUESTION) -> httpx.Response:
    return await client.post("/v1/ask", json={"question": question})


# --- input limits ------------------------------------------------------------------------------


def body_of(size: int) -> bytes:
    """A valid ``AskRequest`` JSON of exactly ``size`` bytes (the extra key is ignored)."""
    base = len(json.dumps({"question": QUESTION, "pad": ""}))
    return json.dumps({"question": QUESTION, "pad": "x" * (size - base)}).encode()


async def post_raw(
    client: httpx.AsyncClient, content: bytes | AsyncIterator[bytes]
) -> httpx.Response:
    return await client.post(
        "/v1/ask", content=content, headers={"Content-Type": "application/json"}
    )


async def test_a_body_of_exactly_4_kb_is_accepted_and_one_byte_more_is_not(db: str) -> None:
    async with app_client(make_settings(database_url=db), wall_clock=clock()) as client:
        accepted = await post_raw(client, body_of(MAX_BODY_BYTES))
        rejected = await post_raw(client, body_of(MAX_BODY_BYTES + 1))

    assert accepted.status_code == 200
    assert rejected.status_code == 422
    error = rejected.json()["error"]
    assert (error["code"], error["retry_after_s"]) == ("bad_request", None)
    assert "4096 bytes" in error["message"]
    [answered, refused] = rows(db)
    assert (answered["outcome"], refused["outcome"]) == ("answered", "bad_request")
    assert refused["question"] is None  # a rejected body keeps no text


async def test_an_oversized_body_is_refused_before_it_is_parsed(db: str) -> None:
    async with app_client(make_settings(database_url=db), wall_clock=clock()) as client:
        # Not JSON at all: a parser would complain about the JSON, the size rule about the bytes.
        response = await post_raw(client, b"{" * (MAX_BODY_BYTES + 1))

    assert response.status_code == 422
    assert "4096 bytes" in response.json()["error"]["message"]


async def test_a_chunked_body_over_4_kb_is_refused_without_a_content_length(db: str) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(MAX_BODY_BYTES // 512 + 1):
            yield b" " * 512

    async with app_client(make_settings(database_url=db), wall_clock=clock()) as client:
        response = await post_raw(client, chunks())

    assert response.status_code == 422
    assert "4096 bytes" in response.json()["error"]["message"]


@pytest.mark.parametrize(
    ("question", "rule"),
    [
        ("  hi  ", "at least 3 characters"),  # trimmed before the length check
        ("x" * 501, "at most 500 characters"),
        ("how do I\x00run it?", "control characters"),
        ("how do I\nrun it?", "control characters"),
    ],
)
async def test_a_question_that_breaks_a_rule_is_a_422_that_does_not_echo_it(
    db: str, question: str, rule: str
) -> None:
    async with app_client(make_settings(database_url=db), wall_clock=clock()) as client:
        response = await ask(client, question)

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "bad_request"
    assert rule in error["message"]
    assert question.strip() not in response.text
    [row] = rows(db)
    assert (row["outcome"], row["question"]) == ("bad_request", None)


async def test_a_lone_surrogate_is_a_422_with_a_hash_and_no_text(db: str) -> None:
    # httpx cannot JSON-encode a lone surrogate, so the escape is written into the raw body. The
    # model rejects the string, and the handler then hashes it for the log: that used to raise
    # UnicodeEncodeError and answer 500 (R.28).
    body = b'{"question": "How do I \\ud800 declare a path?"}'
    async with app_client(make_settings(database_url=db), wall_clock=clock()) as client:
        response = await post_raw(client, body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "bad_request"
    assert "declare a path" not in response.text
    [row] = rows(db)
    assert (row["outcome"], row["question"]) == ("bad_request", None)
    assert row["question_hash"] is not None


async def test_a_padded_question_is_trimmed_and_answered(db: str) -> None:
    async with app_client(make_settings(database_url=db), wall_clock=clock()) as client:
        response = await ask(client, f"  \n{QUESTION}\t ")

    assert response.status_code == 200
    [row] = rows(db)
    assert row["question"] == QUESTION


# --- the rate limit ----------------------------------------------------------------------------


def limited(db: str, **overrides: Any) -> Any:
    return make_settings(database_url=db, ip_hash_secret=IP_KEY, **overrides)


async def test_the_request_over_the_minute_limit_is_a_429_with_retry_after(db: str) -> None:
    async with app_client(limited(db), client=(PEER, 1), wall_clock=clock()) as client:
        sent = [await ask(client) for _ in range(6)]

    assert [r.status_code for r in sent] == [200] * 5 + [429]
    over = sent[5]
    error = over.json()["error"]
    assert error["code"] == "rate_limited"
    assert error["retry_after_s"] == 30  # 10:00:30 to the end of the minute
    assert over.headers["Retry-After"] == "30"
    assert error["message"] == "Too many requests. Try again in 30s."
    assert over.json()["request_id"] == str(rows(db)[5]["id"])
    last = rows(db)[5]
    assert (last["outcome"], last["http_status"], last["error_code"]) == (
        "rate_limited",
        429,
        "rate_limited",
    )
    assert last["ip_hash"] is not None


async def test_the_day_limit_is_a_429_with_the_seconds_to_midnight(db: str) -> None:
    wall = clock()
    settings = limited(db, rate_limit_per_min=100, rate_limit_per_day=2)
    async with app_client(settings, client=(PEER, 1), wall_clock=wall) as client:
        sent = [await ask(client) for _ in range(3)]

    assert [r.status_code for r in sent] == [200, 200, 429]
    error = sent[2].json()["error"]
    assert error["retry_after_s"] == 13 * 3600 + 59 * 60 + 30  # 10:00:30 to 24:00:00
    assert sent[2].headers["Retry-After"] == str(error["retry_after_s"])
    assert error["message"] == "Daily question limit reached. Try again in about 14 h."


async def test_a_cache_hit_counts_against_the_limit(db: str) -> None:
    # The same question every time: the 2nd to 5th are answer-cache hits, and still counted.
    async with app_client(limited(db), client=(PEER, 1), wall_clock=clock()) as client:
        sent = [await ask(client) for _ in range(6)]

    answered = [r for r in rows(db) if r["outcome"] == "answered"]
    assert sum(1 for r in answered if r["cache_hit"]) == 4
    assert sent[5].status_code == 429


async def test_invalid_requests_do_not_use_up_the_limit(db: str) -> None:
    async with app_client(limited(db), client=(PEER, 1), wall_clock=clock()) as client:
        refused = [await ask(client, "x") for _ in range(10)]
        sent = [await ask(client) for _ in range(5)]

    assert {r.status_code for r in refused} == {422}
    assert [r.status_code for r in sent] == [200] * 5


async def test_different_clients_have_their_own_limit(db: str) -> None:
    async with app_client(limited(db), client=(PEER, 1), wall_clock=clock()) as first:
        assert [(await ask(first)).status_code for _ in range(6)][-1] == 429
    async with app_client(limited(db), client=("198.51.100.77", 1), wall_clock=clock()) as other:
        assert (await ask(other)).status_code == 200


async def test_requests_without_any_client_ip_share_one_bucket(db: str) -> None:
    # D54e: no X-Client-IP and no socket peer, with IP_HASH_SECRET set, is hash_ip("unknown").
    settings = limited(db, rate_limit_per_min=1)
    async with app_client(settings, no_peer=True, wall_clock=clock()) as first:
        assert (await ask(first)).status_code == 200
    async with app_client(settings, no_peer=True, wall_clock=clock()) as second:
        assert (await ask(second)).status_code == 429  # the same bucket as the first request

    unknown = hmac.new(IP_KEY.encode(), b"unknown", hashlib.sha256).hexdigest()
    assert [r["ip_hash"] for r in rows(db)] == [unknown, unknown]


async def test_without_a_client_key_nothing_is_limited(db: str) -> None:
    # No IP_HASH_SECRET (dev and test only): ip_hash is NULL and the limiter has nothing to count.
    async with app_client(
        make_settings(database_url=db), client=(PEER, 1), wall_clock=clock()
    ) as client:
        sent = [await ask(client) for _ in range(8)]

    assert [r.status_code for r in sent] == [200] * 8


async def test_a_failing_counter_query_is_an_internal_error_not_a_pass(
    db: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def broken(self: RateLimiter, ip_hash: str | None) -> None:
        raise psycopg.OperationalError("the counter query failed")

    monkeypatch.setattr(RateLimiter, "check", broken)
    async with app_client(
        limited(db), client=(PEER, 1), raise_app_exceptions=False, wall_clock=clock()
    ) as client:
        response = await ask(client)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    [row] = rows(db)
    assert row["outcome"] == "internal_error"


# --- eval mode ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"app_env": "test"}, [200] * 5 + [429] * 3),
        (EVAL_SETTINGS, [200] * 8),
    ],
    ids=["test", "eval"],
)
async def test_eval_mode_bypasses_the_limiter(
    db: str, tmp_path: Path, overrides: dict[str, Any], expected: list[int]
) -> None:
    settings = limited(db, cache_dir=tmp_path / "cache", **overrides)
    async with app_client(settings, client=(PEER, 1), wall_clock=clock()) as client:
        sent = [(await ask(client)).status_code for _ in range(8)]

    assert sent == expected  # RATE_LIMIT_PER_MIN is 5
    if overrides is EVAL_SETTINGS:
        with psycopg.connect(db) as conn:
            assert conn.execute("SELECT count(*) FROM rate_limit_counters").fetchone() == (0,)
