"""``/v1/ask`` behind the proxy secret: what reaches the row and the logs (5.03, Tech §12).

The point of these tests is the negative: whatever the request, the raw client IP appears in no log
line, no stdout/stderr and no ``request_logs`` column (AGENTS.md §6.13). Both the visitor's address
(``X-Client-IP``) and the socket peer's are searched for.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
from collections.abc import Iterator
from typing import Any

import httpx
import psycopg
import pytest
from psycopg.rows import dict_row

from grounded.infra.logging import JsonFormatter
from grounded.settings import Settings
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import app_client, insert_index_version, make_settings

pytestmark = pytest.mark.integration

SECRET = "proxy-secret-for-tests"
IP_KEY = "ip-hash-key-for-tests"
PEER = "198.51.100.9"
VISITOR = "203.0.113.7"
QUESTION = "Where does the quokka sleep?"
PROXY = {"X-Proxy-Secret": SECRET, "X-Client-IP": VISITOR}


def expected_hash(ip: str) -> str:
    return hmac.new(IP_KEY.encode(), ip.encode(), hashlib.sha256).hexdigest()


def settings(url: str, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "database_url": url,
        "proxy_shared_secret": SECRET,
        "ip_hash_secret": IP_KEY,
        "allow_direct_api": False,
    }
    return make_settings(**{**base, **overrides})


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")

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


async def ask(
    url: str, headers: dict[str, str], question: str = QUESTION, **overrides: Any
) -> httpx.Response:
    async with app_client(
        settings(url, **overrides), client=(PEER, 4321), direct_api=False
    ) as client:
        return await client.post("/v1/ask", json={"question": question}, headers=headers)


async def test_the_proxy_gets_an_answer_and_a_web_row_with_the_hashed_visitor(db: str) -> None:
    response = await ask(db, PROXY)

    assert response.status_code == 200
    [row] = rows(db)
    assert (row["source"], row["outcome"]) == ("web", "answered")
    assert row["ip_hash"] == expected_hash(VISITOR)


async def test_x_client_ip_without_the_secret_is_ignored(db: str) -> None:
    # Direct access allowed (dev), a forged X-Client-IP, no secret: the peer is who is hashed.
    response = await ask(db, {"X-Client-IP": VISITOR}, allow_direct_api=True)

    assert response.status_code == 200
    [row] = rows(db)
    assert (row["source"], row["ip_hash"]) == ("api", expected_hash(PEER))


async def test_a_rejected_request_writes_no_row(db: str) -> None:
    for headers in ({}, {"X-Proxy-Secret": "wrong", "X-Client-IP": VISITOR}):
        response = await ask(db, headers)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthorized"
    assert rows(db) == []


async def test_a_failed_request_keeps_its_source_and_ip_hash(db: str) -> None:
    # The error path writes the row from the same trace, so the hash is on it too.
    response = await ask(db, PROXY, question="x")

    assert response.status_code == 422
    [row] = rows(db)
    assert (row["source"], row["outcome"], row["ip_hash"]) == (
        "web",
        "bad_request",
        expected_hash(VISITOR),
    )


async def test_no_raw_ip_reaches_a_log_line_the_output_or_a_row(
    db: str, caplog: pytest.LogCaptureFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    garbage = {**PROXY, "X-Client-IP": f"{VISITOR}, 10.0.0.1"}
    wrong = {"X-Proxy-Secret": "wrong", "X-Client-IP": VISITOR}
    with caplog.at_level(logging.DEBUG):
        async with app_client(settings(db), client=(PEER, 4321), direct_api=False) as client:
            sent = [
                await client.post("/v1/ask", json={"question": q}, headers=h)
                for q, h in (
                    (QUESTION, PROXY),
                    ("x", PROXY),
                    (QUESTION, garbage),
                    (QUESTION, wrong),
                    (QUESTION, {"X-Client-IP": VISITOR}),
                )
            ]
    assert [r.status_code for r in sent] == [200, 422, 200, 401, 401]

    formatter = JsonFormatter()
    lines = [formatter.format(record) for record in caplog.records]
    assert any("request completed" in line for line in lines)  # the search is not vacuous
    output = capsys.readouterr()
    haystacks = [*lines, output.out, output.err, str(rows(db))]
    for needle in (VISITOR, PEER, "10.0.0.1"):
        assert not [h for h in haystacks if needle in h], needle
