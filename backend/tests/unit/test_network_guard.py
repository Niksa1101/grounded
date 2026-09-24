"""The autouse guard in conftest.py must stop real network calls (AGENTS.md §8)."""

from __future__ import annotations

import socket

import httpx
import psycopg
import pytest

from tests.support import NetworkBlockedError


def test_dns_lookup_of_external_host_is_blocked() -> None:
    with pytest.raises(NetworkBlockedError):
        socket.getaddrinfo("example.com", 443)


def test_connect_to_external_ip_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkBlockedError),
    ):
        sock.connect(("93.184.215.14", 443))


async def test_http_client_cannot_reach_the_internet() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(NetworkBlockedError):
            await client.get("https://example.com")


def test_localhost_is_allowed() -> None:
    assert socket.getaddrinfo("localhost", 5433)


# psycopg connects through libpq's C code, which never touches Python's socket module, so the guard
# checks conninfo hosts at psycopg's connect entry points instead.
@pytest.mark.parametrize(
    "conninfo",
    [
        "postgresql://u@db.example.com/grounded",
        "host=localhost,db.example.com dbname=grounded",
        "host=localhost hostaddr=93.184.215.14 dbname=grounded",
    ],
)
def test_sync_psycopg_cannot_reach_external_hosts(conninfo: str) -> None:
    with pytest.raises(NetworkBlockedError):
        psycopg.connect(conninfo)


async def test_async_psycopg_cannot_reach_external_hosts() -> None:
    with pytest.raises(NetworkBlockedError):
        await psycopg.AsyncConnection.connect("postgresql://u@db.example.com/grounded")


def test_psycopg_host_keyword_is_checked() -> None:
    with pytest.raises(NetworkBlockedError):
        psycopg.connect("dbname=grounded", host="db.example.com")
