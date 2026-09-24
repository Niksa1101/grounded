"""Shared fixtures.

Two guarantees for every test (AGENTS.md §8):
- **No real network.** Name resolution and socket connects to anything but localhost fail loudly.
- **Never a real database.** Integration tests get a fresh local ``*_test`` database, migrated
  once per session and dropped at the end. The fixture refuses any non-local host or non-``_test``
  name, so a misconfigured TEST_DATABASE_URL can't touch Neon.
"""

from __future__ import annotations

import asyncio
import socket
import sys
from collections.abc import Iterator
from typing import Any

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from grounded.infra.migrations import migrate
from tests.support import NetworkBlockedError, make_settings

if sys.platform == "win32":
    # psycopg's async mode can't run on the default Proactor loop on Windows.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # pyright: ignore[reportDeprecated]

_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _check_host(host: object) -> None:
    if host is None or (isinstance(host, str | bytes) and _decode(host) in _LOCAL_HOSTS):
        return
    raise NetworkBlockedError(f"tests must not call external hosts (attempted {host!r})")


def _decode(host: str | bytes) -> str:
    return host.decode() if isinstance(host, bytes) else host


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        _check_host(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_host(address[0])
        real_connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture(scope="session")
def test_database_url() -> Iterator[str]:
    """Create a fresh, migrated test database for the session and drop it afterwards."""
    url = make_settings().test_database_url.get_secret_value()
    params = conninfo_to_dict(url)
    host, dbname = str(params.get("host", "")), str(params.get("dbname", ""))
    if host not in _LOCAL_HOSTS or not dbname.endswith("_test"):
        pytest.fail(f"TEST_DATABASE_URL must be a local *_test database, got {host}/{dbname}")

    admin_url = make_conninfo(url, dbname="postgres")
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as exc:
        pytest.fail(
            "Integration tests need the local pgvector DB. Start it with "
            f"`docker compose -f infra/docker-compose.yml up -d db`. ({exc})"
        )

    drop = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(dbname))
    with admin:
        admin.execute(drop)
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(dbname)))
        migrate(url)
        yield url
        admin.execute(drop)
