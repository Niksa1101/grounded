"""Shared fixtures.

Two guarantees for every test (AGENTS.md §8):
- **No real network.** Name resolution, socket connects and psycopg connections to anything but
  localhost fail loudly. psycopg is guarded separately because libpq opens its sockets in C, out of
  reach of the ``socket`` patch (a ``hostaddr=`` IP skips Python name resolution entirely).
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
    if isinstance(host, str) and host.startswith("/"):  # Unix-domain socket directory
        return
    raise NetworkBlockedError(f"tests must not call external hosts (attempted {host!r})")


def _decode(host: str | bytes) -> str:
    return host.decode() if isinstance(host, bytes) else host


def _check_conninfo(conninfo: str, kwargs: dict[str, Any]) -> None:
    params = conninfo_to_dict(conninfo)
    for key in ("host", "hostaddr"):
        value = kwargs.get(key) or params.get(key)
        # libpq accepts comma-separated lists for multi-host failover.
        for host in str(value or "").split(","):
            if host.strip():
                _check_host(host.strip())


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_psycopg_connect = psycopg.Connection.connect
    real_psycopg_connect_async = psycopg.AsyncConnection.connect

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        _check_host(host)
        return real_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if self.family in (socket.AF_INET, socket.AF_INET6):
            _check_host(address[0])
        real_connect(self, address)

    def guarded_psycopg_connect(conninfo: str = "", **kwargs: Any) -> psycopg.Connection[Any]:
        _check_conninfo(conninfo, kwargs)
        return real_psycopg_connect(conninfo, **kwargs)

    async def guarded_psycopg_connect_async(
        conninfo: str = "", **kwargs: Any
    ) -> psycopg.AsyncConnection[Any]:
        _check_conninfo(conninfo, kwargs)
        return await real_psycopg_connect_async(conninfo, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    # psycopg.connect is a module-level alias of Connection.connect, so both are patched. The pool
    # connects through AsyncConnection.connect.
    monkeypatch.setattr(psycopg, "connect", guarded_psycopg_connect)
    monkeypatch.setattr(psycopg.Connection, "connect", staticmethod(guarded_psycopg_connect))
    monkeypatch.setattr(
        psycopg.AsyncConnection, "connect", staticmethod(guarded_psycopg_connect_async)
    )


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
