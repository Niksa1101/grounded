"""``ActiveIndexCache`` against the real ``index_versions`` table, with a hand-driven clock."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from grounded.infra.db import create_pool
from grounded.retrieval.index import ActiveIndexCache, NoActiveIndexError
from tests.support import insert_index_version, make_settings

pytestmark = pytest.mark.integration

TTL_S = 300.0


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    yield test_database_url
    wipe()


@pytest.fixture
async def pool(db: str) -> AsyncIterator[AsyncConnectionPool]:
    pool = create_pool(make_settings(database_url=db))
    await pool.open()
    yield pool
    await pool.close()


def activate(db: str, version_id: int) -> None:
    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute("UPDATE index_versions SET is_active = (id = %s)", (version_id,))


async def test_version_is_cached_until_the_ttl_passes(db: str, pool: AsyncConnectionPool) -> None:
    with psycopg.connect(db) as conn:
        first = insert_index_version(conn, active=True, config_hash="1" * 64)
        second = insert_index_version(conn, active=False, config_hash="2" * 64)
    clock = Clock()
    cache = ActiveIndexCache(pool, ttl_s=TTL_S, clock=clock)

    assert (await cache.get()).id == first
    activate(db, second)  # ingest activates another version

    clock.now += TTL_S - 1
    assert (await cache.get()).id == first  # still the cached one
    clock.now += 1
    assert (await cache.get()).id == second  # refreshed once the TTL has passed


async def test_no_active_index_raises_and_is_not_cached(db: str, pool: AsyncConnectionPool) -> None:
    clock = Clock()
    cache = ActiveIndexCache(pool, ttl_s=TTL_S, clock=clock)
    with pytest.raises(NoActiveIndexError):
        await cache.get()

    with psycopg.connect(db) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
    assert (await cache.get()).id == version  # no waiting out the TTL after a failure
