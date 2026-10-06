"""Which index version queries run against (DB.md §7.3): exactly one is active at a time."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import class_row
from psycopg_pool import AsyncConnectionPool

from grounded.retrieval.types import IndexVersion


class NoActiveIndexError(Exception):
    """No index version is active: nothing to retrieve from."""


_ACTIVE_VERSION = """
SELECT id, git_ref, git_sha, embedding_model, embedding_dim, config_hash
FROM index_versions
WHERE is_active
"""


async def active_index_version(conn: AsyncConnection[Any]) -> IndexVersion:
    """The active version. The partial unique index guarantees at most one row, and the
    ``active_must_be_ready`` check that it is ``ready``."""
    cur = conn.cursor(row_factory=class_row(IndexVersion))
    await cur.execute(_ACTIVE_VERSION)
    version = await cur.fetchone()
    if version is None:
        raise NoActiveIndexError("no active index version: run `grounded ingest --activate`")
    return version


class ActiveIndexCache:
    """The active index version, re-read at most every ``ttl_s`` seconds (DB.md §7.3).

    Every request needs the version id and label, and it changes only when ingest activates a new
    one, so a short cache saves a query per request at the price of a stale answer for up to
    ``ttl_s`` after an activation. A failed refresh raises and is not cached.
    """

    def __init__(
        self,
        pool: AsyncConnectionPool,
        *,
        ttl_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pool = pool
        self._ttl_s = ttl_s
        self._clock = clock
        self._version: IndexVersion | None = None
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()  # one refresh at a time; the others wait for its result

    async def get(self) -> IndexVersion:
        if self._fresh():
            return self._cached()
        async with self._lock:
            if not self._fresh():  # another request may have refreshed while this one waited
                async with self._pool.connection() as conn:
                    self._version = await active_index_version(conn)
                self._loaded_at = self._clock()
            return self._cached()

    def _fresh(self) -> bool:
        return self._version is not None and self._clock() - self._loaded_at < self._ttl_s

    def _cached(self) -> IndexVersion:
        assert self._version is not None
        return self._version


_CHUNK_TITLES = """
SELECT c.id, d.title
FROM chunks AS c
JOIN documents AS d ON d.id = c.document_id
WHERE c.id = ANY(%(ids)s)
"""


async def chunk_titles(conn: AsyncConnection[Any], chunk_ids: Sequence[int]) -> dict[int, str]:
    """Page title (the H1) per chunk id: ``RetrievedChunk`` doesn't carry it, and it is only needed
    for the chunks that reach the context."""
    cur = await conn.execute(_CHUNK_TITLES, {"ids": list(chunk_ids)})
    return {int(row[0]): str(row[1]) for row in await cur.fetchall()}
