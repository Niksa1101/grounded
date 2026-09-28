"""Which index version queries run against (DB.md §7.3): exactly one is active at a time."""

from __future__ import annotations

from typing import Any

from psycopg import AsyncConnection
from psycopg.rows import class_row

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
