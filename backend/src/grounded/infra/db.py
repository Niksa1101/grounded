"""Async Postgres connection pool for the request path (DB.md §2)."""

from __future__ import annotations

from pgvector.psycopg import register_vector_async  # pyright: ignore[reportMissingTypeStubs]
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from grounded.settings import Settings


async def _configure_connection(conn: AsyncConnection) -> None:
    # Registers the pgvector type adapters on each new connection. Requires the `vector` extension,
    # i.e. migrations applied; until then connections fail to configure and /readyz reports it.
    await register_vector_async(conn)


def create_pool(settings: Settings) -> AsyncConnectionPool:
    """Build the runtime pool without opening it; the app lifespan owns open/close.

    Opening happens with ``wait=False`` so the process (and ``/healthz``) starts even when the
    database is down or asleep (Neon scale-to-zero). ``/readyz`` reports whether it is reachable.
    """
    return AsyncConnectionPool(
        conninfo=settings.database_url.get_secret_value(),
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout_s,
        configure=_configure_connection,
        open=False,
        name="grounded",
    )
