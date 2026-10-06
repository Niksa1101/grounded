"""Long-lived resources of the request path, built once and shared by the API and the CLI.

``open_runtime`` creates the connection pool, the query embedder, the generator provider and the
``AskPipeline`` over them, and closes what it opened. The API lifespan and ``grounded ask`` both go
through it, so they cannot drift apart. Tests pass their own ``embedder`` and ``provider``
(AGENTS.md §8: no real calls).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from psycopg_pool import AsyncConnectionPool

from grounded.generation.pipeline import AskPipeline
from grounded.generation.prompts import load_answer_prompt
from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.fake import StubLLMProvider
from grounded.infra.db import create_pool
from grounded.ingest.embed import Embedder
from grounded.retrieval.index import ActiveIndexCache
from grounded.retrieval.query_embedding import build_query_embedder
from grounded.settings import Settings


class ProviderConfigError(Exception):
    """``GENERATOR_PROVIDERS`` names a provider this build cannot create."""


@dataclass(frozen=True, slots=True)
class Runtime:
    pool: AsyncConnectionPool
    pipeline: AskPipeline


def build_provider(settings: Settings) -> LLMProvider:
    """The generator: the first entry of ``GENERATOR_PROVIDERS`` (no router before Phase 7)."""
    name = settings.generator_providers[0]
    if name == "fake":
        if settings.app_env == "prod":
            raise ProviderConfigError("the fake provider must not be used with APP_ENV=prod")
        return StubLLMProvider()
    raise ProviderConfigError(
        f"no adapter for provider {name!r} yet (Gemini arrives in ticket 3.08); "
        "set GENERATOR_PROVIDERS=fake for local runs"
    )


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    provider: LLMProvider | None = None,
) -> AsyncGenerator[Runtime]:
    async with AsyncExitStack() as stack:
        pool = create_pool(settings)
        # wait=False: start serving even if the database is unreachable; /readyz reports it.
        await pool.open(wait=False)
        stack.push_async_callback(pool.close)
        pipeline = AskPipeline(
            settings=settings,
            pool=pool,
            index_cache=ActiveIndexCache(pool, ttl_s=settings.active_index_ttl_s),
            embedder=embedder or build_query_embedder(settings, stack),
            provider=provider or build_provider(settings),
            prompt=load_answer_prompt(),
        )
        yield Runtime(pool=pool, pipeline=pipeline)
