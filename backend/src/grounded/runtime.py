"""Long-lived resources of the request path, built once and shared by the API and the CLI.

``open_runtime`` creates the connection pool, the query embedder, the generator provider and the
``AskPipeline`` over them, and closes what it opened. The API lifespan and ``grounded ask`` both go
through it, so they cannot drift apart. Tests pass their own ``embedder`` and ``provider``
(AGENTS.md §8: no real calls).
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from google import genai
from psycopg_pool import AsyncConnectionPool

from grounded.generation.pipeline import AskPipeline
from grounded.generation.prompts import load_answer_prompt, load_no_rag_prompt
from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.fake import StubLLMProvider
from grounded.generation.providers.gemini import GeminiProvider
from grounded.infra.db import create_pool
from grounded.infra.timing import Clock
from grounded.ingest.embed import Embedder
from grounded.observability.cost import Pricing, load_pricing
from grounded.observability.request_log import RequestLogger
from grounded.retrieval.index import ActiveIndexCache
from grounded.retrieval.query_embedding import build_query_embedder
from grounded.settings import Settings


class ProviderConfigError(Exception):
    """``GENERATOR_PROVIDERS`` names a provider this build cannot create."""


@dataclass(frozen=True, slots=True)
class Runtime:
    pool: AsyncConnectionPool
    pipeline: AskPipeline
    request_logger: RequestLogger


def build_provider(settings: Settings) -> LLMProvider:
    """The generator: the first entry of ``GENERATOR_PROVIDERS`` (no router before Phase 7).

    ``gemini`` is the real adapter; ``fake`` is a dev-only stand-in (refused in prod).
    """
    name = settings.generator_providers[0]
    if name == "gemini":
        return _build_gemini(settings)
    if name == "fake":
        if settings.app_env == "prod":
            raise ProviderConfigError("the fake provider must not be used with APP_ENV=prod")
        return StubLLMProvider()
    raise ProviderConfigError(
        f"no adapter for provider {name!r} yet; set GENERATOR_PROVIDERS=gemini "
        "(or fake for local runs)"
    )


def _build_gemini(settings: Settings) -> GeminiProvider:
    # A blank value counts as unset (an unfilled deploy secret arrives as ""), as in the embedder.
    key = settings.gemini_api_key.get_secret_value().strip() if settings.gemini_api_key else ""
    model = (settings.gemini_model or "").strip()
    if not key:
        raise ProviderConfigError("GEMINI_API_KEY must be set to use the gemini provider")
    if not model:
        raise ProviderConfigError(
            "GEMINI_MODEL must be set to a pinned model ID (PRD D46: gemini-3.5-flash-lite)"
        )
    return GeminiProvider(
        genai.Client(api_key=key), model=model, thinking_level=settings.gemini_thinking_level
    )


@asynccontextmanager
async def open_runtime(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    provider: LLMProvider | None = None,
    pricing: Pricing | None = None,
    clock: Clock = time.perf_counter,
) -> AsyncGenerator[Runtime]:
    async with AsyncExitStack() as stack:
        pool = create_pool(settings)
        # wait=False: start serving even if the database is unreachable; /readyz reports it.
        await pool.open(wait=False)
        stack.push_async_callback(pool.close)
        if provider is None:
            provider = build_provider(settings)
            if isinstance(provider, GeminiProvider):
                stack.push_async_callback(provider.aclose)
        pricing = pricing or load_pricing()
        pipeline = AskPipeline(
            settings=settings,
            pool=pool,
            index_cache=ActiveIndexCache(pool, ttl_s=settings.active_index_ttl_s),
            embedder=embedder or build_query_embedder(settings, stack),
            provider=provider,
            prompt=load_answer_prompt(),
            no_rag_prompt=load_no_rag_prompt(),
            pricing=pricing,
            clock=clock,
        )
        request_logger = RequestLogger(pool, pricing, clock=clock)
        yield Runtime(pool=pool, pipeline=pipeline, request_logger=request_logger)
