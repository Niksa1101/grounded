"""FastAPI app factory and lifespan.

``create_app`` takes explicit settings so tests can build isolated apps. There is deliberately no
module-level ``app``: importing this module must not read settings. Serve it with ``grounded serve``
(or ``uvicorn --factory grounded.main:create_app``). On Vercel the thin entry module
``backend/app.py`` calls ``create_app()`` (Tech §18).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from grounded.api.errors import install_error_handlers
from grounded.api.routes_ask import router as ask_router
from grounded.api.routes_health import router as health_router
from grounded.api.routes_metrics import router as metrics_router
from grounded.generation.providers.base import LLMProvider
from grounded.generation.router import SYSTEM_CLOCK, RouterClock
from grounded.infra.logging import configure_logging
from grounded.infra.timing import Clock
from grounded.ingest.embed import Embedder
from grounded.observability.cost import Pricing
from grounded.retrieval.rerank import Reranker
from grounded.runtime import open_runtime
from grounded.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def _app_version() -> str:
    """The installed package version for the OpenAPI document. It must not be needed to serve: a
    bundle that carries the source but not the package metadata (Vercel, Tech §18) has none."""
    try:
        return version("grounded")
    except PackageNotFoundError:
        return "0+unknown"


def _has_ip_hash_key(settings: Settings) -> bool:
    secret = settings.ip_hash_secret
    return secret is not None and bool(secret.get_secret_value())


def create_app(
    settings: Settings | None = None,
    *,
    embedder: Embedder | None = None,
    provider: LLMProvider | None = None,
    pricing: Pricing | None = None,
    clock: Clock = time.perf_counter,
    wall_clock: Callable[[], datetime] | None = None,
    reranker: Reranker | None = None,
    fallback_providers: Sequence[LLMProvider] | None = None,
    router_clock: RouterClock = SYSTEM_CLOCK,
) -> FastAPI:
    """``embedder``, ``provider`` and ``reranker`` replace the real ones (tests); by default they
    are built from ``settings`` in the lifespan, once per process (no reranker for
    ``RERANK_PROVIDER=none``). ``pricing`` defaults to ``pricing.toml`` and ``clock`` to the
    monotonic clock; tests inject their own. ``wall_clock`` is the rate limiter's clock (default:
    the database's ``now()``, Tech §12). The provider router, with the breaker state that outlives
    a request, is built once per process in the lifespan; a test gives it ``fallback_providers``
    (an injected ``provider`` has none otherwise) and a ``router_clock`` it moves by hand."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        async with open_runtime(
            settings,
            embedder=embedder,
            provider=provider,
            pricing=pricing,
            clock=clock,
            wall_clock=wall_clock,
            reranker=reranker,
            fallback_providers=fallback_providers,
            router_clock=router_clock,
        ) as runtime:
            app.state.db_pool = runtime.pool
            app.state.pipeline = runtime.pipeline
            app.state.request_logger = runtime.request_logger
            app.state.rate_limiter = runtime.rate_limiter
            if runtime.rate_limiter is not None and not _has_ip_hash_key(settings):
                # Prod refuses to start without the key (Settings), so this is a dev or test run.
                logger.warning(
                    "IP_HASH_SECRET is unset: requests have no client key and are not rate limited"
                )
            yield

    app = FastAPI(title="Grounded", version=_app_version(), lifespan=lifespan)
    app.state.settings = settings
    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(ask_router)
    app.include_router(metrics_router)
    return app
