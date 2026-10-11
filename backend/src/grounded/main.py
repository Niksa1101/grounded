"""FastAPI app factory and lifespan.

``create_app`` takes explicit settings so tests can build isolated apps. There is deliberately no
module-level ``app``: importing this module must not read settings. Serve it with ``grounded serve``
(or ``uvicorn --factory grounded.main:create_app``).
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.metadata import version

from fastapi import FastAPI

from grounded.api.errors import install_error_handlers
from grounded.api.routes_ask import router as ask_router
from grounded.api.routes_health import router as health_router
from grounded.generation.providers.base import LLMProvider
from grounded.infra.logging import configure_logging
from grounded.infra.timing import Clock
from grounded.ingest.embed import Embedder
from grounded.observability.cost import Pricing
from grounded.runtime import open_runtime
from grounded.settings import Settings, get_settings

logger = logging.getLogger(__name__)


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
) -> FastAPI:
    """``embedder`` and ``provider`` replace the real ones (tests); by default they are built from
    ``settings`` in the lifespan, once per process. ``pricing`` defaults to ``pricing.toml`` and
    ``clock`` to the monotonic clock; tests inject their own. ``wall_clock`` is the rate limiter's
    clock (default: the database's ``now()``, Tech §12)."""
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

    app = FastAPI(title="Grounded", version=version("grounded"), lifespan=lifespan)
    app.state.settings = settings
    install_error_handlers(app)
    app.include_router(health_router)
    app.include_router(ask_router)
    return app
