"""FastAPI app factory and lifespan.

``create_app`` takes explicit settings so tests can build isolated apps; the module-level ``app`` is
what uvicorn serves (``uvicorn grounded.main:app``).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import FastAPI

from grounded.api.routes_health import router as health_router
from grounded.infra.db import create_pool
from grounded.infra.logging import configure_logging
from grounded.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        pool = create_pool(settings)
        # wait=False: start serving even if the database is unreachable; /readyz reports it.
        await pool.open(wait=False)
        app.state.db_pool = pool
        try:
            yield
        finally:
            await pool.close()

    app = FastAPI(title="Grounded", version=version("grounded"), lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health_router)
    return app


app = create_app()
