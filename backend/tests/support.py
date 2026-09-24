"""Test helpers importable from any test module (conftest.py is for fixtures only)."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx

from grounded.main import create_app
from grounded.settings import Settings


class NetworkBlockedError(RuntimeError):
    """Raised by the conftest network guard when a test tries to reach a non-local host."""


def make_settings(**overrides: Any) -> Settings:
    """Settings for tests: ignores any developer .env, but still honors real env vars (CI)."""
    return Settings(_env_file=None, **{"app_env": "test", **overrides})  # pyright: ignore[reportCallIssue]


@asynccontextmanager
async def app_client(settings: Settings) -> AsyncGenerator[httpx.AsyncClient]:
    """An app built from ``settings`` with its lifespan running, plus an in-process HTTP client."""
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
