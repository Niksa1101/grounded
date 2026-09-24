"""FastAPI dependencies that expose app-scoped resources created in the lifespan."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request
from psycopg_pool import AsyncConnectionPool

from grounded.settings import Settings


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_pool(request: Request) -> AsyncConnectionPool:
    return cast(AsyncConnectionPool, request.app.state.db_pool)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
PoolDep = Annotated[AsyncConnectionPool, Depends(get_pool)]
