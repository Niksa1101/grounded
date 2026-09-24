from __future__ import annotations

from tests.support import app_client, make_settings

# Nothing listens on port 1; the connection is refused locally (no external network).
_UNREACHABLE_DB = "postgresql://grounded:grounded@127.0.0.1:1/grounded"


async def test_healthz_does_not_need_the_database() -> None:
    settings = make_settings(database_url=_UNREACHABLE_DB, db_pool_min_size=0)
    async with app_client(settings) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_reports_unreachable_database_as_503() -> None:
    settings = make_settings(
        database_url=_UNREACHABLE_DB, db_pool_min_size=0, db_pool_timeout_s=0.5
    )
    async with app_client(settings) as client:
        response = await client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "database": "error",
        "active_index_version": None,
    }
