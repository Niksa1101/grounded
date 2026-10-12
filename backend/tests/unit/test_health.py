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


# --- /readyz needs the proxy secret (PRD D54c) ---------------------------------------------------
# The database is unreachable here: a 401 proves the route never queried it (that would be a 503).

SECRET = "proxy-secret-for-tests"


async def test_readyz_without_the_secret_is_a_401_and_touches_no_database() -> None:
    settings = make_settings(
        database_url=_UNREACHABLE_DB,
        db_pool_min_size=0,
        db_pool_timeout_s=0.5,
        proxy_shared_secret=SECRET,
    )
    async with app_client(settings, direct_api=False) as client:
        response = await client.get("/readyz")
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["request_id"]


async def test_readyz_with_a_wrong_secret_is_a_401() -> None:
    settings = make_settings(
        database_url=_UNREACHABLE_DB, db_pool_min_size=0, proxy_shared_secret=SECRET
    )
    async with app_client(settings, direct_api=False) as client:
        response = await client.get("/readyz", headers={"X-Proxy-Secret": "not-it"})
    assert response.status_code == 401


async def test_readyz_with_the_secret_reaches_the_database_check() -> None:
    settings = make_settings(
        database_url=_UNREACHABLE_DB,
        db_pool_min_size=0,
        db_pool_timeout_s=0.5,
        proxy_shared_secret=SECRET,
    )
    async with app_client(settings, direct_api=False) as client:
        response = await client.get("/readyz", headers={"X-Proxy-Secret": SECRET})
    assert response.status_code == 503  # past the check: the (unreachable) database answered
    assert response.json()["database"] == "error"


async def test_healthz_stays_public() -> None:
    settings = make_settings(
        database_url=_UNREACHABLE_DB, db_pool_min_size=0, proxy_shared_secret=SECRET
    )
    async with app_client(settings, direct_api=False) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
