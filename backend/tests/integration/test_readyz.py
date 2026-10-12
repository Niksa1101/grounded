from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest

from tests.support import app_client, insert_index_version, make_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def clean_db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    yield test_database_url
    wipe()


async def test_readyz_ok_with_an_active_index(clean_db: str) -> None:
    with psycopg.connect(clean_db) as conn:
        insert_index_version(conn, active=True, config_hash="ab" * 32)
    async with app_client(make_settings(database_url=clean_db)) as client:
        response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "database": "ok",
        "active_index_version": "0.0.1@abababab",
    }


async def test_readyz_fails_without_an_active_index(clean_db: str) -> None:
    with psycopg.connect(clean_db) as conn:  # a built but inactive version is not enough
        insert_index_version(conn, active=False, config_hash="ab" * 32)
    async with app_client(make_settings(database_url=clean_db)) as client:
        response = await client.get("/readyz")
    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "database": "ok",
        "active_index_version": None,
    }


SECRET = "proxy-secret-for-tests"


async def test_readyz_answers_200_with_the_proxy_secret(clean_db: str) -> None:
    with psycopg.connect(clean_db) as conn:
        insert_index_version(conn, active=True, config_hash="ab" * 32)
    settings = make_settings(database_url=clean_db, proxy_shared_secret=SECRET)
    async with app_client(settings, direct_api=False) as client:
        response = await client.get("/readyz", headers={"X-Proxy-Secret": SECRET})
    assert response.status_code == 200
    assert response.json()["active_index_version"] == "0.0.1@abababab"


@pytest.mark.parametrize("headers", [{}, {"X-Proxy-Secret": "wrong"}])
async def test_readyz_refuses_a_missing_or_wrong_secret_and_writes_no_row(
    clean_db: str, headers: dict[str, str]
) -> None:
    with psycopg.connect(clean_db) as conn:
        insert_index_version(conn, active=True, config_hash="ab" * 32)
        conn.execute("TRUNCATE request_logs")
    settings = make_settings(database_url=clean_db, proxy_shared_secret=SECRET)
    async with app_client(settings, direct_api=False) as client:
        response = await client.get("/readyz", headers=headers)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    with psycopg.connect(clean_db) as conn:
        assert conn.execute("SELECT count(*) FROM request_logs").fetchone() == (0,)
