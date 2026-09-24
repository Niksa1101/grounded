from __future__ import annotations

import pytest

from tests.support import app_client, make_settings

pytestmark = pytest.mark.integration


async def test_readyz_ok_against_migrated_database(test_database_url: str) -> None:
    async with app_client(make_settings(database_url=test_database_url)) as client:
        response = await client.get("/readyz")
    assert response.status_code == 200
    # No index is built before Phase 1, so none is active yet.
    assert response.json() == {"status": "ok", "database": "ok", "active_index_version": None}
