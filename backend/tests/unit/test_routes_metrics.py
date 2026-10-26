"""The HTTP edges of ``GET /v1/metrics/summary`` (ticket 8.02, Tech §13): auth, the window rule
and the OpenAPI contract. The numbers are tested on the real database in
``tests/integration/test_metrics_summary.py``.

The app is built over an unreachable database on purpose, as in ``test_proxy_auth.py``: a 401 and
a 422 must be answered before anything touches the pool, so a test that wrongly got that far would
fail on the connection instead of passing by accident.
"""

from __future__ import annotations

from typing import Any

import pytest

from grounded.main import create_app
from grounded.settings import Settings
from tests.support import app_client, make_settings

SECRET = "proxy-secret-for-tests"
_UNREACHABLE_DB = "postgresql://grounded:grounded@127.0.0.1:1/grounded"
PATH = "/v1/metrics/summary"


def offline_settings() -> Settings:
    return make_settings(
        database_url=_UNREACHABLE_DB,
        db_pool_min_size=0,
        proxy_shared_secret=SECRET,
        allow_direct_api=False,
    )


@pytest.mark.parametrize("headers", [{}, {"X-Proxy-Secret": "wrong"}])
async def test_a_missing_or_wrong_secret_is_a_401_in_the_shared_shape(
    headers: dict[str, str],
) -> None:
    async with app_client(offline_settings(), direct_api=False) as client:
        response = await client.get(PATH, headers=headers)
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["request_id"]


async def test_the_secret_is_checked_before_the_window() -> None:
    # A stranger learns no parameter rule: a bad window without the secret is still a 401.
    async with app_client(offline_settings(), direct_api=False) as client:
        response = await client.get(PATH, params={"window": "nope"})
    assert response.status_code == 401


@pytest.mark.parametrize("window", ["bogus-window-value", "7", "7D", "365d", ""])
async def test_an_invalid_window_is_a_422_that_does_not_echo_it(window: str) -> None:
    async with app_client(offline_settings(), direct_api=False) as client:
        response = await client.get(
            PATH, params={"window": window}, headers={"X-Proxy-Secret": SECRET}
        )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "bad_request"
    assert "window" in body["error"]["message"]
    if window not in ("", "7"):  # too short to search for: "7" is in "7d", the allowed value
        assert window not in response.text


def _resolve(schema: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if ref is None:
        return node
    name = ref.rsplit("/", 1)[-1]
    return schema["components"]["schemas"][name]


def test_the_openapi_schema_describes_the_contract() -> None:
    schema = create_app(make_settings()).openapi()
    operation = schema["paths"][PATH]["get"]

    [param] = operation["parameters"]
    assert param["name"] == "window"
    assert param["in"] == "query"
    assert param["schema"]["enum"] == ["1d", "7d", "30d"]
    assert param["schema"]["default"] == "7d"

    responses = operation["responses"]
    ok = _resolve(schema, responses["200"]["content"]["application/json"]["schema"])
    assert ok["title"] == "MetricsSummary"
    for status in ("401", "422", "500"):
        error = responses[status]["content"]["application/json"]["schema"]
        assert error["$ref"].endswith("/ErrorResponse")

    assert set(ok["properties"]) == {
        "window",
        "generated_at",
        "requests",
        "rates",
        "latency_ms",
        "cost",
        "daily",
        "evals",
    }
    # Nothing private anywhere in the response models, at any depth.
    names = {
        "MetricsSummary",
        "RequestCounts",
        "Rates",
        "StageLatency",
        "Percentiles",
        "Cost",
        "DailyStats",
        "EvalRunSummary",
    }
    components = schema["components"]["schemas"]
    for name in names:
        fields = set(components[name]["properties"])
        assert not fields & {"question", "question_hash", "ip_hash", "id", "request_id"}, name
