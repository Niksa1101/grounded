"""The error shape of ``POST /v1/ask`` (Tech.md §13) for failures that need no database."""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest
from starlette.requests import Request

from grounded.api.errors import _bad_request  # pyright: ignore[reportPrivateUsage]
from tests.support import app_client, make_settings

# Nothing listens on port 1; the connection is refused locally (no external network).
_UNREACHABLE_DB = "postgresql://grounded:grounded@127.0.0.1:1/grounded"


def assert_error(body: dict[str, Any], code: str) -> None:
    assert body["error"]["code"] == code
    assert body["error"]["retry_after_s"] is None
    UUID(body["request_id"])  # a valid UUID


@pytest.mark.parametrize("question", ["ab", "x" * 501])
async def test_question_length_is_a_422_bad_request(question: str) -> None:
    settings = make_settings(database_url=_UNREACHABLE_DB, db_pool_min_size=0)
    async with app_client(settings) as client:
        response = await client.post("/v1/ask", json={"question": question})
    assert response.status_code == 422
    body = response.json()
    assert_error(body, "bad_request")
    assert body["error"]["message"].startswith("Invalid request. question: ")
    # The rejected input is not echoed back. The random request_id is left out of the search: a
    # UUID can contain a short question such as "ab" by chance.
    assert question not in response.text.replace(body["request_id"], "")


async def test_missing_question_is_a_422_bad_request() -> None:
    settings = make_settings(database_url=_UNREACHABLE_DB, db_pool_min_size=0)
    async with app_client(settings) as client:
        response = await client.post("/v1/ask", json={})
    assert response.status_code == 422
    assert_error(response.json(), "bad_request")


async def test_unhandled_error_is_a_500_with_a_request_id_and_no_stack_trace() -> None:
    # With the database down the very first stage (reading the active index) fails: nothing the
    # error shape promised for "anything else".
    settings = make_settings(
        database_url=_UNREACHABLE_DB, db_pool_min_size=0, db_pool_timeout_s=0.5
    )
    async with app_client(settings, raise_app_exceptions=False) as client:
        response = await client.post("/v1/ask", json={"question": "How do I run a task?"})
    assert response.status_code == 500
    assert_error(response.json(), "internal_error")
    assert response.json()["error"]["message"] == "Internal error."
    assert "Traceback" not in response.text
    assert "psycopg" not in response.text


async def test_the_bad_request_handler_given_another_error_answers_internal_error() -> None:
    # It is registered for RequestValidationError only; anything else reaching it is a wiring bug,
    # answered as "anything else" instead of an assert that python -O would strip.
    request = Request({"type": "http", "method": "POST", "path": "/v1/ask", "headers": []})
    response = await _bad_request(request, ValueError("not a validation error"))  # pyright: ignore[reportPrivateUsage]
    assert response.status_code == 500
    assert b'"internal_error"' in response.body
