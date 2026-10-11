"""The proxy secret and the client IP rules of ``api/security.py`` (5.03, Tech §12).

The HTTP-level cases use an unreachable database on purpose: the secret is checked before anything
touches it, so a 401 needs no database, and a test that wrongly got past the check would fail on
the connection instead of passing by accident. The rules themselves (which IP is trusted, what is
hashed) are tested on ``authenticate`` directly, with a hand-built request.
"""

from __future__ import annotations

import hashlib
import hmac
import logging

import pytest
from pydantic import SecretStr
from starlette.requests import Request

from grounded.api.security import ClientIdentity, ProxyAuthError, authenticate
from grounded.infra.hashing import hash_ip
from grounded.settings import Settings
from tests.support import app_client, make_settings

SECRET = "proxy-secret-for-tests"
IP_KEY = "ip-hash-key-for-tests"
PEER = "198.51.100.9"
VISITOR = "203.0.113.7"
_UNREACHABLE_DB = "postgresql://grounded:grounded@127.0.0.1:1/grounded"


def expected_hash(ip: str, key: str = IP_KEY) -> str:
    """The HMAC spelled out, so the test does not just call the code under test."""
    return hmac.new(key.encode(), ip.encode(), hashlib.sha256).hexdigest()


def settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "proxy_shared_secret": SECRET,
        "ip_hash_secret": IP_KEY,
        "allow_direct_api": False,
    }
    return make_settings(**{**base, **overrides})


def request_with(headers: dict[str, str], peer: str | None = PEER) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode("utf-8")) for k, v in headers.items()],
            "client": (peer, 5000) if peer is not None else None,
        }
    )


def offline_settings() -> Settings:
    return settings(database_url=_UNREACHABLE_DB, db_pool_min_size=0)


# --- the secret --------------------------------------------------------------------------------


async def test_a_missing_secret_is_a_401_in_the_shared_error_shape() -> None:
    async with app_client(offline_settings(), direct_api=False) as client:
        response = await client.post("/v1/ask", json={"question": "What is FastAPI?"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert body["error"]["retry_after_s"] is None
    assert body["request_id"]
    assert SECRET not in response.text


@pytest.mark.parametrize("supplied", ["wrong", "", SECRET + "x", SECRET[:-1], "sécret-é"])
async def test_a_wrong_secret_is_a_401(supplied: str) -> None:
    async with app_client(offline_settings(), direct_api=False) as client:
        response = await client.post(
            "/v1/ask",
            json={"question": "What is FastAPI?"},
            headers=[(b"x-proxy-secret", supplied.encode("latin-1"))],
        )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


async def test_the_secret_is_checked_before_the_body_is_read() -> None:
    # An unauthenticated caller learns nothing about the validation rules, and a malformed body
    # does not turn into a 422 that bypasses the check.
    async with app_client(offline_settings(), direct_api=False) as client:
        response = await client.post(
            "/v1/ask", content=b"{not json", headers={"Content-Type": "application/json"}
        )
    assert response.status_code == 401


async def test_health_probes_stay_public() -> None:
    async with app_client(offline_settings(), direct_api=False) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200


def test_a_secret_header_is_refused_when_no_secret_is_configured() -> None:
    # Not even an empty header matches an empty (unset) secret.
    config = settings(proxy_shared_secret=None, allow_direct_api=True)
    for supplied in ("", "anything"):
        with pytest.raises(ProxyAuthError) as caught:
            authenticate(request_with({"X-Proxy-Secret": supplied}), config)
        assert caught.value.reason == "invalid"


def test_the_correct_secret_is_the_proxy() -> None:
    identity = authenticate(
        request_with({"X-Proxy-Secret": SECRET, "X-Client-IP": VISITOR}), settings()
    )
    assert identity == ClientIdentity(source="web", ip_hash=expected_hash(VISITOR))


def test_a_wrong_secret_is_refused_even_when_direct_access_is_allowed() -> None:
    with pytest.raises(ProxyAuthError) as caught:
        authenticate(request_with({"X-Proxy-Secret": "wrong"}), settings(allow_direct_api=True))
    assert caught.value.reason == "invalid"


def test_direct_access_works_only_when_allowed() -> None:
    with pytest.raises(ProxyAuthError) as caught:
        authenticate(request_with({}), settings(allow_direct_api=False))
    assert caught.value.reason == "missing"
    identity = authenticate(request_with({}), settings(allow_direct_api=True))
    assert identity.source == "api"


def test_the_prod_check_still_refuses_direct_access() -> None:
    with pytest.raises(ValueError, match="ALLOW_DIRECT_API"):
        settings(
            app_env="prod", database_url="postgresql://u:p@db.example/x", allow_direct_api=True
        )


# --- the client IP -----------------------------------------------------------------------------


def test_x_client_ip_is_ignored_without_the_secret() -> None:
    identity = authenticate(request_with({"X-Client-IP": VISITOR}), settings(allow_direct_api=True))
    assert identity == ClientIdentity(source="api", ip_hash=expected_hash(PEER))


def test_the_proxy_without_a_client_ip_falls_back_to_the_peer() -> None:
    identity = authenticate(request_with({"X-Proxy-Secret": SECRET}), settings())
    assert identity == ClientIdentity(source="web", ip_hash=expected_hash(PEER))


def test_an_unusable_client_ip_falls_back_to_the_peer_and_is_not_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    garbage = "203.0.113.7, 10.0.0.1"
    with caplog.at_level(logging.DEBUG):
        identity = authenticate(
            request_with({"X-Proxy-Secret": SECRET, "X-Client-IP": garbage}), settings()
        )
    assert identity.ip_hash == expected_hash(PEER)
    assert [r.getMessage() for r in caplog.records] == ["proxy sent no usable client ip"]
    assert "203.0.113.7" not in caplog.text


def test_one_client_has_one_hash_whatever_the_spelling() -> None:
    def hashed(ip: str) -> str | None:
        headers = {"X-Proxy-Secret": SECRET, "X-Client-IP": ip}
        return authenticate(request_with(headers), settings()).ip_hash

    assert hashed("::ffff:203.0.113.7") == hashed(f" {VISITOR} ") == expected_hash(VISITOR)
    assert hashed("2001:DB8::1") == expected_hash("2001:db8::1")


def test_without_an_ip_hash_secret_there_is_no_hash() -> None:
    headers = {"X-Proxy-Secret": SECRET, "X-Client-IP": VISITOR}
    assert authenticate(request_with(headers), settings(ip_hash_secret=None)).ip_hash is None
    blank = settings(ip_hash_secret=SecretStr(""))
    assert authenticate(request_with(headers), blank).ip_hash is None


def test_without_a_peer_or_a_client_ip_there_is_no_hash() -> None:
    config = settings(allow_direct_api=True)
    assert authenticate(request_with({}, peer=None), config).ip_hash is None


def test_hash_ip_is_a_keyed_hmac_sha256() -> None:
    assert hash_ip(VISITOR, IP_KEY.encode()) == expected_hash(VISITOR)
    assert hash_ip(VISITOR, b"another key") != hash_ip(VISITOR, IP_KEY.encode())
    assert VISITOR not in hash_ip(VISITOR, IP_KEY.encode())
