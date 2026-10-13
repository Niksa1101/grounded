"""Who is calling ``/v1/*``: the proxy secret and the client IP (Tech.md §12, AGENTS.md §6.13).

The browser never talks to this API. The Next.js Route Handler does, and it sends two headers:
``X-Proxy-Secret`` (shared with this service) and ``X-Client-IP`` (the visitor, read from the
platform's forwarded-for header). So the two headers are trusted together or not at all:

- a valid secret means "this is the proxy": ``X-Client-IP`` is the client, ``source='web'``;
- no secret, with ``ALLOW_DIRECT_API=true`` (a developer's ``curl``; ``Settings`` refuses it in
  prod): the socket peer is the client, ``source='api'``, and a forged ``X-Client-IP`` is ignored;
- anything else, including a *wrong* secret even when direct access is allowed, is a 401. A caller
  that sends a secret meant to be the proxy, and gets it wrong, has a broken deploy that should be
  loud, not quietly treated as an anonymous developer.

**The raw IP lives only inside ``authenticate``.** It is HMAC-hashed at once and the string is
dropped, so nothing downstream (logs, the ``request_logs`` row, the rate limiter of 5.04) can leak
it. Without ``IP_HASH_SECRET`` (dev and test only; prod refuses to start without it) there is
nothing to key the HMAC with, and the hash is ``None``: no hash is better than an unkeyed one.
With the key, a request that has no client IP at all (no usable ``X-Client-IP`` and no socket
peer) is hashed as the sentinel ``"unknown"`` (PRD D54e): all such requests share one bucket of
the rate limiter, which is strict, never unlimited.

``authenticate`` runs before ``AskRoute`` starts the request trace, so a rejected request writes no
``request_logs`` row (``unauthorized`` is not an outcome in DB.md §4: unauthenticated callers must
not be able to fill the table). ``/readyz`` checks the secret alone (``check_proxy_secret``, PRD
D54c): it queries the database, so it must not add a free path that wakes Neon. (``/v1/ask`` touches
the database by design, so that risk is accepted, PRD D56b.)
``/healthz`` touches nothing and stays public.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
from dataclasses import dataclass
from typing import Literal

from fastapi import Request

from grounded.api.deps import get_settings_dep, request_id_of
from grounded.infra.hashing import hash_ip
from grounded.settings import Settings

logger = logging.getLogger(__name__)

PROXY_SECRET_HEADER = "X-Proxy-Secret"
CLIENT_IP_HEADER = "X-Client-IP"
# What a request with no client IP at all is hashed as (D54e). Not an address, so it can never be
# the hash of a real visitor.
UNKNOWN_CLIENT = "unknown"


class ProxyAuthError(Exception):
    """The request did not come from the proxy (or a developer allowed to call directly)."""

    def __init__(self, reason: Literal["missing", "invalid"]) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ClientIdentity:
    """What the rest of the request may know about the caller. There is no raw IP in here."""

    source: Literal["web", "api"]
    ip_hash: str | None


def _canonical_ip(value: str) -> str | None:
    """The address in one spelling (so ``::ffff:1.2.3.4`` and ``1.2.3.4`` hash alike), or ``None``
    when ``value`` is not an IP address at all (e.g. ``"1.2.3.4, 5.6.7.8"`` from a raw XFF)."""
    try:
        ip = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        return str(ip.ipv4_mapped)
    return str(ip)


def check_proxy_secret(request: Request, settings: Settings) -> bool:
    """True for a valid ``X-Proxy-Secret``; False for none at all when ``ALLOW_DIRECT_API`` allows
    it. Raises ``ProxyAuthError`` (-> 401) for a missing secret otherwise, and for a wrong one."""
    supplied = request.headers.get(PROXY_SECRET_HEADER)
    if supplied is None:
        if settings.allow_direct_api:
            return False
        raise ProxyAuthError("missing")
    configured = settings.proxy_shared_secret
    expected = configured.get_secret_value() if configured is not None else ""
    # Constant time, on bytes: compare_digest refuses a str with non-ASCII characters. An empty
    # expected secret (not configured) never matches, not even an empty header.
    if not expected or not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise ProxyAuthError("invalid")
    return True


def authenticate(request: Request, settings: Settings | None = None) -> ClientIdentity:
    """Check the proxy secret and hash the client IP. Raises ``ProxyAuthError`` (-> 401)."""
    settings = settings or get_settings_dep(request)
    proxied = check_proxy_secret(request, settings)

    ip: str | None = None
    if proxied:
        header = request.headers.get(CLIENT_IP_HEADER)
        ip = _canonical_ip(header) if header is not None else None
        if ip is None:
            # The proxy is trusted, so this is a bug on its side. Fall back to the socket peer
            # (every visitor then shares the proxy's bucket: strict, never lenient). The value
            # itself is not logged: it may be anything.
            logger.warning(
                "proxy sent no usable client ip",
                extra={"request_id": str(request_id_of(request))},
            )
    if ip is None and request.client is not None:
        peer = request.client.host
        ip = _canonical_ip(peer) or peer

    secret = settings.ip_hash_secret
    ip_hash = None
    if secret is not None and secret.get_secret_value():
        ip_hash = hash_ip(
            ip if ip is not None else UNKNOWN_CLIENT, secret.get_secret_value().encode()
        )
    return ClientIdentity(source="web" if proxied else "api", ip_hash=ip_hash)
