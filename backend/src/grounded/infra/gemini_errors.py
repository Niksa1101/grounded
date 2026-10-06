"""Gemini API errors → the typed provider errors (Tech.md §9.1).

Shared by the embedder (``ingest/embed.py``) and the generator (``generation/providers/gemini.py``):
one place knows how Google reports a rate limit, so the two adapters cannot drift apart.

A 429 body carries ``google.rpc`` detail objects: a ``RetryInfo`` with the suggested delay and a
``QuotaFailure`` naming the violated quota. A quota ID with ``PerDay`` in it means the daily quota,
where waiting seconds or minutes won't help (``ProviderRateLimited.is_quota``).
"""

from __future__ import annotations

import re
from typing import Any, cast

import httpx
from google.genai import errors as genai_errors

from grounded.infra.provider_errors import (
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderUnavailable,
)


def map_api_error(exc: genai_errors.APIError) -> ProviderError:
    message = f"Gemini API {exc.code} {exc.status}: {exc.message}"
    if exc.code == 429:
        # ``details`` is the parsed error body and ``response`` the raw HTTP response; the SDK
        # types both loosely, so read them as Any and check shapes ourselves.
        raw = cast(Any, exc)
        details = _error_details(raw.details)
        headers = getattr(raw.response, "headers", None)
        return ProviderRateLimited(
            message,
            retry_after_s=_retry_after_s(details, headers),
            is_quota=_is_daily_quota(details),
        )
    if isinstance(exc, genai_errors.ServerError):
        return ProviderUnavailable(message)
    return ProviderRequestRejected(message, status_code=exc.code)


def _dicts(value: Any) -> list[dict[str, Any]]:
    """The dict items of ``value`` if it is a list; JSON from the wire has no guaranteed shape."""
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], item) for item in cast(list[Any], value) if isinstance(item, dict)]


def _error_details(details: Any) -> list[dict[str, Any]]:
    """The ``google.rpc`` detail objects of an error body (``{"error": {"details": [...]}}``)."""
    if not isinstance(details, dict):
        return []
    body = cast(dict[str, Any], details)
    inner = body.get("error", body)
    return _dicts(cast(dict[str, Any], inner).get("details")) if isinstance(inner, dict) else []


def _retry_after_s(details: list[dict[str, Any]], headers: object) -> float | None:
    """``google.rpc.RetryInfo.retryDelay`` (e.g. ``"53s"``), else a ``Retry-After`` header."""
    for item in details:
        if str(item.get("@type", "")).endswith("google.rpc.RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    value = headers.get("retry-after") if isinstance(headers, httpx.Headers) else None
    try:
        return float(value) if value is not None else None
    except ValueError:  # an HTTP date instead of seconds: fall back to our own backoff
        return None


def _is_daily_quota(details: list[dict[str, Any]]) -> bool:
    """A ``google.rpc.QuotaFailure`` naming a per-day quota (``...PerDay...`` quota ID)."""
    for item in details:
        if str(item.get("@type", "")).endswith("google.rpc.QuotaFailure") and any(
            "PerDay" in str(v.get("quotaId", "")) for v in _dicts(item.get("violations"))
        ):
            return True
    return False
