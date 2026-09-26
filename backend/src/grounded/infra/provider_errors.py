"""Typed provider errors shared by every external-API adapter (Tech.md §9.1, AGENTS.md §9).

Adapters translate SDK-specific exceptions into these, so callers (the embedder's retry loop now,
the LLM router in Phase 7) decide what to do from the type alone and never import an SDK.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class: something went wrong talking to an external provider."""


class ProviderRateLimited(ProviderError):
    """HTTP 429. ``is_quota`` means a daily quota, where waiting a few seconds won't help."""

    def __init__(self, message: str, *, retry_after_s: float | None, is_quota: bool) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s
        self.is_quota = is_quota


class ProviderUnavailable(ProviderError):
    """5xx or a transport failure (connection reset, DNS): worth retrying later."""


class ProviderTimeout(ProviderError):
    """The request didn't complete within our timeout."""


class ProviderBadOutput(ProviderError):
    """The provider answered, but the answer is unusable (wrong shape, dimension, count)."""

    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


class ProviderRequestRejected(ProviderError):
    """A 4xx other than 429 (bad request, bad key, forbidden): our fault, retrying won't help."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code
