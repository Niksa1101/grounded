"""Typed provider errors shared by every external-API adapter (Tech.md §9.1, AGENTS.md §9).

Adapters translate SDK-specific exceptions into these, so callers (the embedder's retry loop now,
the LLM router in Phase 7) decide what to do from the type alone and never import an SDK.
"""

from __future__ import annotations

from pydantic import ValidationError


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
    """The provider answered, but the answer is unusable (wrong shape, dimension, count).

    ``validation_error`` is what the retry (Tech §9.5) quotes back to the model, so it is kept apart
    from the short ``message`` and written for the model (``compact_validation_error``).

    ``input_tokens`` / ``output_tokens`` are the usage of the call that produced the bad output:
    the provider billed it, so the request's shadow cost counts it (Tech §14). They are plain ints,
    not ``Usage``, so this module does not depend on the generation package; 0 means "not
    reported" (an embedding error, or an error raised by our own checks after the usage was
    already counted).

    ``retryable`` is False when the same request would fail the same way (a prompt or an answer
    blocked by the provider's content filter): the retry of Tech §9.5 is then skipped, since it
    would only spend a request of the daily quota.
    """

    def __init__(
        self,
        message: str,
        *,
        raw: str = "",
        validation_error: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        retryable: bool = True,
    ) -> None:
        super().__init__(message)
        self.raw = raw
        self.validation_error = validation_error
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.retryable = retryable


# The retry feedback for a reply cut off at the output limit (it fills the prompt's ``{{error}}``).
# Shared by the generator adapters: the JSON error alone would not tell the model why it broke.
TRUNCATED_FEEDBACK = (
    "the answer was cut off at the output token limit before the JSON was complete; "
    "write a shorter answer with fewer, shorter claims"
)


def compact_validation_error(exc: ValidationError) -> str:
    """The Pydantic errors as ``"<field path>: <message>"``, joined with ``"; "``.

    This is the reason the retry gives the model (Tech §9.5). ``str(ValidationError)`` would also
    carry every rejected ``input_value`` (up to the whole bad output again) and an
    ``errors.pydantic.dev`` link per error, which is noise for the model and contradicts the
    prompt's "do not write URLs". An error about the whole output (invalid JSON) has no field path
    and is reported against ``output``.
    """
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'output'}: {error['msg']}"
        for error in exc.errors(include_url=False, include_input=False)
    )


class ProviderRequestRejected(ProviderError):
    """A 4xx other than 429 (bad request, bad key, forbidden): our fault, retrying won't help."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code
