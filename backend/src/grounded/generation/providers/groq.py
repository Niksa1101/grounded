"""The Groq adapter behind the ``LLMProvider`` contract: the judge now, the fallback generator in
Phase 7 (Tech.md §9.1, PRD D48).

One call = one ``chat.completions.create`` request over the SDK's async client, with strict
structured output: the JSON Schema of the requested Pydantic model goes in ``response_format`` and
the reply comes back as JSON text in ``message.content``. The text is **always** validated with
Pydantic again (AGENTS.md §6.2).

Facts verified on 2026-10-08 against ``groq`` 1.7.0, Groq's docs and five real calls to
``openai/gpt-oss-120b`` (ticket 4.02; the fixtures in ``tests/fixtures/groq/`` are recordings of
some of them):

- **Strict mode rules.** Every property is in ``required`` and every object is closed
  (``additionalProperties: false``); ``to_groq_schema`` makes any Pydantic schema so (the
  ``follow_up_questions`` default is why ``LLMAnswer`` needs it).
- **Constraints are not decoded, they are checked after the call.** Strict mode accepts
  ``minLength``/``maxLength``/``minItems``/``maxItems``/``minimum``/``maximum``/``pattern`` in the
  schema, but the model is not held to them while it generates: Groq validates the finished JSON
  and answers a violation with ``400 json_validate_failed`` and the rejected text in
  ``failed_generation`` (recorded with ``maxLength``), *without* the usage of that call. Sending
  them would only turn a reply our own Pydantic check could report with its usage into a usage-less
  error, so ``to_groq_schema`` drops them like ``to_gemini_schema`` does.
- **That same 400 is how a cut-off reply arrives.** A ``max_completion_tokens`` that the reasoning
  and the answer use up gives ``400 json_validate_failed`` with an empty ``failed_generation``
  (recorded with a cap of 20), not a ``200`` with ``finish_reason=length``. It is bad output, not a
  rejected request: it becomes ``ProviderBadOutput`` and gets the one retry. A ``200`` with
  ``finish_reason=length`` is handled the same way (it was not seen).
- **Reasoning.** ``reasoning_effort`` comes from ``GROQ_REASONING_EFFORT``, and
  ``include_reasoning=False`` keeps the reasoning out of the reply (the message then has only
  ``content``). Reasoning tokens count inside ``max_completion_tokens`` and **inside**
  ``usage.completion_tokens``: a cap of 20 left the content empty, and a throwaway probe (not
  recorded) with 34 content tokens reported ``completion_tokens=114`` and ``reasoning_tokens=68``,
  the rest presumably being message framing. So ``output_tokens`` is ``completion_tokens`` as it
  comes and ``thinking_tokens`` is ``reasoning_tokens`` (unlike Gemini, nothing is added).
- A system message works (Groq's reasoning page advises against system prompts; no ill effect
  showed), and temperature ``0`` is accepted although Groq recommends 0.5 to 0.7 for reasoning
  models.
- **429.** ``retry-after`` (seconds) comes on a 429 only. Groq documents neither the body of a 429
  nor how a daily limit differs from a per-minute one, and no 429 could be provoked, so
  ``is_quota`` is a guess from two signals: the message names a per-day limit (``(TPD)``,
  ``(RPD)``, "per day"), or the wait is longer than the 60 s window of the per-minute limits.
  *Unverified until a real daily 429 is recorded.*

``max_retries=0`` is not optional: the SDK's default is two retries that *sleep* (it honors
``retry-after`` up to 60 s) on connection errors, 408, 409, 429 and 5xx, which would put sleeping in
the request path (Tech §9.5) and hide a 429 from the router. ``GroqProvider`` refuses a client that
has any.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import Any, Self, cast

import groq
import httpx
from groq import AsyncGroq
from groq.types.chat import ChatCompletion
from pydantic import BaseModel, ValidationError

from grounded.generation.providers.base import GenerationResult, Usage
from grounded.infra.provider_errors import (
    TRUNCATED_FEEDBACK,
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
    compact_validation_error,
)
from grounded.settings import GroqReasoningEffort

logger = logging.getLogger(__name__)

# The error code of Groq's own check of the generated JSON against the schema (see above).
_JSON_VALIDATE_FAILED = "json_validate_failed"

# The window of the per-minute limits (RPM, TPM): they cannot ask to wait longer, so a longer
# ``retry-after`` can only come from a daily limit.
_PER_MINUTE_WINDOW_S = 60.0
_DAILY_LIMIT_MARKERS = ("(tpd)", "(rpd)", "per day")

# Error messages go into exceptions and logs; the server's text is short, but cap it anyway.
_MAX_DETAIL_CHARS = 300

# The keywords strict mode documents (types, enums, ``anyOf``, ``$defs``/``$ref``, ``required``,
# ``additionalProperties``). Everything else, the constraints included, is left out.
_SUPPORTED_KEYWORDS = frozenset(
    {
        "$defs",
        "$ref",
        "type",
        "enum",
        "items",
        "anyOf",
        "properties",
        "required",
        "additionalProperties",
    }
)


def to_groq_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The Pydantic model's JSON Schema in the shape strict mode wants.

    Only the keywords in ``_SUPPORTED_KEYWORDS`` stay (``title`` and ``default`` are noise here, the
    constraints are checked by Pydantic after the call), every object lists all its properties as
    ``required`` and is closed with ``additionalProperties: false``. Property *names* are never
    treated as keywords: only the schema objects around them are filtered.
    """
    return _strict(schema.model_json_schema())


def _strict(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    schema = cast(dict[str, Any], node)
    strict: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SUPPORTED_KEYWORDS:
            continue
        if key in ("properties", "$defs"):  # name -> schema
            strict[key] = {name: _strict(sub) for name, sub in cast(dict[str, Any], value).items()}
        elif key == "anyOf":  # list of schemas
            strict[key] = [_strict(sub) for sub in cast(list[Any], value)]
        elif key == "items":  # a schema
            strict[key] = _strict(value)
        else:  # scalars, enum values
            strict[key] = value
    if "properties" in strict:
        strict["required"] = list(strict["properties"])
        strict["additionalProperties"] = False
    return strict


class GroqProvider:
    """``LLMProvider`` for a pinned Groq model. Errors are mapped to the typed provider errors;
    the pipeline (and later the router) decides what to do from the type alone."""

    name = "groq"

    def __init__(
        self,
        client: AsyncGroq,
        *,
        model: str,
        reasoning_effort: GroqReasoningEffort,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if client.max_retries != 0:
            raise ValueError(
                "the Groq client must be built with max_retries=0: the SDK's retries sleep in the "
                "request path and hide a 429 from the router (use GroqProvider.create)"
            )
        self._client = client
        self.model = model
        self._reasoning_effort: GroqReasoningEffort = reasoning_effort
        self._clock = clock

    @classmethod
    def create(cls, api_key: str, *, model: str, reasoning_effort: GroqReasoningEffort) -> Self:
        return cls(
            AsyncGroq(api_key=api_key, max_retries=0),
            model=model,
            reasoning_effort=reasoning_effort,
        )

    async def aclose(self) -> None:
        await self._client.close()

    async def generate[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> GenerationResult[T]:
        started = self._clock()
        try:
            async with asyncio.timeout(timeout_s):
                completion = await self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    max_completion_tokens=max_output_tokens,
                    reasoning_effort=self._reasoning_effort,
                    include_reasoning=False,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": _format_name(schema),
                            "strict": True,
                            "schema": to_groq_schema(schema),
                        },
                    },
                )
        except groq.APIStatusError as exc:
            raise _map_status_error(exc, schema) from exc
        except (TimeoutError, groq.APITimeoutError) as exc:
            raise ProviderTimeout(f"generation call exceeded {timeout_s:g}s") from exc
        except groq.APIConnectionError as exc:
            raise ProviderUnavailable(f"generation transport error: {exc}") from exc
        latency_ms = round((self._clock() - started) * 1000)

        usage = _usage(completion)  # read first: a reply that fails below was billed all the same
        choice = completion.choices[0] if completion.choices else None
        raw = (choice.message.content or "") if choice else ""
        parsed = _validate(raw, schema, choice.finish_reason if choice else "unknown", usage)
        return GenerationResult(
            parsed=parsed,
            raw_text=raw,
            usage=usage,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
        )


def _format_name(schema: type[BaseModel]) -> str:
    """The ``json_schema.name``: ``a-zA-Z0-9_-`` and 64 characters at most."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64]


def _validate[T: BaseModel](raw: str, schema: type[T], finish_reason: str, usage: Usage) -> T:
    try:
        return schema.model_validate_json(raw)
    except ValidationError as exc:
        # A ``length`` cut shows up here as invalid JSON; the finish reason says why.
        raise ProviderBadOutput(
            f"output does not match {schema.__name__} (finish_reason={finish_reason})",
            raw=raw,
            validation_error=(
                TRUNCATED_FEEDBACK if finish_reason == "length" else compact_validation_error(exc)
            ),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
        ) from exc


def _usage(completion: ChatCompletion) -> Usage:
    usage = completion.usage
    if usage is None:
        logger.warning("Groq response has no usage; counting zero tokens")
        return Usage(input_tokens=0, output_tokens=0)
    details = usage.completion_tokens_details
    return Usage(
        input_tokens=usage.prompt_tokens,
        # Reasoning is already inside completion_tokens (see the module docstring).
        output_tokens=usage.completion_tokens,
        thinking_tokens=details.reasoning_tokens if details else 0,
    )


# --- Errors ---------------------------------------------------------------------------------------


def _map_status_error(exc: groq.APIStatusError, schema: type[BaseModel]) -> ProviderError:
    error = _error_object(exc)
    detail = str(error.get("message") or exc.message)[:_MAX_DETAIL_CHARS]
    message = f"Groq API {exc.status_code}: {detail}"
    if isinstance(exc, groq.RateLimitError):
        retry_after_s = _retry_after_s(exc.response.headers)
        return ProviderRateLimited(
            message, retry_after_s=retry_after_s, is_quota=_is_daily_quota(detail, retry_after_s)
        )
    if exc.status_code >= 500:
        return ProviderUnavailable(message)
    if exc.status_code == 400 and error.get("code") == _JSON_VALIDATE_FAILED:
        return _bad_generation(str(error.get("failed_generation") or ""), schema, message)
    return ProviderRequestRejected(message, status_code=exc.status_code)


def _error_object(exc: groq.APIStatusError) -> dict[str, Any]:
    """The ``error`` object of the response body, or ``{}`` if the body has another shape."""
    body = exc.body
    error = cast(dict[str, Any], body).get("error") if isinstance(body, dict) else None
    return cast(dict[str, Any], error) if isinstance(error, dict) else {}


def _bad_generation(raw: str, schema: type[BaseModel], message: str) -> ProviderBadOutput:
    """Groq's own schema check failed: ``raw`` is the ``failed_generation`` it returned.

    The error carries no usage, so the call counts as 0 tokens (it may have been billed). The retry
    feedback is what our Pydantic check says about the text. Text that is not JSON at all, which
    strict decoding only produces when the output limit cut it off, gets the truncation sentence.
    """
    try:
        schema.model_validate_json(raw)
    except ValidationError as exc:
        cut_off = all(e["type"] == "json_invalid" for e in exc.errors())
        feedback = TRUNCATED_FEEDBACK if cut_off else compact_validation_error(exc)
    else:  # Groq is stricter than the model: quote its reason
        feedback = message
    return ProviderBadOutput(message, raw=raw, validation_error=feedback)


def _retry_after_s(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:  # an HTTP date instead of seconds: fall back to our own backoff
        return None


def _is_daily_quota(detail: str, retry_after_s: float | None) -> bool:
    if any(marker in detail.lower() for marker in _DAILY_LIMIT_MARKERS):
        return True
    return retry_after_s is not None and retry_after_s > _PER_MINUTE_WINDOW_S
