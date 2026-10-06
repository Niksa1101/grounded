"""The Gemini generator behind the ``LLMProvider`` contract (Tech.md §9.1, PRD D46).

One call = one ``generate_content`` request over the SDK's async client, with native structured
output: the JSON Schema of the requested Pydantic model goes in ``response_json_schema`` and the
reply comes back as JSON text. The text is **always** validated with Pydantic again (AGENTS.md
§6.2): the schema Gemini sees is only the subset it documents, so lengths and the citation-label
pattern are enforced here, after the call.

Facts verified on 2026-10-06 against the installed ``google-genai`` 2.25.0 and Google's docs:

- ``GenerateContentConfig.response_json_schema`` takes JSON Schema and supports only ``$id``,
  ``$defs``, ``$ref``, ``$anchor``, ``type``, ``format``, ``title``, ``description``, ``enum``,
  ``items``, ``prefixItems``, ``minItems``, ``maxItems``, ``minimum``, ``maximum``, ``anyOf``,
  ``oneOf``, ``properties``, ``additionalProperties`` and ``required``. ``to_gemini_schema`` keeps
  exactly those, so nothing undocumented is sent.
- ``ThinkingConfig.thinking_level`` takes ``minimal | low | medium | high``
  (``gemini-3.5-flash-lite`` accepts all four and defaults to ``minimal``; other 3.x models reject
  some).
- ``usage_metadata.candidates_token_count`` does **not** include thoughts: ``total = prompt +
  candidates + tool_use + thoughts``, and thinking is billed as output. So ``output_tokens`` is
  ``candidates + thoughts`` and ``thinking_tokens`` is ``thoughts``. ``max_output_tokens`` is a
  limit on candidates *and* thoughts together.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from grounded.generation.providers.base import GenerationResult, Usage
from grounded.infra.gemini_errors import map_api_error
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.settings import GeminiThinkingLevel

logger = logging.getLogger(__name__)

# The JSON Schema keywords Gemini documents for ``response_json_schema`` (see the module docstring).
_SUPPORTED_KEYWORDS = frozenset(
    {
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "type",
        "format",
        "title",
        "description",
        "enum",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "oneOf",
        "properties",
        "additionalProperties",
        "required",
    }
)


def to_gemini_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The Pydantic model's JSON Schema, reduced to the keywords Gemini documents.

    Dropped: ``minLength``/``maxLength``/``pattern`` (Pydantic enforces them after the call) and
    ``default`` (not documented). Property *names* are never treated as keywords: only the schema
    objects around them are filtered.
    """
    return _clean(schema.model_json_schema())


def _clean(node: Any) -> Any:
    if not isinstance(node, dict):
        return node
    schema = cast(dict[str, Any], node)
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SUPPORTED_KEYWORDS:
            continue
        if key in ("properties", "$defs"):  # name -> schema
            cleaned[key] = {name: _clean(sub) for name, sub in cast(dict[str, Any], value).items()}
        elif key in ("anyOf", "oneOf", "prefixItems"):  # list of schemas
            cleaned[key] = [_clean(sub) for sub in cast(list[Any], value)]
        elif key in ("items", "additionalProperties"):  # a schema (or a bool)
            cleaned[key] = _clean(value)
        else:  # scalars, enum values, "required" names
            cleaned[key] = value
    return cleaned


class GeminiProvider:
    """``LLMProvider`` for a pinned Gemini model. Errors are mapped to the typed provider errors;
    the pipeline (and later the router) decides what to do from the type alone."""

    name = "gemini"

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        thinking_level: GeminiThinkingLevel,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._client = client
        self.model = model
        self._thinking_level = genai_types.ThinkingLevel(thinking_level.upper())
        self._clock = clock

    async def aclose(self) -> None:
        await self._client.aio.aclose()

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
        config = genai_types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=to_gemini_schema(schema),
            thinking_config=genai_types.ThinkingConfig(thinking_level=self._thinking_level),
        )
        # Part of the SDK's content types refer to PIL (optional, not installed), which leaves the
        # method's signature partially unknown to pyright; pin the one shape we use.
        generate_content = cast(
            Callable[..., Awaitable[genai_types.GenerateContentResponse]],
            self._client.aio.models.generate_content,  # pyright: ignore[reportUnknownMemberType]
        )
        started = self._clock()
        try:
            async with asyncio.timeout(timeout_s):
                response = await generate_content(model=self.model, contents=user, config=config)
        except genai_errors.APIError as exc:
            raise map_api_error(exc) from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderTimeout(f"generation call exceeded {timeout_s:g}s") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"generation transport error: {exc}") from exc
        latency_ms = round((self._clock() - started) * 1000)

        raw = _response_text(response)
        parsed = _validate(raw, schema, _finish_reason(response))
        return GenerationResult(
            parsed=parsed,
            raw_text=raw,
            usage=_usage(response),
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
        )


def _response_text(response: genai_types.GenerateContentResponse) -> str:
    """The reply text, or ``ProviderBadOutput`` when there is none (blocked, or no candidate)."""
    if not response.candidates:
        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback else None
        raise ProviderBadOutput(
            f"Gemini returned no candidate (prompt block reason: {reason or 'none'})"
        )
    parts = response.candidates[0].content.parts if response.candidates[0].content else None
    # Thought parts (if any) are not part of the answer; ``include_thoughts`` is never set.
    return "".join(p.text for p in parts or [] if p.text and not p.thought)


def _finish_reason(response: genai_types.GenerateContentResponse) -> str:
    candidates = response.candidates or []
    reason = candidates[0].finish_reason if candidates else None
    return reason.name if reason else "unknown"


def _validate[T: BaseModel](raw: str, schema: type[T], finish_reason: str) -> T:
    try:
        return schema.model_validate_json(raw)
    except ValidationError as exc:
        # A MAX_TOKENS cut shows up here as invalid JSON; the finish reason says why.
        raise ProviderBadOutput(
            f"output does not match {schema.__name__} (finish_reason={finish_reason})",
            raw=raw,
            validation_error=str(exc),
        ) from exc


def _usage(response: genai_types.GenerateContentResponse) -> Usage:
    meta = response.usage_metadata
    if meta is None:
        logger.warning("Gemini response has no usage_metadata; counting zero tokens")
        return Usage(input_tokens=0, output_tokens=0)
    thoughts = meta.thoughts_token_count or 0
    return Usage(
        input_tokens=meta.prompt_token_count or 0,
        # Thinking is billed as output and is not part of candidates_token_count.
        output_tokens=(meta.candidates_token_count or 0) + thoughts,
        thinking_tokens=thoughts,
    )
