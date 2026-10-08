"""The contract every LLM adapter implements (Tech.md §9.1).

An adapter turns a Pydantic model into the provider's structured-output format, calls the API and
returns an **already validated** object, or raises one of the typed errors in
``infra/provider_errors.py``. Callers (the pipeline now, the router in Phase 7) decide what to do
from the exception type alone and never import an SDK.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token counts as the provider reported them (the input to the shadow cost, Tech §14)."""

    input_tokens: int = Field(ge=0)
    # Includes reasoning/thinking tokens when the provider reports them in the same count, since
    # that is what the provider bills as output.
    output_tokens: int = Field(ge=0)
    thinking_tokens: int = Field(default=0, ge=0)


class GenerationResult[T: BaseModel](BaseModel):
    parsed: T  # already Pydantic-validated against the requested schema
    raw_text: str
    usage: Usage
    provider: str
    model: str
    latency_ms: int = Field(ge=0)
    # True only for a result the eval LLM cache served (4.03, Tech §15.6). Its ``usage`` is the
    # original call's, so tokens and shadow cost add up on a cached run; its ``latency_ms`` is the
    # cache lookup, not a provider latency, so latency statistics must skip it.
    cache_hit: bool = False


class LLMProvider(Protocol):
    """``name`` and ``model`` identify the adapter in logs and in ``GenerationResult``; ``model``
    is a pinned ID, never a floating alias (AGENTS.md §6.8)."""

    name: str
    model: str

    async def generate[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float,
        max_output_tokens: int,
        timeout_s: float,
    ) -> GenerationResult[T]: ...
