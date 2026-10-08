"""Eval-mode wrappers around an ``LLMProvider``: the eval LLM cache and the 429 backoff (4.03).

``APP_ENV=eval`` puts two decorators in front of an adapter, outermost first:

- ``CachingProvider`` stores each successful reply (raw text and usage) in
  ``.cache/llm_eval.sqlite`` and replays it for an identical call, so a re-run costs no quota and
  answers the same (Tech §11).
- ``BackoffProvider`` waits out a per-minute 429 for the ``Retry-After`` the provider sent, within a
  bounded total, and asks again. A daily quota, or a wait that would pass the bound, is raised.

Both implement ``LLMProvider``, so the adapters stay as they are and the pipeline (and the judge,
4.04) cannot tell a wrapped provider from a bare one. The generator and the judge go through the
same ``EvalLLM``: one cache file, one set of counters. The backoff sleeps, which only the eval path
may do: the request path never waits out a ``Retry-After`` (Tech §9.5), and the adapters themselves
never sleep (the Groq client is built with ``max_retries=0``), so all eval waiting is here.

**What a hit is.** The stored text is validated again with Pydantic exactly like a live reply
(AGENTS.md §6.2). A hit gives back the original call's usage, so tokens and shadow cost add up on
a cached run as on a live one, with ``cache_hit=True`` and the lookup time as ``latency_ms``: a
cached hit has no provider latency, and latency statistics skip results with the flag. An entry that
does not validate (a corrupt file, a validator changed under an unchanged schema) is deleted and
counts as a miss, so the live reply replaces it. That is one extra live call, never a loop: nothing
here retries. The pipeline's own retry sends a different user message, hence a different key.
Only successful replies are stored: a bad output or an error is never cached.

**What the key is made of.** Everything that changes the reply: the provider, the model, the call's
temperature and output cap, the system and user text, the schema (a hash of its JSON Schema) and the
settings the adapter reads by itself (``params.adapter_params``: thinking level, reasoning effort).
The timeout is not in it: it decides whether a reply arrives, not what it says.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from functools import cache

from pydantic import BaseModel, ConfigDict, ValidationError

from grounded.generation.params import adapter_params
from grounded.generation.providers.base import GenerationResult, LLMProvider, Usage
from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import ProviderRateLimited
from grounded.infra.timing import Clock
from grounded.settings import Settings

logger = logging.getLogger(__name__)

LLM_EVAL_CACHE_FILE = "llm_eval.sqlite"

type Sleep = Callable[[float], Awaitable[None]]


@dataclass(slots=True)
class EvalStats:
    """Counters for one run: what the cache saved and how long the backoff waited.

    ``hits + misses`` is the number of ``generate`` calls. An entry that failed validation counts
    in ``invalid_entries`` and in ``misses`` (a live call replaced it).
    """

    hits: int = 0
    misses: int = 0
    invalid_entries: int = 0
    rate_limit_waits: int = 0
    waited_s: float = 0.0

    def snapshot(self) -> EvalStats:
        """A copy, so a caller can subtract the counters of an earlier moment (one case's share)."""
        return replace(self)

    def render(self) -> str:
        return (
            f"Eval LLM cache: {self.hits} hits, {self.misses} misses ({self.hits + self.misses} "
            f"calls); rate-limit waits: {self.rate_limit_waits} ({self.waited_s:g}s); "
            f"unusable entries replaced: {self.invalid_entries}"
        )


# --- The cache ------------------------------------------------------------------------------------


@cache
def _schema_hash(schema: type[BaseModel]) -> str:
    canonical = json.dumps(schema.model_json_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def eval_cache_key(
    *,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    system: str,
    user: str,
    schema: type[BaseModel],
    adapter_settings: dict[str, str],
) -> str:
    """sha256 of the parts as a JSON array, not joined with ``|``, so a ``|`` in a prompt can never
    shift a field into its neighbor."""
    parts = [
        provider,
        model,
        float(temperature),
        max_output_tokens,
        system,
        user,
        _schema_hash(schema),
        dict(sorted(adapter_settings.items())),
    ]
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode("utf-8")).hexdigest()


class _StoredGeneration(BaseModel):
    """What one cache entry holds. ``extra="forbid"`` so an entry of another layout is unusable
    (a miss), not half-read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_text: str
    usage: Usage


class CachingProvider:
    """``LLMProvider`` that replays the stored reply of an identical call (module docstring)."""

    def __init__(
        self,
        inner: LLMProvider,
        cache: KVCache,
        adapter_settings: dict[str, str],
        *,
        stats: EvalStats,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.name = inner.name
        self.model = inner.model
        self._inner = inner
        self._cache = cache
        self._adapter_settings = adapter_settings
        self._stats = stats
        self._clock = clock

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
        key = eval_cache_key(
            provider=self.name,
            model=self.model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system=system,
            user=user,
            schema=schema,
            adapter_settings=self._adapter_settings,
        )
        stored = (await asyncio.to_thread(self._cache.get_many, [key])).get(key)
        if stored is not None:
            result = self._replay(stored, schema, started)
            if result is not None:
                self._stats.hits += 1
                return result
            # Unusable: delete it, or the reply below could not take its place (first value wins).
            await asyncio.to_thread(self._cache.delete_many, [key])
            self._stats.invalid_entries += 1
        self._stats.misses += 1
        result = await self._inner.generate(
            system=system,
            user=user,
            schema=schema,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            timeout_s=timeout_s,
        )
        entry = _StoredGeneration(raw_text=result.raw_text, usage=result.usage)
        await asyncio.to_thread(self._cache.put_many, {key: entry.model_dump_json().encode()})
        return result

    def _replay[T: BaseModel](
        self, stored: bytes, schema: type[T], started: float
    ) -> GenerationResult[T] | None:
        """The stored reply as a ``GenerationResult``, or ``None`` if it no longer validates."""
        try:
            entry = _StoredGeneration.model_validate_json(stored)
            parsed = schema.model_validate_json(entry.raw_text)
        except ValidationError as exc:
            # The type only: the entry holds model output, which a log line must not repeat.
            logger.warning(
                "unusable eval cache entry, replacing it",
                extra={"provider": self.name, "model": self.model, "error": type(exc).__name__},
            )
            return None
        return GenerationResult(
            parsed=parsed,
            raw_text=entry.raw_text,
            usage=entry.usage,
            provider=self.name,
            model=self.model,
            latency_ms=round((self._clock() - started) * 1000),
            cache_hit=True,
        )


# --- The backoff ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackoffPolicy:
    """Bounds on waiting out per-minute 429s, per ``generate`` call."""

    max_total_wait_s: float  # ``EVAL_MAX_TOTAL_WAIT_S``
    default_wait_s: float = 60.0  # a 429 with no Retry-After: the per-minute window (Tech §10)
    max_retries: int = 3  # also ends a loop of tiny or zero waits, which the total would not


class BackoffExhaustedError(ProviderRateLimited):
    """The eval backoff gave up on a per-minute 429: the wait it was asked for would pass the total
    bound, or the retries ran out. Still a ``ProviderRateLimited`` (``is_quota`` is False), so
    whatever sorts errors by type sees a rate-limited call; the subclass says the waiting has been
    done already, and ``ask_batch`` therefore does not wait again.

    A daily quota is not wrapped: the original ``ProviderRateLimited`` with ``is_quota=True`` is
    raised at once, since no wait helps.
    """

    def __init__(self, message: str, *, retry_after_s: float | None, waited_s: float) -> None:
        super().__init__(message, retry_after_s=retry_after_s, is_quota=False)
        self.waited_s = waited_s


class BackoffProvider:
    """``LLMProvider`` that waits out per-minute 429s for the advertised ``Retry-After``.

    The total waited per call is bounded by the policy; the sleep function is injected so tests
    never sleep. Only a 429 is handled: a 5xx, a timeout, bad output and a rejected request pass
    through unchanged (and a failed call is not repeated here).
    """

    def __init__(
        self,
        inner: LLMProvider,
        policy: BackoffPolicy,
        *,
        stats: EvalStats,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.name = inner.name
        self.model = inner.model
        self._inner = inner
        self._policy = policy
        self._stats = stats
        self._sleep = sleep

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
        waited_s = 0.0
        retries = 0
        while True:
            try:
                return await self._inner.generate(
                    system=system,
                    user=user,
                    schema=schema,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    timeout_s=timeout_s,
                )
            except ProviderRateLimited as exc:
                if exc.is_quota:
                    raise  # a daily quota: waiting won't help, and the call is provider-errored
                advertised = exc.retry_after_s
                wait_s = max(0.0, self._policy.default_wait_s if advertised is None else advertised)
                if retries >= self._policy.max_retries:
                    raise BackoffExhaustedError(
                        f"still rate limited after {retries} waits ({waited_s:g}s)",
                        retry_after_s=exc.retry_after_s,
                        waited_s=waited_s,
                    ) from exc
                bound_s = self._policy.max_total_wait_s
                if waited_s + wait_s > bound_s:
                    raise BackoffExhaustedError(
                        f"Retry-After {wait_s:g}s would pass the {bound_s:g}s total wait "
                        f"(already waited {waited_s:g}s)",
                        retry_after_s=exc.retry_after_s,
                        waited_s=waited_s,
                    ) from exc
                retries += 1
                waited_s += wait_s
                self._stats.rate_limit_waits += 1
                self._stats.waited_s += wait_s
                logger.info(
                    "rate limited, waiting for the advertised Retry-After",
                    extra={"provider": self.name, "wait_s": wait_s, "retry": retries},
                )
                await self._sleep(wait_s)


# --- Wiring ---------------------------------------------------------------------------------------


class EvalLLM:
    """The eval-mode kit: one cache file and one set of counters for every provider that goes
    through ``wrap`` (the generator now, the judge in 4.04)."""

    def __init__(
        self,
        settings: Settings,
        cache: KVCache,
        *,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.perf_counter,
    ) -> None:
        self.stats = EvalStats()
        self._settings = settings
        self._cache = cache
        self._policy = BackoffPolicy(max_total_wait_s=settings.eval_max_total_wait_s)
        self._sleep = sleep
        self._clock = clock

    @classmethod
    def open(
        cls,
        settings: Settings,
        stack: AsyncExitStack,
        *,
        sleep: Sleep = asyncio.sleep,
        clock: Clock = time.perf_counter,
    ) -> EvalLLM:
        """The kit over ``<CACHE_DIR>/llm_eval.sqlite``, closed with ``stack``."""
        cache = stack.enter_context(KVCache(settings.cache_dir / LLM_EVAL_CACHE_FILE))
        return cls(settings, cache, sleep=sleep, clock=clock)

    def wrap(self, provider: LLMProvider) -> CachingProvider:
        """``provider`` behind the backoff and, outermost, the cache: a hit never reaches either."""
        backoff = BackoffProvider(provider, self._policy, stats=self.stats, sleep=self._sleep)
        return CachingProvider(
            backoff,
            self._cache,
            adapter_params(self._settings, provider.name),
            stats=self.stats,
            clock=self._clock,
        )
