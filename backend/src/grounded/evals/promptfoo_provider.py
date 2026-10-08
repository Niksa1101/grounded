"""The promptfoo Python provider: the real pipeline, in-process, in eval mode (Tech §15.3, 4.05).

``eval/promptfoo/provider.py`` re-exports ``call_api`` from here. promptfoo keeps one persistent
Python worker per provider entry of the config (here ``no_rag`` and ``hybrid``) and calls
``call_api(prompt, options, context)`` in it once per test, one at a time (``-j 1``). Module state
survives between those calls, which is what ``PipelineRunner`` relies on.

**Why a loop thread.** promptfoo runs an ``async def call_api`` with a plain ``asyncio.run`` per
call, which on Windows is the Proactor loop where psycopg async cannot run, and it would also open
and close the connection pool for every call. So ``call_api`` is synchronous: it hands the work to
one event loop that lives in a thread for the whole process (``infra/event_loop.py`` picks a loop
psycopg can use) and opens the runtime (pool, embedder, provider behind the eval LLM cache) once,
lazily, on that loop.

**Result of one call** (the dict promptfoo reads, camelCase as written):

- success: ``output`` is the ``AskResponse`` as a JSON object; ``tokenUsage`` and ``cost`` (the
  shadow cost) come from the request trace; ``latencyMs`` is the pipeline's own total; ``metadata``
  holds what the assertions and the gate read (``success_metadata``);
- a provider-side or infrastructure failure: ``error`` only, **tagged** ``"[<ErrorClass>
  quota=<true|false>] <short message>"``, and ``metadata`` with ``error_kind`` (the exception's
  class name, ``BackoffExhaustedError`` included), ``error_bases`` (the names of its base classes,
  so a reader can tell that ``BackoffExhaustedError`` is a ``ProviderRateLimited``) and
  ``is_quota``, plus whatever the request had spent. promptfoo keeps ``error`` and ``metadata`` on
  the results row (checked in the 4.05 run), which is how the generation gate (4.08) counts
  provider-errored cases. Which tags count as "provider reasons" is 4.08's decision.
- anything else (a bug, a bad configuration such as a missing key) is raised and shows up as a
  promptfoo error with the traceback.

**promptfoo must not retry a tagged error.** Its scheduler re-calls a provider (up to 3 more times,
with delays) when a result's ``error`` contains ``429`` or ``rate limit``, unless the result's
``metadata.rateLimitKind`` is ``"quota"`` (read from the 0.123.1 source and confirmed in the 4.05
run, where a "rate limited" message made a run wait). Our messages can contain those words, and a
retry would call the quota-limited API again on top of the eval backoff (Tech §15.6), so every
tagged error carries ``rateLimitKind: "quota"``. It is a switch for promptfoo, not our tag: read
``error_kind`` and ``is_quota``.

**A stopped run is not retried.** A daily quota or a rejected request (a bad key) cannot recover
inside the run, so after the first one the provider answers every later call with the same tag and
"skipped", without calling anything: a free-tier quota is never asked again and again (AGENTS.md
§6.15). A per-minute 429 is not that: the eval backoff (Tech §15.6) already waited it out, and the
next question may succeed.

Privacy: no question text is logged here; the failure message is the exception's, shortened.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final, Protocol, cast
from uuid import UUID, uuid4

import psycopg

from grounded.evals.ask_batch import MAX_REASON_CHARS
from grounded.generation.pipeline import AskMode, IndexMismatchError
from grounded.infra.event_loop import new_event_loop
from grounded.infra.logging import configure_logging
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
)
from grounded.infra.timing import Clock, StageTimer
from grounded.ingest.embed import EmbedderUnavailableError
from grounded.observability.cost import Pricing, load_pricing
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.index import NoActiveIndexError
from grounded.runtime import ProviderConfigError, open_runtime
from grounded.schemas.api import AskResponse
from grounded.settings import Settings, get_settings

logger = logging.getLogger(__name__)

CLOSE_TIMEOUT_S: Final = 15.0

# Failures that are a result about the provider or the setup, not a bug: they become tagged errors.
# Everything else propagates, so a bug is loud.
_TAGGED_FAILURES: Final = (
    ProviderError,
    EmbedderUnavailableError,
    NoActiveIndexError,
    IndexMismatchError,
    psycopg.Error,
)
_IGNORED_BASES: Final = (Exception, BaseException, object)

# promptfoo's "do not retry this result" switch (see the module docstring), on every tagged error.
_NO_PROMPTFOO_RETRY: Final = {"rateLimitKind": "quota"}


class Asker(Protocol):
    """What the provider needs from the pipeline (``AskPipeline`` satisfies it)."""

    async def ask(
        self,
        question: str,
        request_id: UUID,
        *,
        mode: AskMode = ...,
        trace: RequestTrace | None = ...,
    ) -> AskResponse: ...


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One call's promptfoo dict, and whether the run cannot go on after it."""

    payload: dict[str, Any]
    stop_run: bool = False


# --- One call ----------------------------------------------------------------------------------


async def answer(
    asker: Asker,
    question: str,
    mode: AskMode,
    *,
    pricing: Pricing,
    cold_start: bool = False,
    clock: Clock = time.perf_counter,
) -> CaseResult:
    """Ask once and build the promptfoo result: the answer, or the tagged failure."""
    request_id = uuid4()
    trace = RequestTrace(request_id=request_id, timer=StageTimer(clock))
    try:
        response = await asker.ask(question, request_id, mode=mode, trace=trace)
    except _TAGGED_FAILURES as exc:
        return CaseResult(
            failure_payload(exc, trace, mode, pricing=pricing, cold_start=cold_start),
            stop_run=cannot_recover(exc),
        )
    return CaseResult(success_payload(response, trace, mode, cold_start=cold_start))


def cannot_recover(exc: BaseException) -> bool:
    """A daily quota or a rejected request: asking again in this run cannot work."""
    return (isinstance(exc, ProviderRateLimited) and exc.is_quota) or isinstance(
        exc, ProviderRequestRejected
    )


def success_payload(
    response: AskResponse, trace: RequestTrace, mode: AskMode, *, cold_start: bool
) -> dict[str, Any]:
    meta = response.meta
    return {
        "output": response.model_dump(mode="json"),
        "tokenUsage": {**_token_usage(trace), "numRequests": 1 + trace.validation_retries},
        "cost": meta.shadow_cost_usd,
        "latencyMs": meta.latency_ms["total"],
        "metadata": success_metadata(response, trace, mode, cold_start=cold_start),
    }


def success_metadata(
    response: AskResponse, trace: RequestTrace, mode: AskMode, *, cold_start: bool
) -> dict[str, Any]:
    """What the assertions and the gate read besides ``output`` (Tech §15.3).

    ``invalid_citation_count`` is the count of the attempt that produced the returned answer, taken
    before the invalid labels were removed; an attempt that was retried shows in
    ``validation_retries``. ``context`` is what the model could cite, with the chunk text (the judge
    reads it, 4.06), keyed back to the answer by ``chunk_id``; ``retrieved_section_ids`` is the
    whole retrieval in rank order. Both are empty in ``no_rag``. A case with ``llm_cache_hits > 0``
    has a replayed reply, so its latency says nothing about the provider and is left out of the
    latency statistics; ``cold_start`` marks the first call of a worker, which pays for the
    connection pool and the embedder.
    """
    chunk_by_citation = {c.n: c.chunk_id for c in response.citations}
    meta = response.meta
    return {
        **_common_metadata(trace, mode, cold_start=cold_start),
        "status": response.status,
        "provider": meta.provider,
        "model": meta.model,
        "prompt_version": meta.prompt_version,
        "index_version": meta.index_version,
        "retrieval_config_hash": meta.retrieval_config_hash,
        "latency_ms": dict(meta.latency_ms),
        "shadow_cost_usd": meta.shadow_cost_usd,
        "tokens": dict(meta.tokens),
        "claims": [
            {
                "text": claim.text,
                "citations": list(claim.citations),
                "chunk_ids": [chunk_by_citation[n] for n in claim.citations],
                "confidence": claim.confidence,
                "confidence_components": dict(claim.confidence_components),
            }
            for claim in response.claims
        ],
        "context": [
            {
                "label": label,
                "chunk_id": chunk.chunk_id,
                "section_id": chunk.section_id,
                "anchor_path": list(chunk.anchor_path),
                "url": chunk.url,
                "breadcrumb": chunk.breadcrumb_text,
                "content": chunk.content,
            }
            for label, chunk in trace.context_chunks.items()
        ],
        "retrieved_section_ids": list(trace.retrieved_section_ids),
    }


def _common_metadata(trace: RequestTrace, mode: AskMode, *, cold_start: bool) -> dict[str, Any]:
    return {
        "mode": mode.value,
        "cold_start": cold_start,
        "validation_retries": trace.validation_retries,
        "invalid_citation_count": trace.invalid_citation_count,
        "dropped_claim_count": trace.dropped_claim_count,
        "removed_url_count": trace.removed_url_count,
        "llm_cache_hits": trace.llm_cache_hits,
    }


def _token_usage(trace: RequestTrace) -> dict[str, int]:
    """``completion`` is the provider's output count, which already holds the thinking tokens
    (``Usage.output_tokens``)."""
    prompt, completion = trace.input_tokens or 0, trace.output_tokens or 0
    return {"total": prompt + completion, "prompt": prompt, "completion": completion}


# --- Failures ----------------------------------------------------------------------------------


def error_tag(kind: str, is_quota: bool) -> str:
    return f"[{kind} quota={'true' if is_quota else 'false'}]"


def failure_payload(
    exc: Exception, trace: RequestTrace, mode: AskMode, *, pricing: Pricing, cold_start: bool
) -> dict[str, Any]:
    """The tagged error of Tech §15.3, with what the request had spent before it failed."""
    kind = type(exc).__name__
    is_quota = isinstance(exc, ProviderRateLimited) and exc.is_quota
    message = " ".join(str(exc).split())[:MAX_REASON_CHARS]
    metadata: dict[str, Any] = {
        **_common_metadata(trace, mode, cold_start=cold_start),
        "error_kind": kind,
        "error_bases": [c.__name__ for c in type(exc).__mro__[1:] if c not in _IGNORED_BASES],
        "is_quota": is_quota,
        **_NO_PROMPTFOO_RETRY,
    }
    if isinstance(exc, ProviderRateLimited):
        metadata["retry_after_s"] = exc.retry_after_s
        waited_s: object = getattr(exc, "waited_s", None)  # BackoffExhaustedError only
        if waited_s is not None:
            metadata["waited_s"] = waited_s
    if isinstance(exc, ProviderBadOutput):
        metadata["validation_error"] = exc.validation_error[:MAX_REASON_CHARS]
    payload: dict[str, Any] = {
        "error": f"{error_tag(kind, is_quota)} {message}".rstrip(),
        "cost": float(trace.shadow_cost(pricing)),
        "latencyMs": trace.total_ms(),
        "metadata": metadata,
    }
    if trace.input_tokens is not None or trace.output_tokens is not None:
        payload["tokenUsage"] = _token_usage(trace)
    return payload


def skipped_payload(first: Mapping[str, Any]) -> dict[str, Any]:
    """The answer to every call after the run stopped: the first error's tag, nothing called."""
    metadata: dict[str, Any] = {**first["metadata"], "skipped": True, "cold_start": False}
    kind, is_quota = str(metadata["error_kind"]), bool(metadata["is_quota"])
    return {
        "error": f"{error_tag(kind, is_quota)} skipped: not asked, the run stopped on an earlier "
        f"{kind}",
        "cost": 0.0,
        "metadata": metadata,
    }


# --- The long-lived part -----------------------------------------------------------------------


class PipelineRunner:
    """A loop thread and the pipeline opened on it, once, for the life of the worker process.

    ``open_asker`` is a factory so a test can open a pipeline of its own; the real one is
    ``open_pipeline``. ``call`` is synchronous and safe from any thread; calls are expected one at
    a time (promptfoo ``-j 1``), and the first one pays for opening everything.
    """

    def __init__(
        self,
        open_asker: Callable[[], AbstractAsyncContextManager[Asker]],
        *,
        pricing: Pricing,
        clock: Clock = time.perf_counter,
    ) -> None:
        self._open_asker = open_asker
        self._pricing = pricing
        self._clock = clock
        self._guard = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # The fields below belong to the loop thread.
        self._open_lock: asyncio.Lock | None = None
        self._stack: AsyncExitStack | None = None
        self._asker: Asker | None = None
        self._calls = 0
        self._halt: dict[str, Any] | None = None

    def call(self, question: str, mode: AskMode) -> dict[str, Any]:
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(self._call(question, mode), loop).result()

    def close(self) -> None:
        """Close what the first call opened and stop the loop thread. Safe to call twice."""
        with self._guard:
            loop, thread = self._loop, self._thread
            self._loop = self._thread = None
        if loop is None or thread is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._close_async(), loop).result(CLOSE_TIMEOUT_S)
        except TimeoutError:
            logger.warning("closing the pipeline timed out")
        loop.call_soon_threadsafe(loop.stop)
        thread.join(CLOSE_TIMEOUT_S)
        if not thread.is_alive():
            loop.close()

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._guard:
            if self._loop is None:
                loop = new_event_loop()
                thread = threading.Thread(
                    target=_serve, args=(loop,), name="promptfoo-pipeline-loop", daemon=True
                )
                thread.start()
                self._loop, self._thread = loop, thread
            return self._loop

    async def _call(self, question: str, mode: AskMode) -> dict[str, Any]:
        if self._halt is not None:
            return skipped_payload(self._halt)
        asker = await self._open()
        self._calls += 1
        result = await answer(
            asker,
            question,
            mode,
            pricing=self._pricing,
            cold_start=self._calls == 1,
            clock=self._clock,
        )
        if result.stop_run:
            self._halt = result.payload
        return result.payload

    async def _open(self) -> Asker:
        if self._open_lock is None:
            self._open_lock = asyncio.Lock()
        async with self._open_lock:
            if self._asker is None:
                stack = AsyncExitStack()
                try:
                    asker = await stack.enter_async_context(self._open_asker())
                except BaseException:
                    await stack.aclose()
                    raise
                self._stack, self._asker = stack, asker
            return self._asker

    async def _close_async(self) -> None:
        stack, self._stack, self._asker = self._stack, None, None
        if stack is not None:
            await stack.aclose()


def _serve(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


@asynccontextmanager
async def open_pipeline(settings: Settings) -> AsyncGenerator[Asker]:
    """The real pipeline of ``settings`` (pool, query embedder, provider behind the eval cache)."""
    async with open_runtime(settings) as runtime:
        yield runtime.pipeline


@lru_cache(maxsize=1)
def default_runner() -> PipelineRunner:
    """The worker's runner, built on the first call from the environment's settings."""
    settings = get_settings()
    if settings.app_env != "eval":
        raise ProviderConfigError(
            f"the promptfoo provider runs the pipeline in eval mode, got APP_ENV={settings.app_env}"
        )
    configure_logging(settings.log_level)
    runner = PipelineRunner(lambda: open_pipeline(settings), pricing=load_pricing())
    atexit.register(runner.close)
    return runner


def mode_of(options: Mapping[str, Any]) -> AskMode:
    """The provider entry's ``config.mode``: ``hybrid`` or ``no_rag``."""
    config: Any = options.get("config")
    mode: Any = cast(Mapping[str, Any], config).get("mode") if isinstance(config, Mapping) else None
    try:
        return AskMode(mode)
    except ValueError as exc:
        names = ", ".join(m.value for m in AskMode)
        raise ValueError(f"provider config.mode must be one of {names}, got {mode!r}") from exc


def call_api(prompt: str, options: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    """promptfoo's entry point. ``prompt`` is the rendered passthrough prompt, i.e. the question."""
    return default_runner().call(prompt, mode_of(options))
