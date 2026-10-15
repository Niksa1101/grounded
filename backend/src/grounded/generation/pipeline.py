"""The `/ask` flow: question, embedding, hybrid retrieval, context, generation, response.

``AskPipeline.ask`` is the one place the stages are wired together; the route stays thin and every
stage lives in its own module. Failures are raised as the typed errors of the stage that failed
(``ProviderError``, ``NoActiveIndexError``) and translated to HTTP in ``api/errors.py``.

Tracer-bullet state (Phase 3): the stages thicken in later tickets, and what is deliberately thin
today is marked below.

- Citation mapping lives in ``citations.py`` (3.06). A semantic failure (an answer with no valid
  citation) is treated like a schema failure: ``ProviderBadOutput``, which gets the one retry.
  ``invalid_citation_count`` is computed per request and recorded on the trace.
- Generation goes through the provider router (7.04, Tech §10): ``_generate_routed`` hands it the
  hooks and reads back the answering provider. The one retry with error feedback (3.07, Tech §9.5
  step 3) is the router's now, as are the fallback to the next provider and the circuit breaker;
  the pipeline has no retry loop of its own, so a bad output costs two calls on a provider, not
  four. The hooks are the budget reserve before **every** call (a retry and a fallback included),
  the citation checks (``validate``), the retry wording (``Prompt.render_retry``), the usage record
  of every call (``on_attempt``: tokens, retries, the provider that answered) and the time left
  before the request deadline. The router is built here from ``Settings`` and the providers
  ``open_runtime`` passes, once per process, so the breaker state outlives a request.
- Claim confidence comes from ``confidence.score_claims`` (3.10): the server computes it from the
  retrieval signals of the cited chunks, and the model's ``self_confidence`` is only one weak input,
  never shown as confidence (AGENTS.md §6.6).
- ``RequestTrace`` (3.12) is filled as the stages run: versions, usage, retries and citation counts
  for the ``request_logs`` row, and the stage timings and shadow cost for ``meta``. The pipeline
  never writes the row itself (the route does, and the error handlers for the failures), so a
  request that raises still leaves its trace behind. The trace also keeps the retrieval (section
  ids in rank order) and the labelled context chunks, which only the eval harness reads (4.05).
- ``AskMode.NO_RAG`` (3.11) is the Phase 4 baseline: no embedding, no index lookup, no retrieval,
  the sibling prompt ``answer_no_rag_v1`` and no zero-citations check (there is nothing to cite, so
  the check would send every answer to the retry). It is reachable from ``grounded ask --mode
  no_rag`` and from the eval harness, never from the HTTP API: the route does not pass a mode.
- The answer cache (3.13, ``infra/answer_cache.py``) is looked up right after the active index is
  known and before the query is embedded, so a hit costs no embedding, retrieval or LLM call. The
  daily budget (5.05, ``infra/budget.py``) comes **after** that lookup, so a hit never consumes it:
  an advisory check (no reserve) before the embedding, and one atomic reserve before every LLM
  call, a validation retry included. Its token totals are added after each call.
  ``no_rag`` and ``APP_ENV=eval`` never read or write the cache. Only a response that was built
  successfully is stored; every failure raises before that point, so errors are never cached.
- Rerank (6.03, Tech §8) is optional and sits between the search and the context. With a
  ``Reranker`` configured the K_FUSED candidates are reranked to K_CONTEXT inside the ``rerank``
  stage, after one atomic reserve on ``daily_usage.rerank_calls`` (``RerankCap``; none in eval
  mode). Every way it can fail to produce a ranking (the cap, a database error on the reserve,
  a typed provider error, the stage timeout) **degrades**: the context is built from the RRF order
  exactly as without a reranker, ``trace.rerank_error`` names the reason and the request goes on.
  The ``rerank`` stage never aborts a request: only a cancellation (the request deadline) passes
  through it. A degraded answer is not stored in the answer cache, because the key of a reranked
  configuration must only ever hold reranked answers.
- A **fallback** answer (the router's ``fallback_used``) is not stored either (D57c): the key names
  the configured primary's model and generation parameters, so a Groq answer under it would be
  served as Gemini's for up to 29 days. A cache lookup is unaffected, so a cached answer keeps
  being served while a breaker is open.
- In eval mode (and for ``grounded ask``: ``primary_only``) the router has no breakers and one
  provider, and the pipeline re-raises the typed error of that provider instead of the router's
  wrapper (``AllProvidersUnavailable.causes[-1]``), so ``ask_batch``, the promptfoo provider and
  the gate see the same ``ProviderRateLimited`` (``is_quota``, ``retry_after_s``),
  ``BackoffExhaustedError`` or ``ProviderUnavailable`` as before the router existed.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID

import psycopg
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from grounded.generation.citations import MappedAnswer, map_citations
from grounded.generation.confidence import ConfidenceConfig, score_claims
from grounded.generation.context import BuiltContext, build_context, escape_content
from grounded.generation.params import GenerationParams
from grounded.generation.prompts import Prompt
from grounded.generation.providers.base import GenerationResult, LLMProvider, Usage
from grounded.generation.router import (
    SYSTEM_CLOCK,
    AllProvidersUnavailable,
    Attempt,
    AttemptOutcome,
    BreakerConfig,
    ProviderRouter,
    RoutedGeneration,
    RouterClock,
)
from grounded.infra.answer_cache import AnswerCache, CacheKey, from_stored
from grounded.infra.budget import DailyBudget, RerankCap, RerankCapReachedError
from grounded.infra.provider_errors import ProviderError, ProviderRequestRejected
from grounded.infra.timing import Clock, StageTimer
from grounded.ingest.embed import Embedder, Vector
from grounded.observability.cost import Pricing
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.hybrid import hybrid_search
from grounded.retrieval.index import ActiveIndexCache, chunk_titles
from grounded.retrieval.rerank import (
    Reranker,
    RerankErrorCode,
    pacing_allowance_s,
    rerank_error_code,
)
from grounded.retrieval.types import IndexVersion, RetrievedChunk
from grounded.schemas.api import AskResponse, Claim, Meta
from grounded.schemas.llm import LLMAnswer
from grounded.settings import Settings

logger = logging.getLogger(__name__)

# AGENTS.md §6.4: one retry on the same provider with error feedback, then the next provider. The
# router makes it (7.04); the judge (``evals/judge.py``) applies the same bound to its own call.
MAX_VALIDATION_RETRIES = 1

# ``Meta`` has two required strings that describe the retrieval; a mode without retrieval fills them
# with these markers, which no real index label or config hash can equal (those are
# "<ref>@<8 hex>" and 64 hex characters).
NO_RAG_INDEX_VERSION = "none"
NO_RAG_RETRIEVAL_CONFIG_HASH = "no_rag"


class AskMode(StrEnum):
    HYBRID = "hybrid"  # the product: retrieve, then answer from the sources
    NO_RAG = "no_rag"  # the baseline: the model alone, same schema, no sources


@dataclass(frozen=True, slots=True)
class Generation:
    """An answer that passed the checks, plus what it took to get it. ``routed`` is the router's
    report: the answering provider and model, ``fallback_used`` and every attempt."""

    result: GenerationResult[LLMAnswer]  # the successful attempt
    mapped: MappedAnswer
    routed: RoutedGeneration[LLMAnswer]


@dataclass(frozen=True, slots=True)
class _Degraded:
    """Why the RRF order stands: the ``rerank_error`` code, and the exception type behind it for
    the log line (never a message: a provider's text has no business in our logs).

    ``status_code`` is the HTTP status of a ``rejected`` call (a 401 is the key, a 403 or 404 the
    model), a bare number the runbook reads; ``None`` for every other code."""

    code: RerankErrorCode
    cause: str
    status_code: int | None = None


class IndexMismatchError(RuntimeError):
    """The configured embedder is not the one that built the active index: its query vectors would
    be compared with chunk vectors from another space and rank nonsense without any error."""


class RetrievalTimeoutError(Exception):
    """The retrieval stage outlived ``RETRIEVAL_TIMEOUT_S`` (Tech §6, PRD D54a); ``api/errors.py``
    turns it into a 504 ``deadline_exceeded``. It lives here, not in ``api/``: the pipeline never
    imports from the HTTP layer."""

    def __init__(self, timeout_s: float) -> None:
        super().__init__(f"retrieval stage timeout of {timeout_s:g}s exceeded")
        self.timeout_s = timeout_s


class AskPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        pool: AsyncConnectionPool,
        index_cache: ActiveIndexCache,
        embedder: Embedder,
        provider: LLMProvider,
        prompt: Prompt,
        no_rag_prompt: Prompt,
        pricing: Pricing,
        clock: Clock = time.perf_counter,
        budget: DailyBudget | None = None,
        reranker: Reranker | None = None,
        rerank_cap: RerankCap | None = None,
        fallback_providers: Sequence[LLMProvider] = (),
        router_clock: RouterClock = SYSTEM_CLOCK,
        primary_only: bool = False,
    ) -> None:
        """``provider`` is the primary, the first entry of ``GENERATOR_PROVIDERS`` (it names the
        answer-cache key); ``fallback_providers`` are the next entries, in order. The router over
        them is built here, once per pipeline (so once per process). ``router_clock`` is how a
        test moves the breaker's time without sleeping.

        ``primary_only`` (always on in eval mode, where ``Settings`` allows one provider anyway,
        and for ``grounded ask``) makes the router a pass-through for the primary alone: no
        fallback, no breaker, and the provider's own typed error instead of the router's wrapper.
        The eval harness and the golden batch wait out a 429 and stop at a daily quota by reading
        that error (``ask_batch``, ``promptfoo_provider``, the gate), and a breaker would skip
        every question after one 429 and hide the quota from them (Tech §10, "Breakers off").
        """
        self._settings = settings
        self._pool = pool
        self._index_cache = index_cache
        self._embedder = embedder
        self._provider = provider
        self._primary_only = primary_only or settings.app_env == "eval"
        if self._primary_only and fallback_providers:
            raise ValueError("a primary-only pipeline (eval mode, the CLI) takes no fallback")
        self._router = ProviderRouter(
            [provider, *fallback_providers],
            config=BreakerConfig.from_settings(settings),
            clock=router_clock,
            breakers_enabled=not self._primary_only,
        )
        self._prompt = prompt
        self._no_rag_prompt = no_rag_prompt
        self._cfg = RetrievalConfig.from_settings(settings, "hybrid")
        self._confidence_cfg = ConfidenceConfig.from_settings(settings)
        # One reading of the generation settings for the call and the cache key alike (D47).
        self._generation = GenerationParams.from_settings(settings, provider.name)
        self._pricing = pricing
        self._clock = clock
        # None when no cap is set (dev, test) and in eval mode (Tech §12, §15.6).
        self._budget = budget
        # The config hash says whether retrieval is reranked, and it keys the answer cache, so the
        # pipeline must do exactly what the hash says: no reranker with RERANK_PROVIDER=cohere (or
        # one with ``none``) would file plain answers under the reranked key, or the reverse.
        if (reranker is not None) != (self._cfg.rerank_provider != "none"):
            raise ValueError(
                "a reranker must be passed exactly when RERANK_PROVIDER is not 'none' "
                f"(RERANK_PROVIDER={self._cfg.rerank_provider}, reranker "
                f"{'passed' if reranker is not None else 'missing'})"
            )
        self._reranker = reranker
        # None when no cap applies (eval mode: the Author's manual runs have their own call budget,
        # Tech §15.6). With a reranker and no cap the stage still runs; it is the caller's choice.
        self._rerank_cap = rerank_cap
        # Off in eval mode, like the request deadline: eval runs measure, they do not time out.
        self._retrieval_timeout_s = (
            None if settings.app_env == "eval" else settings.retrieval_timeout_s
        )
        # Off in eval mode (Tech §11): eval runs must measure the pipeline, not the cache.
        self._cache = (
            None
            if settings.app_env == "eval"
            else AnswerCache(pool, ttl_days=settings.answer_cache_ttl_days)
        )
        # An unpriced model stops the service here, at startup, instead of turning into a silent
        # cost of 0 (Tech §14). The embedder is built from ``settings.embedding_model``, so that
        # is the model whose price the shadow cost uses.
        for generator in (provider, *fallback_providers):
            pricing.require_generator(generator.name, generator.model)
        pricing.require_embedding(settings.embedding_model)
        if reranker is not None:
            rerank_model = (self._cfg.rerank_model or "").strip()
            if not rerank_model:
                raise ValueError("RERANK_MODEL must be set when a reranker is configured")
            pricing.require_rerank(rerank_model)

    async def ask(
        self,
        question: str,
        request_id: UUID,
        *,
        mode: AskMode = AskMode.HYBRID,
        trace: RequestTrace | None = None,
    ) -> AskResponse:
        """``trace`` is the request's scratchpad (3.12). The route passes the one it created before
        parsing the body, so the total covers the whole request and a failure leaves it filled; the
        CLI and the evals pass none and get a private one."""
        if trace is None:
            trace = RequestTrace(request_id=request_id, timer=StageTimer(self._clock))
        timer = trace.timer
        trace.provider = self._provider.name
        trace.model = self._provider.model
        cache_key: CacheKey | None = None  # set only when this request may use the answer cache

        if mode is AskMode.NO_RAG:
            # No embedding and no database access at all: the index is not even looked up, so this
            # mode also works before the first ingest.
            prompt = self._no_rag_prompt
            index_version = NO_RAG_INDEX_VERSION
            config_hash = NO_RAG_RETRIEVAL_CONFIG_HASH
            chunks: list[RetrievedChunk] = []
            labels: Mapping[str, RetrievedChunk] = {}
            titles: Mapping[int, str] = {}
            user = prompt.render_user(question=question)
            trace.prompt_version = prompt.version
            trace.retrieval_config_hash = config_hash  # index_version_id stays None: no index
        else:
            prompt = self._prompt
            trace.prompt_version = prompt.version
            config_hash = self._cfg.config_hash
            trace.retrieval_config_hash = config_hash
            with timer.stage("retrieval"):
                index = await self._index_cache.get()
            trace.index_version_id = index.id
            self._check_embedder(index)
            index_version = index.label

            if self._cache is not None:
                cache_key = CacheKey.build(
                    question,
                    prompt_version=prompt.version,
                    index_version_id=index.id,
                    retrieval_config_hash=config_hash,
                    generator_model=self._provider.model,
                    confidence=self._confidence_cfg,
                    generation=self._generation,
                )
                cached = await self._cached_response(cache_key, request_id, trace, index_version)
                if cached is not None:
                    return cached

            if self._budget is not None:
                await self._budget.check_available()  # advisory: spend no embedding on a lost day

            # Recorded before the call: a failed call may still have been billed.
            trace.embedding_model = self._settings.embedding_model
            trace.embedding_tokens = len(question)
            with timer.stage("embed"):
                [vector] = await self._embedder.embed([question], "RETRIEVAL_QUERY")

            chunks, context, titles = await self._retrieve(
                question, vector, index.id, request_id, trace
            )
            labels = context.labels
            trace.retrieved_section_ids = [chunk.section_id for chunk in chunks]
            trace.context_chunks = dict(context.labels)
            # The connection went back to the pool before the (slow) LLM call. The question is
            # escaped like chunk text, so it cannot open a fake <source> block of its own.
            user = prompt.render_user(question=escape_content(question), sources=context.text)

        with timer.stage("llm"):
            generation = await self._generate_routed(
                prompt,
                user,
                labels,
                titles,
                request_id,
                trace,
                require_citations=mode is AskMode.HYBRID,
            )
        result, mapped, routed = generation.result, generation.mapped, generation.routed

        # ``mapped.claims`` are the model's claims one to one (citations.py), except that an
        # ``insufficient_context`` answer has none: its claims are dropped, so nothing is scored.
        model_claims = (
            [] if result.parsed.status == "insufficient_context" else result.parsed.claims
        )
        scores = score_claims(model_claims, labels, chunks, config=self._confidence_cfg)
        claims = [
            Claim(
                text=claim.text,
                citations=list(claim.citations),
                confidence=score.confidence,
                confidence_components=score.components,
            )
            for claim, score in zip(mapped.claims, scores, strict=True)
        ]
        min_confidence = min((c.confidence for c in claims), default=None)

        trace.status = result.parsed.status
        trace.citation_count = len(mapped.citations)
        trace.min_claim_confidence = min_confidence
        trace.latency_total_ms = timer.total_ms()  # frozen: the response and the row agree
        shadow_cost = trace.shadow_cost(self._pricing)

        response = AskResponse(
            status=result.parsed.status,
            answer_markdown=mapped.answer_markdown,
            claims=claims,
            citations=list(mapped.citations),
            follow_up_questions=result.parsed.follow_up_questions,
            min_confidence=min_confidence,
            meta=Meta(
                request_id=request_id,
                provider=routed.provider,
                model=routed.model,
                fallback_used=routed.fallback_used,
                cache_hit=False,
                rerank_used=trace.rerank_used,
                prompt_version=prompt.version,
                index_version=index_version,
                retrieval_config_hash=config_hash,
                latency_ms=trace.latency_ms(),  # embed and retrieval are 0 in no_rag: skipped
                tokens={"input": trace.input_tokens or 0, "output": trace.output_tokens or 0},
                shadow_cost_usd=float(shadow_cost),
            ),
        )
        # An answer built from the RRF order because the rerank degraded is not what this key (it
        # holds the rerank config) stands for, and a hit would serve it for the answer cache's TTL
        # (``ANSWER_CACHE_TTL_DAYS``) after the reranker is back. It is simply not stored. The same
        # goes for a fallback answer (D57c): the key names the primary's model and generation
        # parameters, so Groq's answer would be served as Gemini's for up to 29 days.
        if (
            self._cache is not None
            and cache_key is not None
            and trace.rerank_error is None
            and not routed.fallback_used
        ):
            await self._cache.put(cache_key, response)
        return response

    async def _retrieve(
        self,
        question: str,
        vector: Vector,
        index_version_id: int,
        request_id: UUID,
        trace: RequestTrace,
    ) -> tuple[list[RetrievedChunk], BuiltContext, dict[int, str]]:
        """The candidates (RRF order), the context built from them and the titles of its chunks.

        Without a reranker this is the single scope of before 6.03: one pool connection and one
        ``RETRIEVAL_TIMEOUT_S`` cover the search, the context build and the title lookup, so a
        deployment with ``RERANK_PROVIDER=none`` behaves exactly as it did.

        With one, the connection goes back to the pool before the rerank call, because a pooled
        connection must not sit idle for up to ``RERANK_TIMEOUT_S`` of someone else's latency.
        The titles are needed only for the chunks that reach the context, and those are known only
        after the rerank, so a second short scope (same timeout) reads them.
        """
        cfg = self._cfg
        timer = trace.timer

        async def search(conn: AsyncConnection[Any]) -> list[RetrievedChunk]:
            return await hybrid_search(
                conn, question, vector, index_version_id=index_version_id, cfg=cfg
            )

        if self._reranker is None:

            async def search_and_build(
                conn: AsyncConnection[Any],
            ) -> tuple[list[RetrievedChunk], BuiltContext, dict[int, str]]:
                found = await search(conn)
                built = build_context(found, k_context=cfg.k_context)
                titles = await chunk_titles(conn, [c.chunk_id for c in built.labels.values()])
                return found, built, titles

            with timer.stage("retrieval"):
                return await self._in_retrieval_scope(search_and_build)

        with timer.stage("retrieval"):
            candidates = await self._in_retrieval_scope(search)
        reranked = await self._rerank(self._reranker, question, candidates, request_id, trace)
        with timer.stage("retrieval"):
            context = build_context(
                candidates if reranked is None else reranked, k_context=cfg.k_context
            )
            chunk_ids = [c.chunk_id for c in context.labels.values()]
            titles = await self._in_retrieval_scope(lambda conn: chunk_titles(conn, chunk_ids))
        return candidates, context, titles

    async def _in_retrieval_scope[T](
        self, work: Callable[[AsyncConnection[Any]], Awaitable[T]]
    ) -> T:
        """Run ``work`` on a pooled connection inside the retrieval stage timeout.

        Only this scope's own expiry is the retrieval timeout; a ``TimeoutError`` from the work
        inside, or the request deadline around it, passes through unchanged (the same rule as
        api/deadline.py). ``None`` (eval mode) never expires.
        """
        limit = self._retrieval_timeout_s
        scope = asyncio.timeout(limit)
        try:
            async with scope, self._pool.connection() as conn:
                return await work(conn)
        except TimeoutError as exc:
            if limit is not None and scope.expired():
                raise RetrievalTimeoutError(limit) from exc
            raise

    async def _rerank(
        self,
        reranker: Reranker,
        question: str,
        candidates: Sequence[RetrievedChunk],
        request_id: UUID,
        trace: RequestTrace,
    ) -> list[RetrievedChunk] | None:
        """The candidates reranked to ``K_CONTEXT``, or ``None`` when the RRF order stands.

        Never raises for a reason of the rerank itself (Tech §8): the cap, a failed reserve, a
        typed provider error and the stage timeout each become ``trace.rerank_error`` and a
        WARNING with the code, the exception type and, for ``rejected``, the HTTP status (no
        question, no provider text). Only ``CancelledError`` (the request deadline, a client that
        left) and a bug propagate; so does the ``ValueError`` of a bad ``top_n``, which
        ``K_CONTEXT`` never is.

        The ``rerank`` stage times the call alone: a request that reached the cap or failed the
        reserve has ``latency_rerank_ms`` NULL, so the column describes what the reranker
        costs in time. The unit is reserved before the call and counted on the trace right
        before it (``rerank_calls``): a call that then fails or is cut off was attempted, and
        the shadow cost and the daily cap both charge it.
        """
        if not candidates:
            return None  # nothing to rank: no unit, no call, and not a degradation
        degraded = await self._reserve_rerank_call()
        if degraded is None:
            trace.rerank_model = self._cfg.rerank_model
            trace.rerank_calls += 1
            try:
                with trace.timer.stage("rerank"):
                    # The adapter bounds its own call; this scope makes the bound ours, whatever
                    # ``Reranker`` is injected. A TimeoutError here cannot be the request
                    # deadline: that one arrives as a cancellation and is not caught below.
                    # The eval pacer's wait for its turn (6.04) is the one thing the scope allows
                    # for on top: the call itself is still bounded by the adapter's own timeout.
                    # It is 0 for every reranker on the request path, which never waits.
                    limit = self._settings.rerank_timeout_s + pacing_allowance_s(reranker)
                    async with asyncio.timeout(limit):
                        ranked = await reranker.rerank(
                            question, candidates, top_n=self._cfg.k_context
                        )
            except ProviderError as exc:
                status = exc.status_code if isinstance(exc, ProviderRequestRejected) else None
                degraded = _Degraded(rerank_error_code(exc), type(exc).__name__, status)
            except TimeoutError as exc:
                degraded = _Degraded("timeout", type(exc).__name__)
            else:
                trace.rerank_used = True
                return ranked
        trace.rerank_error = degraded.code
        logger.warning(
            "rerank degraded, using the RRF order",
            extra={
                "request_id": str(request_id),
                "rerank_error": degraded.code,
                "error": degraded.cause,
                "status_code": degraded.status_code,
            },
        )
        return None

    async def _reserve_rerank_call(self) -> _Degraded | None:
        """Take one unit of today's cap, or say why no call may be made (``None``: go ahead).

        Fail-closed: when the unit cannot be counted (the database errors), no call is made. The
        quota behind the cap is the scarce thing, a rerank is an optimization, and ``unavailable``
        is the nearest code of Tech §8; the exception type in the log tells it from a Cohere outage.
        """
        if self._rerank_cap is None:
            return None
        try:
            await self._rerank_cap.reserve()
        except RerankCapReachedError as exc:
            return _Degraded("cap_reached", type(exc).__name__)
        except psycopg.Error as exc:
            return _Degraded("unavailable", type(exc).__name__)
        return None

    async def _cached_response(
        self, key: CacheKey, request_id: UUID, trace: RequestTrace, index_version: str
    ) -> AskResponse | None:
        """The cached answer with a ``meta`` of its own, or ``None`` on a miss.

        The hit did none of the work, so it reports zero tokens and zero cost, and the real latency
        of this request (index lookup and cache lookup). ``provider`` and ``model`` are the current
        generator's, which is what the key's ``generator_model`` pins.
        """
        if self._cache is None:  # eval mode: the caller builds no key, but stay safe without it
            return None
        stored = await self._cache.get(key)
        if stored is None:
            return None
        trace.latency_total_ms = trace.timer.total_ms()  # frozen: the response and the row agree
        meta = Meta(
            request_id=request_id,
            provider=self._provider.name,
            model=self._provider.model,
            fallback_used=False,
            cache_hit=True,
            rerank_used=False,
            prompt_version=key.prompt_version,
            index_version=index_version,
            retrieval_config_hash=key.retrieval_config_hash,
            latency_ms=trace.latency_ms(),
            tokens={"input": 0, "output": 0},
            shadow_cost_usd=0.0,
        )
        try:
            response = from_stored(stored, meta)
        except ValidationError:
            # A row that no longer fits the schema (written before a schema change): serve a fresh
            # answer instead of an error, and delete the row so that answer can replace it (``put``
            # only overwrites expired rows). Logged without the row, which holds the question.
            logger.warning("cached answer no longer matches AskResponse, treating as a miss")
            await self._cache.discard(key)
            trace.latency_total_ms = None
            return None
        trace.cache_hit = True
        trace.status = response.status
        trace.citation_count = len(response.citations)
        trace.min_claim_confidence = response.min_confidence
        return response

    async def _generate_routed(
        self,
        prompt: Prompt,
        user: str,
        labels: Mapping[str, RetrievedChunk],
        titles: Mapping[int, str],
        request_id: UUID,
        trace: RequestTrace,
        *,
        require_citations: bool,
    ) -> Generation:
        """Generate through the router and check the answer (Tech §9.5, §10).

        Bad output is either a schema failure raised by the adapter or a semantic failure from the
        citation checks (``validate``). The router retries it once on the same provider with the
        prompt's "Retry feedback" section (``retry_user``), then tries the next provider. A rate
        limit, 5xx or timeout goes to the next provider at once and opens that provider's breaker
        (nothing here sleeps). What comes out when nobody answers is the router's: bad output
        again is a ``ProviderBadOutput`` (502), a refused attempt (too little time before the
        request deadline) a ``RouterDeadlineError`` (504), and any transient failure an
        ``AllProvidersUnavailable`` (503, with the wait the breakers say), except in eval mode,
        where it is the provider's own typed error (the handler below).

        The hooks are the pipeline's side of the contract. ``before_call`` reserves one budget unit
        before **every** call, so a retry and a fallback each reserve and a skipped provider
        reserves nothing; the router calls it only for a call it is going to make. ``validate`` is
        pure and cannot raise: ``map_citations`` over this request's labels, remembering the last
        mapping (the router validates the answering result last). ``on_attempt`` records each
        call as it ends: the trace follows the provider that answered (or the last that reported
        back), counts the retries, and adds the usage of failed calls that were billed too, so a
        request that fails later still logs what it spent.

        ``require_citations=False`` (``no_rag``) turns the zero-valid-citations check off, so only a
        schema failure can cause the retry.
        """
        mapping: list[MappedAnswer] = []  # the last ``validate`` call's result, the answering one

        async def before_call() -> None:
            if self._budget is not None:
                await self._budget.reserve()  # a retry or a fallback reserves again

        def validate(result: GenerationResult[LLMAnswer]) -> str | None:
            mapped = map_citations(
                result.parsed, labels, titles, require_citations=require_citations
            )
            trace.invalid_citation_count = mapped.invalid_citation_count
            trace.dropped_claim_count = mapped.dropped_claim_count
            trace.removed_url_count = mapped.removed_url_count
            mapping[:] = [mapped]
            return mapped.bad_output

        async def on_attempt(attempt: Attempt) -> None:
            await self._record_attempt(trace, attempt)

        def time_left() -> float:
            left = trace.time_left_s()
            return math.inf if left is None else left

        try:
            routed = await self._router.generate(
                system=prompt.system,
                user=user,
                schema=LLMAnswer,
                temperature=self._generation.temperature,
                max_output_tokens=self._generation.max_output_tokens,
                timeout_s=self._settings.call_timeout_s,
                retry_user=lambda original, reason: prompt.render_retry(original, error=reason),
                before_call=before_call if self._budget is not None else None,
                validate=validate,
                on_attempt=on_attempt,
                time_left=time_left if trace.deadline_s is not None else None,
                log_extra={"request_id": str(request_id)},
            )
        except AllProvidersUnavailable as exc:
            if self._primary_only and exc.causes:
                # The router wraps what the provider raised; the eval harness and the golden
                # batch read the provider's own error (``is_quota``, ``retry_after_s``,
                # ``BackoffExhaustedError``).
                raise exc.causes[-1] from exc
            raise
        return Generation(routed.result, mapping[-1], routed)

    async def _record_attempt(self, trace: RequestTrace, attempt: Attempt) -> None:
        """What one finished call adds to the trace (the router's ``on_attempt``, never a skip).

        ``trace.provider`` and ``model`` follow the calls, so a request that dies after trying a
        fallback names the one that was tried last. ``fallback_used`` means a provider other than
        the primary was called, whether or not it answered. ``validation_retries`` counts the
        retry attempts across providers. The usage of a call that billed tokens is added even when
        the call failed (a bad output was billed), at the price of the model that billed it.
        """
        trace.provider, trace.model = attempt.provider, attempt.model
        if attempt.provider != self._provider.name:
            trace.fallback_used = True
        trace.validation_retries += int(attempt.is_retry)
        trace.llm_cache_hits += int(attempt.cache_hit)
        usage = attempt.usage
        if attempt.outcome is AttemptOutcome.SUCCESS or usage.input_tokens or usage.output_tokens:
            await self._record_usage(trace, usage, attempt.provider, attempt.model)

    async def _record_usage(
        self, trace: RequestTrace, usage: Usage, provider: str, model: str
    ) -> None:
        trace.add_usage(usage, provider=provider, model=model)
        if self._budget is not None:
            await self._budget.add_tokens(usage)

    def _check_embedder(self, index: IndexVersion) -> None:
        if (self._embedder.model, self._embedder.dim) != (
            index.embedding_model,
            index.embedding_dim,
        ):
            raise IndexMismatchError(
                f"embedder {self._embedder.model}/{self._embedder.dim} does not match active "
                f"index {index.label} ({index.embedding_model}/{index.embedding_dim})"
            )
