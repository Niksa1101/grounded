"""The `/ask` flow: question, embedding, hybrid retrieval, context, generation, response.

``AskPipeline.ask`` is the one place the stages are wired together; the route stays thin and every
stage lives in its own module. Failures are raised as the typed errors of the stage that failed
(``ProviderError``, ``NoActiveIndexError``) and translated to HTTP in ``api/errors.py``.

Tracer-bullet state (Phase 3): the stages thicken in later tickets, and what is deliberately thin
today is marked below.

- Citation mapping lives in ``citations.py`` (3.06). A semantic failure (an answer with no valid
  citation) is treated like a schema failure: ``ProviderBadOutput``, which gets the one retry.
  ``invalid_citation_count`` is computed per request and recorded on the trace.
- ``_generate_validated`` is the one retry with feedback (3.07, Tech §9.5 step 3). It counts
  ``validation_retries`` on the trace, and 7.04 moves it into the router.
- Claim confidence comes from ``confidence.score_claims`` (3.10): the server computes it from the
  retrieval signals of the cited chunks, and the model's ``self_confidence`` is only one weak input,
  never shown as confidence (AGENTS.md §6.6).
- ``RequestTrace`` (3.12) is filled as the stages run: versions, usage, retries and citation counts
  for the ``request_logs`` row, and the stage timings and shadow cost for ``meta``. The pipeline
  never writes the row itself (the route does, and the error handlers for the failures), so a
  request that raises still leaves its trace behind.
- ``AskMode.NO_RAG`` (3.11) is the Phase 4 baseline: no embedding, no index lookup, no retrieval,
  the sibling prompt ``answer_no_rag_v1`` and no zero-citations check (there is nothing to cite, so
  the check would send every answer to the retry). It is reachable from ``grounded ask --mode
  no_rag`` and from the eval harness, never from the HTTP API: the route does not pass a mode.
- The answer cache (3.13, ``infra/answer_cache.py``) is looked up right after the active index is
  known and before the query is embedded, so a hit costs no embedding, retrieval or LLM call. The
  daily budget reservation (5.05) must go **after** that lookup: a hit does not consume budget.
  ``no_rag`` and ``APP_ENV=eval`` never read or write the cache. Only a response that was built
  successfully is stored; every failure raises before that point, so errors are never cached.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from psycopg_pool import AsyncConnectionPool
from pydantic import ValidationError

from grounded.generation.citations import MappedAnswer, map_citations
from grounded.generation.confidence import ConfidenceConfig, score_claims
from grounded.generation.context import build_context, escape_content
from grounded.generation.params import GenerationParams
from grounded.generation.prompts import Prompt
from grounded.generation.providers.base import GenerationResult, LLMProvider, Usage
from grounded.infra.answer_cache import AnswerCache, CacheKey, from_stored
from grounded.infra.provider_errors import ProviderBadOutput
from grounded.infra.timing import Clock, StageTimer
from grounded.ingest.embed import Embedder
from grounded.observability.cost import Pricing
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.hybrid import hybrid_search
from grounded.retrieval.index import ActiveIndexCache, chunk_titles
from grounded.retrieval.types import IndexVersion, RetrievedChunk
from grounded.schemas.api import AskResponse, Claim, Meta
from grounded.schemas.llm import LLMAnswer
from grounded.settings import Settings

logger = logging.getLogger(__name__)

# AGENTS.md §6.4: one retry on the same provider, then give up (fallback arrives with the router).
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
    """An answer that passed the checks, plus what it took to get it."""

    result: GenerationResult[LLMAnswer]  # the successful attempt
    mapped: MappedAnswer


class IndexMismatchError(RuntimeError):
    """The configured embedder is not the one that built the active index: its query vectors would
    be compared with chunk vectors from another space and rank nonsense without any error."""


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
    ) -> None:
        self._settings = settings
        self._pool = pool
        self._index_cache = index_cache
        self._embedder = embedder
        self._provider = provider
        self._prompt = prompt
        self._no_rag_prompt = no_rag_prompt
        self._cfg = RetrievalConfig.from_settings(settings, "hybrid")
        self._confidence_cfg = ConfidenceConfig.from_settings(settings)
        # One reading of the generation settings for the call and the cache key alike (D47).
        self._generation = GenerationParams.from_settings(settings, provider.name)
        self._pricing = pricing
        self._clock = clock
        # Off in eval mode (Tech §11): eval runs must measure the pipeline, not the cache.
        self._cache = (
            None
            if settings.app_env == "eval"
            else AnswerCache(pool, ttl_days=settings.answer_cache_ttl_days)
        )
        # An unpriced model stops the service here, at startup, instead of turning into a silent
        # cost of 0 (Tech §14). The embedder is built from ``settings.embedding_model``, so that
        # is the model whose price the shadow cost uses.
        pricing.require_generator(provider.name, provider.model)
        pricing.require_embedding(settings.embedding_model)

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

            # Recorded before the call: a failed call may still have been billed.
            trace.embedding_model = self._settings.embedding_model
            trace.embedding_tokens = len(question)
            with timer.stage("embed"):
                [vector] = await self._embedder.embed([question], "RETRIEVAL_QUERY")

            with timer.stage("retrieval"):
                async with self._pool.connection() as conn:
                    chunks = await hybrid_search(
                        conn, question, vector, index_version_id=index.id, cfg=self._cfg
                    )
                    context = build_context(chunks, k_context=self._cfg.k_context)
                    titles = await chunk_titles(conn, [c.chunk_id for c in context.labels.values()])
            labels = context.labels
            # The connection went back to the pool before the (slow) LLM call. The question is
            # escaped like chunk text, so it cannot open a fake <source> block of its own.
            user = prompt.render_user(question=escape_content(question), sources=context.text)

        with timer.stage("llm"):
            generation = await self._generate_validated(
                prompt,
                user,
                labels,
                titles,
                request_id,
                trace,
                require_citations=mode is AskMode.HYBRID,
            )
        result, mapped = generation.result, generation.mapped

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
                provider=result.provider,
                model=result.model,
                fallback_used=False,
                cache_hit=False,
                rerank_used=False,
                prompt_version=prompt.version,
                index_version=index_version,
                retrieval_config_hash=config_hash,
                latency_ms=trace.latency_ms(),  # embed and retrieval are 0 in no_rag: skipped
                tokens={"input": trace.input_tokens or 0, "output": trace.output_tokens or 0},
                shadow_cost_usd=float(shadow_cost),
            ),
        )
        if self._cache is not None and cache_key is not None:
            await self._cache.put(cache_key, response)
        return response

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

    async def _generate_validated(
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
        """Generate, check the answer, and on bad output retry exactly once on the same provider.

        Bad output is either a schema failure raised by the adapter or a semantic failure from the
        citation checks (Tech §9.5). The retry's user message is the original one plus the
        prompt's "Retry feedback" section with the reason filled in. A second failure propagates
        as ``ProviderBadOutput`` (HTTP 502), and so does a first one the adapter marks not
        ``retryable`` (a content-filter block, which the same request would hit again). Rate
        limits, 5xx and timeouts are not bad output and propagate straight away, on the retry as
        well. There is no loop, no sleep, and no fallback yet: 7.04 moves this whole step into the
        router, which then falls back instead of failing.

        ``require_citations=False`` (``no_rag``) turns the zero-valid-citations check off, so only a
        schema failure can cause the retry.

        The trace gets the usage of every attempt, a schema-invalid one included (the adapter puts
        the billed tokens on ``ProviderBadOutput``), the retry count and the citation counts of the
        last attempt, so a request that fails here still logs what it spent. The citation-check
        error raised below carries no tokens: that attempt's usage was added right after the call.
        """
        attempt_user = user
        while True:
            try:
                result = await self._generate(prompt, attempt_user)
                trace.add_usage(result.usage)
                trace.llm_cache_hits += int(result.cache_hit)
                mapped = map_citations(
                    result.parsed, labels, titles, require_citations=require_citations
                )
                trace.invalid_citation_count = mapped.invalid_citation_count
                trace.dropped_claim_count = mapped.dropped_claim_count
                trace.removed_url_count = mapped.removed_url_count
                if mapped.bad_output is not None:
                    raise ProviderBadOutput(
                        "the answer failed the citation checks",
                        raw=result.raw_text,
                        validation_error=mapped.bad_output,
                    )
                return Generation(result, mapped)
            except ProviderBadOutput as exc:
                if exc.input_tokens or exc.output_tokens:
                    trace.add_usage(
                        Usage(input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
                    )
                if not exc.retryable or trace.validation_retries >= MAX_VALIDATION_RETRIES:
                    raise
                trace.validation_retries += 1
                logger.info(
                    "model output invalid, retrying once",
                    extra={
                        "request_id": str(request_id),
                        "validation_retries": trace.validation_retries,
                    },
                )
                attempt_user = prompt.render_retry(user, error=exc.validation_error or str(exc))

    async def _generate(self, prompt: Prompt, user: str) -> GenerationResult[LLMAnswer]:
        return await self._provider.generate(
            system=prompt.system,
            user=user,
            schema=LLMAnswer,
            temperature=self._generation.temperature,
            max_output_tokens=self._generation.max_output_tokens,
            timeout_s=self._settings.llm_timeout_s,
        )

    def _check_embedder(self, index: IndexVersion) -> None:
        if (self._embedder.model, self._embedder.dim) != (
            index.embedding_model,
            index.embedding_dim,
        ):
            raise IndexMismatchError(
                f"embedder {self._embedder.model}/{self._embedder.dim} does not match active "
                f"index {index.label} ({index.embedding_model}/{index.embedding_dim})"
            )
