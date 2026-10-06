"""The `/ask` flow: question, embedding, hybrid retrieval, context, generation, response.

``AskPipeline.ask`` is the one place the stages are wired together; the route stays thin and every
stage lives in its own module. Failures are raised as the typed errors of the stage that failed
(``ProviderError``, ``NoActiveIndexError``) and translated to HTTP in ``api/errors.py``.

Tracer-bullet state (Phase 3): the stages thicken in later tickets, and what is deliberately thin
today is marked below.

- Citation mapping lives in ``citations.py`` (3.06). A semantic failure (an answer with no valid
  citation) is treated like a schema failure: ``ProviderBadOutput``, which gets the one retry.
  ``invalid_citation_count`` is computed per request, and 3.12 logs it.
- ``_generate_validated`` is the one retry with feedback (3.07, Tech §9.5 step 3). It counts
  ``validation_retries`` (3.12 logs it), and 7.04 moves it into the router.
- Claim confidence comes from ``confidence.score_claims`` (3.10): the server computes it from the
  retrieval signals of the cited chunks, and the model's ``self_confidence`` is only one weak input,
  never shown as confidence (AGENTS.md §6.6). ``shadow_cost_usd`` and the timings are partial
  until 3.12.
- ``AskMode.NO_RAG`` (3.11) is the Phase 4 baseline: no embedding, no index lookup, no retrieval,
  the sibling prompt ``answer_no_rag_v1`` and no zero-citations check (there is nothing to cite, so
  the check would send every answer to the retry). It is reachable from ``grounded ask --mode
  no_rag`` and from the eval harness, never from the HTTP API: the route does not pass a mode.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from grounded.generation.citations import MappedAnswer, map_citations
from grounded.generation.confidence import ConfidenceConfig, score_claims
from grounded.generation.context import build_context
from grounded.generation.prompts import Prompt
from grounded.generation.providers.base import GenerationResult, LLMProvider, Usage
from grounded.infra.provider_errors import ProviderBadOutput
from grounded.ingest.embed import Embedder
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
    usage: Usage  # summed over the attempts whose usage we saw
    validation_retries: int  # 0 or 1; 3.12 writes it to the request log


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

    async def ask(
        self, question: str, request_id: UUID, *, mode: AskMode = AskMode.HYBRID
    ) -> AskResponse:
        started = time.perf_counter()
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
            embedded = retrieved = time.perf_counter()
        else:
            prompt = self._prompt
            index = await self._index_cache.get()
            self._check_embedder(index)
            index_version = index.label
            config_hash = self._cfg.config_hash

            [vector] = await self._embedder.embed([question], "RETRIEVAL_QUERY")
            embedded = time.perf_counter()

            async with self._pool.connection() as conn:
                chunks = await hybrid_search(
                    conn, question, vector, index_version_id=index.id, cfg=self._cfg
                )
                context = build_context(chunks, k_context=self._cfg.k_context)
                titles = await chunk_titles(conn, [c.chunk_id for c in context.labels.values()])
            labels = context.labels
            # The connection is back in the pool before the (slow) LLM call.
            retrieved = time.perf_counter()
            user = prompt.render_user(question=question, sources=context.text)

        generation = await self._generate_validated(
            prompt, user, labels, titles, request_id, require_citations=mode is AskMode.HYBRID
        )
        result, mapped = generation.result, generation.mapped
        generated = time.perf_counter()

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
        return AskResponse(
            status=result.parsed.status,
            answer_markdown=mapped.answer_markdown,
            claims=claims,
            citations=list(mapped.citations),
            follow_up_questions=result.parsed.follow_up_questions,
            min_confidence=min((c.confidence for c in claims), default=None),
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
                latency_ms={
                    "total": _ms(started, time.perf_counter()),
                    "embed": _ms(started, embedded),  # embed and retrieval are 0 in no_rag: skipped
                    "retrieval": _ms(embedded, retrieved),
                    "llm": _ms(retrieved, generated),
                },
                tokens={
                    "input": generation.usage.input_tokens,
                    "output": generation.usage.output_tokens,
                },
                shadow_cost_usd=0.0,
            ),
        )

    async def _generate_validated(
        self,
        prompt: Prompt,
        user: str,
        labels: Mapping[str, RetrievedChunk],
        titles: Mapping[int, str],
        request_id: UUID,
        *,
        require_citations: bool,
    ) -> Generation:
        """Generate, check the answer, and on bad output retry exactly once on the same provider.

        Bad output is either a schema failure raised by the adapter or a semantic failure from the
        citation checks (Tech §9.5). The retry's user message is the original one plus the
        prompt's "Retry feedback" section with the reason filled in. A second failure propagates
        as ``ProviderBadOutput`` (HTTP 502). Rate limits, 5xx and timeouts are not bad output and
        propagate straight away, on the retry as well. There is no loop, no sleep, and no fallback
        yet: 7.04 moves this whole step into the router, which then falls back instead of failing.

        ``require_citations=False`` (``no_rag``) turns the zero-valid-citations check off, so only a
        schema failure can cause the retry.

        ``usage`` sums the attempts whose token counts we saw. An adapter that raises
        ``ProviderBadOutput`` reports no usage, so a schema-invalid first attempt is not counted.
        """
        spent = Usage(input_tokens=0, output_tokens=0)
        retries = 0
        attempt_user = user
        while True:
            try:
                result = await self._generate(prompt, attempt_user)
                spent = _add(spent, result.usage)
                mapped = map_citations(
                    result.parsed, labels, titles, require_citations=require_citations
                )
                if mapped.bad_output is not None:
                    raise ProviderBadOutput(
                        "the answer failed the citation checks",
                        raw=result.raw_text,
                        validation_error=mapped.bad_output,
                    )
                return Generation(result, mapped, spent, retries)
            except ProviderBadOutput as exc:
                if retries >= MAX_VALIDATION_RETRIES:
                    raise
                retries += 1
                logger.info(
                    "model output invalid, retrying once",
                    extra={"request_id": str(request_id), "validation_retries": retries},
                )
                attempt_user = prompt.render_retry(user, error=exc.validation_error or str(exc))

    async def _generate(self, prompt: Prompt, user: str) -> GenerationResult[LLMAnswer]:
        return await self._provider.generate(
            system=prompt.system,
            user=user,
            schema=LLMAnswer,
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_output_tokens,
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


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        input_tokens=a.input_tokens + b.input_tokens,
        output_tokens=a.output_tokens + b.output_tokens,
        thinking_tokens=a.thinking_tokens + b.thinking_tokens,
    )


def _ms(start: float, end: float) -> int:
    return round((end - start) * 1000)
