"""One ``request_logs`` row per ``POST /v1/ask``, whatever its outcome (Tech.md §14, DB.md §4).

``RequestTrace`` is the per-request scratchpad. The route creates it before the body is parsed, the
pipeline fills it as the stages run (so a request that fails halfway still records what it had
spent), and exactly one of two places writes the row: the route wrapper on success, or the error
handler that turns the exception into a response (``api/errors.py``). ``RequestTrace.written`` makes
a second attempt a no-op.

**A failed write never breaks the response, and is never silent.** It is logged at ERROR with the
``request_id``. It logs the exception *type* and the SQLSTATE only, never the message or the
traceback: Postgres puts the failing row, i.e. the question, into the DETAIL of constraint errors,
and question text must not reach stdout (AGENTS.md §6.13).

What the row holds that the response does not: ``question`` (kept for the retention job of 5.11),
``question_hash`` (``infra/hashing.py``, the same normalization as the answer cache of 3.13),
``error_code`` and the counts of invalid citations. ``ip_hash`` stays NULL until 5.03 and ``source``
is ``'api'`` until 5.03 sets ``'web'`` behind the proxy. The row's ``id`` is the ``request_id``, so
a log line and a row correlate directly. A ``bad_request`` row has no ``question`` (the text was
rejected, possibly because it is huge or not text at all); only its hash, when one can be computed.
``dropped_claim_count`` and ``removed_url_count`` have no column; they go to the stdout summary line
instead.

``latency_total_ms`` is measured up to the moment the row is written. The insert itself cannot be
inside the number stored by that very insert, so "includes logging" (Tech §14) means everything
before it: validation, retrieval, generation and response assembly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg_pool import AsyncConnectionPool

from grounded.generation.providers.base import Usage
from grounded.infra.hashing import question_hash
from grounded.infra.timing import Clock, StageTimer
from grounded.observability.cost import Pricing
from grounded.schemas.api import ErrorCode
from grounded.schemas.llm import AnswerStatus

logger = logging.getLogger(__name__)

# The outcomes written today. DB.md §4 also allows rate_limited and budget_exhausted (5.04, 5.05).
Outcome = Literal[
    "answered",
    "partial",
    "insufficient_context",
    "bad_request",
    "validation_failed",
    "provider_unavailable",
    "internal_error",
]

Source = Literal["web", "api", "eval"]

_INSERT = """
INSERT INTO request_logs (
    id, source, question, question_hash, http_status, outcome, cache_hit, provider, model,
    fallback_used, validation_retries, rerank_used, rerank_error, prompt_version, index_version_id,
    retrieval_config_hash, latency_total_ms, latency_embed_ms, latency_retrieval_ms,
    latency_rerank_ms, latency_llm_ms, input_tokens, output_tokens, shadow_cost_usd,
    citation_count, invalid_citation_count, min_claim_confidence, error_code
) VALUES (
    %(id)s, %(source)s, %(question)s, %(question_hash)s, %(http_status)s, %(outcome)s,
    %(cache_hit)s, %(provider)s, %(model)s, %(fallback_used)s, %(validation_retries)s,
    %(rerank_used)s, %(rerank_error)s, %(prompt_version)s, %(index_version_id)s,
    %(retrieval_config_hash)s, %(latency_total_ms)s, %(latency_embed_ms)s,
    %(latency_retrieval_ms)s, %(latency_rerank_ms)s, %(latency_llm_ms)s, %(input_tokens)s,
    %(output_tokens)s, %(shadow_cost_usd)s, %(citation_count)s, %(invalid_citation_count)s,
    %(min_claim_confidence)s, %(error_code)s
)
"""


@dataclass(slots=True)
class RequestTrace:
    """What a request learns about itself on the way, for the log row and the response meta."""

    request_id: UUID
    timer: StageTimer
    question: str | None = None
    question_hash: str | None = None  # set directly when the question is not stored (bad_request)
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    index_version_id: int | None = None  # None in no_rag mode: there is no index
    retrieval_config_hash: str | None = None
    status: AnswerStatus | None = None  # of the answer; set only when one was produced
    embedding_model: str | None = None
    # Not reported by the embedder, so the question's length in characters stands in for tokens.
    # That is an upper bound (about 4 characters per token) of a very small number.
    embedding_tokens: int = 0
    input_tokens: int | None = None  # None until a generation attempt reported usage
    output_tokens: int | None = None
    validation_retries: int = 0
    cache_hit: bool = False  # answered from the answer cache (3.13): no embedding, no LLM call
    # Generation attempts the eval LLM cache served (4.03). No column: only eval runs have any, and
    # they read it from the trace to leave a cached latency out of the latency statistics.
    llm_cache_hits: int = 0
    citation_count: int | None = None
    invalid_citation_count: int | None = None
    dropped_claim_count: int | None = None
    removed_url_count: int | None = None  # links and URLs removed from the answer (citations.py)
    min_claim_confidence: float | None = None
    latency_total_ms: int | None = None  # frozen when the response meta is built
    written: bool = field(default=False, repr=False)

    def add_usage(self, usage: Usage) -> None:
        self.input_tokens = (self.input_tokens or 0) + usage.input_tokens
        self.output_tokens = (self.output_tokens or 0) + usage.output_tokens

    def shadow_cost(self, pricing: Pricing) -> Decimal:
        """What the work done so far costs: a request that failed halfway is charged for the part
        that ran (the embedding, and the generation attempts that reported usage)."""
        return pricing.shadow_cost(
            provider=self.provider,
            model=self.model if self.input_tokens is not None else None,
            input_tokens=self.input_tokens or 0,
            output_tokens=self.output_tokens or 0,
            embedding_model=self.embedding_model,
            embedding_tokens=self.embedding_tokens,
        )

    def total_ms(self) -> int:
        if self.latency_total_ms is None:
            return self.timer.total_ms()
        return self.latency_total_ms

    def latency_ms(self) -> dict[str, int]:
        """The response ``meta.latency_ms``: the four stages always present (0 when skipped, as in
        ``no_rag``), ``rerank`` only when it ran."""
        latency = {
            "total": self.total_ms(),
            "embed": self.timer.stage_ms("embed") or 0,
            "retrieval": self.timer.stage_ms("retrieval") or 0,
            "llm": self.timer.stage_ms("llm") or 0,
        }
        rerank = self.timer.stage_ms("rerank")
        if rerank is not None:
            latency["rerank"] = rerank
        return latency

    def digest(self) -> str:
        """``question_hash``; a request with no readable question (bad_request) still gets the
        stable hash of the empty question, since the column is NOT NULL."""
        if self.question_hash is not None:
            return self.question_hash
        return question_hash(self.question or "")


class RequestLogger:
    def __init__(
        self,
        pool: AsyncConnectionPool,
        pricing: Pricing,
        *,
        clock: Clock,
        source: Source = "api",
    ) -> None:
        self._pool = pool
        self._pricing = pricing
        self._clock = clock
        self._source: Source = source

    def start(self, request_id: UUID) -> RequestTrace:
        return RequestTrace(request_id=request_id, timer=StageTimer(self._clock))

    async def write(
        self,
        trace: RequestTrace,
        *,
        outcome: Outcome,
        http_status: int,
        error_code: ErrorCode | None = None,
    ) -> None:
        """Insert the row (once per trace). Never raises."""
        if trace.written:
            return
        trace.written = True
        total_ms = trace.total_ms()
        try:
            row = self._row(trace, outcome, http_status, error_code, total_ms)
            async with self._pool.connection() as conn:
                await conn.execute(_INSERT, row)
        except Exception as exc:
            # Type and SQLSTATE only: see the module docstring for why not the message.
            logger.error(
                "request log write failed",
                extra={
                    "request_id": str(trace.request_id),
                    "error": type(exc).__name__,
                    "sqlstate": exc.sqlstate if isinstance(exc, psycopg.Error) else None,
                },
            )
        # The stdout summary: the numbers that have no column, and the hash instead of the text.
        logger.info(
            "request completed",
            extra={
                "request_id": str(trace.request_id),
                "outcome": outcome,
                "http_status": http_status,
                "error_code": error_code,
                "latency_total_ms": total_ms,
                "validation_retries": trace.validation_retries,
                "invalid_citation_count": trace.invalid_citation_count,
                "dropped_claim_count": trace.dropped_claim_count,
                "removed_url_count": trace.removed_url_count,
                "question_hash": trace.digest(),
            },
        )

    def _row(
        self,
        trace: RequestTrace,
        outcome: Outcome,
        http_status: int,
        error_code: ErrorCode | None,
        total_ms: int,
    ) -> dict[str, Any]:
        timer = trace.timer
        return {
            "id": trace.request_id,
            "source": self._source,
            "question": trace.question,
            "question_hash": trace.digest(),
            "http_status": http_status,
            "outcome": outcome,
            "cache_hit": trace.cache_hit,
            "provider": trace.provider,
            "model": trace.model,
            "fallback_used": False,  # Phase 7
            "validation_retries": trace.validation_retries,
            "rerank_used": False,  # Phase 6
            "rerank_error": None,
            "prompt_version": trace.prompt_version,
            "index_version_id": trace.index_version_id,
            "retrieval_config_hash": trace.retrieval_config_hash,
            "latency_total_ms": total_ms,
            "latency_embed_ms": timer.stage_ms("embed"),
            "latency_retrieval_ms": timer.stage_ms("retrieval"),
            "latency_rerank_ms": timer.stage_ms("rerank"),
            "latency_llm_ms": timer.stage_ms("llm"),
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "shadow_cost_usd": trace.shadow_cost(self._pricing),
            "citation_count": trace.citation_count,
            "invalid_citation_count": trace.invalid_citation_count,
            "min_claim_confidence": trace.min_claim_confidence,
            "error_code": error_code,
        }
