"""The `/ask` flow: question, embedding, hybrid retrieval, context, generation, response.

``AskPipeline.ask`` is the one place the stages are wired together; the route stays thin and every
stage lives in its own module. Failures are raised as the typed errors of the stage that failed
(``ProviderError``, ``NoActiveIndexError``) and translated to HTTP in ``api/errors.py``.

Tracer-bullet state (Phase 3): the stages thicken in later tickets, and what is deliberately thin
today is marked below.

- Citation mapping (``_map_citations``) is the basic version: valid labels become display numbers
  by first appearance, invalid ones are dropped. Counting them, code fences and the semantic
  checks arrive with ``citations.py`` (3.06).
- ``_generate`` is one call with no retry (3.07 adds the single retry with feedback, and 7.04 moves
  it into the router).
- Claim confidence is ``0.0`` with no components until the heuristic lands (3.10), and
  ``shadow_cost_usd`` and the timings are partial until 3.12. The model's ``self_confidence`` is
  never shown as confidence (AGENTS.md §6.6).
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from uuid import UUID

from psycopg_pool import AsyncConnectionPool

from grounded.generation.context import build_context
from grounded.generation.prompts import Prompt
from grounded.generation.providers.base import GenerationResult, LLMProvider
from grounded.ingest.embed import Embedder
from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.hybrid import hybrid_search
from grounded.retrieval.index import ActiveIndexCache, chunk_titles
from grounded.retrieval.types import IndexVersion, RetrievedChunk
from grounded.schemas.api import AskResponse, Citation, Claim, Meta
from grounded.schemas.llm import LLMAnswer
from grounded.settings import Settings

_MARKER = re.compile(r"\[(c[1-9])\]")
_SNIPPET_CHARS = 300


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
    ) -> None:
        self._settings = settings
        self._pool = pool
        self._index_cache = index_cache
        self._embedder = embedder
        self._provider = provider
        self._prompt = prompt
        self._cfg = RetrievalConfig.from_settings(settings, "hybrid")

    async def ask(self, question: str, request_id: UUID) -> AskResponse:
        started = time.perf_counter()
        index = await self._index_cache.get()
        self._check_embedder(index)

        [vector] = await self._embedder.embed([question], "RETRIEVAL_QUERY")
        embedded = time.perf_counter()

        async with self._pool.connection() as conn:
            chunks = await hybrid_search(
                conn, question, vector, index_version_id=index.id, cfg=self._cfg
            )
            context = build_context(chunks, k_context=self._cfg.k_context)
            titles = await chunk_titles(conn, [c.chunk_id for c in context.labels.values()])
        # The connection is back in the pool before the (slow) LLM call.
        retrieved = time.perf_counter()

        user = self._prompt.render_user(question=question, sources=context.text)
        result = await self._generate(user)
        generated = time.perf_counter()

        markdown, claims, citations = _map_citations(result.parsed, context.labels, titles)
        return AskResponse(
            status=result.parsed.status,
            answer_markdown=markdown,
            claims=claims,
            citations=citations,
            follow_up_questions=result.parsed.follow_up_questions,
            min_confidence=min((c.confidence for c in claims), default=None),
            meta=Meta(
                request_id=request_id,
                provider=result.provider,
                model=result.model,
                fallback_used=False,
                cache_hit=False,
                rerank_used=False,
                prompt_version=self._prompt.version,
                index_version=index.label,
                retrieval_config_hash=self._cfg.config_hash,
                latency_ms={
                    "total": _ms(started, time.perf_counter()),
                    "embed": _ms(started, embedded),
                    "retrieval": _ms(embedded, retrieved),
                    "llm": _ms(retrieved, generated),
                },
                tokens={
                    "input": result.usage.input_tokens,
                    "output": result.usage.output_tokens,
                },
                shadow_cost_usd=0.0,
            ),
        )

    async def _generate(self, user: str) -> GenerationResult[LLMAnswer]:
        return await self._provider.generate(
            system=self._prompt.system,
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


def _ms(start: float, end: float) -> int:
    return round((end - start) * 1000)


def _map_citations(
    answer: LLMAnswer, labels: Mapping[str, RetrievedChunk], titles: Mapping[int, str]
) -> tuple[str, list[Claim], list[Citation]]:
    """Per-request labels -> display numbers (Tech §9.6, basic version).

    Numbers follow the first appearance of a label in the text; a label cited only by a claim comes
    after those. A label that is not in ``labels`` is removed, so every URL in the response comes
    from a DB row (AGENTS.md §6.3).
    """
    numbers: dict[str, int] = {}

    def number(label: str) -> int:
        return numbers.setdefault(label, len(numbers) + 1)

    def rewrite(match: re.Match[str]) -> str:
        return f"[{number(match[1])}]" if match[1] in labels else ""

    markdown = _MARKER.sub(rewrite, answer.answer_markdown)
    claims = [
        Claim(
            text=claim.text,
            citations=[
                number(label) for label in dict.fromkeys(claim.citation_ids) if label in labels
            ],
            confidence=0.0,
            confidence_components={},
        )
        for claim in answer.claims
    ]
    citations: list[Citation] = []
    for label, n in numbers.items():
        chunk = labels[label]
        citations.append(
            Citation.model_validate(
                {
                    "n": n,
                    "chunk_id": chunk.chunk_id,
                    "url": chunk.url,
                    "title": titles[chunk.chunk_id],
                    "breadcrumb": chunk.breadcrumb_text,
                    "snippet": chunk.content[:_SNIPPET_CHARS].strip(),
                }
            )
        )
    return markdown, claims, citations
