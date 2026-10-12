"""Re-ranking: the Cohere adapter, a scriptable fake and the rerank cache (Tech §8, §11, PRD D55).

``Reranker`` is the contract the request path and the evals depend on (6.03, 6.04): hand it the
fused candidates in RRF order and get them back best first, each with ``rerank_score`` set.
Everything here builds that from one primitive, ``scores`` (one relevance score per candidate, in
input order), so the three implementations share the ranking and the argument checks:

- ``CohereReranker``: ``scores`` is one ``POST /v2/rerank`` over the SDK's ``AsyncClientV2``;
- ``FakeReranker``: scripted scores or errors, for tests (pytest never reaches Cohere);
- ``CachedReranker``: stores the scores of a call in ``.cache/rerank.sqlite`` and replays them, so
  a manual eval re-run makes no call. Dev and eval only: ``runtime.build_reranker`` does not put
  it in front of the adapter in prod, where the answer cache already covers repeats.

**``top_n`` is ours, not Cohere's.** The adapter sends every candidate and no ``top_n`` (Cohere then
returns all of them), and ``rank_chunks`` cuts the list. So the cache entry, which has no ``top_n``
in its key, serves any ``top_n``, and the retrieval eval can get the whole reranked list.

**One attempt, no sleeping.** The client is built with ``max_retries=0`` (the SDK's default is two
retries that sleep) and the call sits in ``asyncio.timeout(timeout_s)``; httpx's own timeout is per
phase, not total. Every failure becomes one of the typed provider errors (``ProviderRateLimited``,
``ProviderUnavailable``, ``ProviderTimeout``, ``ProviderRequestRejected``, ``ProviderBadOutput``),
and ``rerank_error_code`` names it for ``request_logs.rerank_error``. The request path never waits
for a ``Retry-After``; what to do with the error (degrade to the RRF order) is the caller's.

**The SDK is imported only when the adapter is built** (``CohereReranker.create``): touching
``AsyncClientV2`` costs about 0.65 s, which a deployment with ``RERANK_PROVIDER=none`` must not pay.
``api_key`` and ``base_url`` are passed explicitly because the SDK would otherwise read
``CO_API_KEY`` / ``CO_API_URL`` from the environment, and our code reads it only through
``Settings`` (AGENTS.md §6.7).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, Protocol, Self, cast

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
    compact_validation_error,
)
from grounded.retrieval.types import RetrievedChunk

if TYPE_CHECKING:
    from cohere.core import ApiError

logger = logging.getLogger(__name__)

COHERE_BASE_URL: Final = "https://api.cohere.com"
RERANK_CACHE_FILE: Final = "rerank.sqlite"

# How ``rerank_document`` builds the text Cohere sees. Part of the cache key: a candidate's
# ``content_hash`` covers the chunk, not this format, so changing the format means bumping this.
DOCUMENT_FORMAT_VERSION: Final = 1

# Error messages go into exceptions and logs; Cohere's text is short, but cap it anyway.
_MAX_DETAIL_CHARS = 300

# The ``request_logs.rerank_error`` codes (Tech §8). ``cap_reached`` is not an exception: the daily
# cap is checked before any call (6.03), so ``rerank_error_code`` never returns it.
type RerankErrorCode = Literal[
    "cap_reached", "rate_limited", "timeout", "unavailable", "rejected", "bad_output"
]


def rerank_document(chunk: RetrievedChunk) -> str:
    """The text Cohere ranks for a chunk: breadcrumb, a blank line, content.

    The same string that is embedded and hashed (``Chunk.embedding_text``): the heading context
    that helps retrieval helps the reranker too. The one place that builds it, so
    ``DOCUMENT_FORMAT_VERSION`` has a single home.
    """
    return f"{chunk.breadcrumb_text}\n\n{chunk.content}"


def rank_chunks(
    chunks: Sequence[RetrievedChunk], scores: Sequence[float], top_n: int
) -> list[RetrievedChunk]:
    """The ``top_n`` best chunks with ``rerank_score`` set, best first.

    Ties keep the input order (the RRF order), so equal scores rank the same on every run and the
    result never depends on how a provider happens to order equal results.
    """
    if len(scores) != len(chunks):
        raise ValueError(f"{len(scores)} scores for {len(chunks)} chunks")
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [replace(chunks[i], rerank_score=scores[i]) for i in order[:top_n]]


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        """``chunks`` (RRF order) reordered best first, at most ``top_n`` of them, each with
        ``rerank_score`` in ``[0, 1]``. Raises a ``ProviderError``; never sleeps or retries."""
        ...


class ScoringReranker(ABC):
    """A ``Reranker`` built from ``_scores``; the base owns the argument checks and the ranking."""

    def __init__(self, model: str) -> None:
        self.model = model

    async def scores(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """One relevance score in ``[0, 1]`` per chunk, in input order. No chunks, no call."""
        if not chunks:
            return []
        return await self._scores(query, chunks)

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        # Checked before the call: a bad argument must not spend a call of a scarce quota.
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        return rank_chunks(chunks, await self.scores(query, chunks), top_n)

    @abstractmethod
    async def _scores(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        """``scores`` for a non-empty ``chunks``."""


# --- Cohere ---------------------------------------------------------------------------------------


class RerankClient(Protocol):
    """The slice of the SDK's ``AsyncClientV2`` the adapter uses (tests inject a stand-in)."""

    async def rerank(self, *, model: str, query: str, documents: Sequence[str]) -> object: ...


class _Result(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    # Documented as normalized to [0, 1]; checked anyway, because confidence.py assumes the range.
    relevance_score: float = Field(ge=0.0, le=1.0)


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[_Result]


class CohereReranker(ScoringReranker):
    """``ScoringReranker`` over ``POST /v2/rerank`` (``rerank-v4.0-pro``, PRD D55)."""

    def __init__(
        self,
        client: RerankClient,
        *,
        model: str,
        timeout_s: float,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(model)
        self._client = client
        self._timeout_s = timeout_s
        self._close = close

    @classmethod
    def create(
        cls,
        api_key: str,
        *,
        model: str,
        timeout_s: float,
        http_client: httpx.AsyncClient | None = None,
    ) -> Self:
        """The adapter over a real SDK client. ``http_client`` is for tests (a mock transport)."""
        from cohere import AsyncClientV2  # here, not at the top: see the module docstring

        client = AsyncClientV2(
            api_key=api_key,
            base_url=COHERE_BASE_URL,
            timeout=timeout_s,
            max_retries=0,
            httpx_client=http_client,
        )

        async def close() -> None:
            # The SDK's only public way to close it; its untyped signature needs the cast.
            await cast(Any, client).__aexit__(None, None, None)

        return cls(client, model=model, timeout_s=timeout_s, close=close)

    async def aclose(self) -> None:
        if self._close is not None:
            await self._close()

    async def _scores(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        from cohere.core import ApiError, ParsingError  # loaded already: the client came from it

        documents = [rerank_document(chunk) for chunk in chunks]
        try:
            async with asyncio.timeout(self._timeout_s):
                response = await self._client.rerank(
                    model=self.model, query=query, documents=documents
                )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderTimeout(f"rerank call exceeded {self._timeout_s:g}s") from exc
        except httpx.TransportError as exc:  # after the timeouts: they are transport errors too
            raise ProviderUnavailable(f"rerank transport error: {exc}") from exc
        except ApiError as exc:
            raise _map_api_error(exc) from exc
        except ParsingError as exc:
            raise ProviderBadOutput("the Cohere SDK could not parse the rerank response") from exc
        return _scores_in_input_order(response, len(documents))


def _scores_in_input_order(response: object, expected: int) -> list[float]:
    """Check the response against what we sent and return the scores by document index.

    Cohere returns the results best first with an ``index`` into our documents. We asked for all of
    them, so a valid answer is a permutation: one result per document, no repeated index, none out
    of range. Anything else is ``ProviderBadOutput`` (the SDK does not validate, it only builds).
    """
    try:
        results = _Response.model_validate(response, from_attributes=True).results
    except ValidationError as exc:
        raise ProviderBadOutput(
            "the Cohere rerank response does not have the documented shape",
            validation_error=compact_validation_error(exc),
        ) from exc
    if len(results) != expected:
        raise ProviderBadOutput(f"Cohere returned {len(results)} results for {expected} documents")
    by_index: dict[int, float] = {}
    for result in results:
        if not 0 <= result.index < expected:
            raise ProviderBadOutput(f"result index {result.index} is outside 0..{expected - 1}")
        if result.index in by_index:
            raise ProviderBadOutput(f"result index {result.index} appears twice")
        by_index[result.index] = result.relevance_score
    return [by_index[index] for index in range(expected)]


def _map_api_error(exc: ApiError) -> ProviderError:
    """The SDK's ``ApiError`` (a subclass per status) as a typed error, by status (Tech §8)."""
    status = exc.status_code
    message = f"Cohere rerank {status}: {_detail(exc.body)}"
    if status == 429:
        # Cohere documents neither a Retry-After nor a different body for the monthly limit, so
        # the header is optional and ``is_quota`` is never claimed.
        return ProviderRateLimited(
            message, retry_after_s=_retry_after_s(exc.headers), is_quota=False
        )
    if status is None or status >= 500:
        return ProviderUnavailable(message)
    if 200 <= status < 300:  # the SDK got a success it could not read as JSON
        return ProviderBadOutput(message)
    return ProviderRequestRejected(message, status_code=status)


def _detail(body: object) -> str:
    """Cohere's ``message`` (error bodies are ``{"id", "message"}``), or a non-JSON body's text."""
    if isinstance(body, dict):
        text = cast(dict[str, Any], body).get("message", "")
    else:
        text = body if isinstance(body, str) else ""
    return str(text)[:_MAX_DETAIL_CHARS]


def _retry_after_s(headers: dict[str, str] | None) -> float | None:
    value = {name.lower(): text for name, text in (headers or {}).items()}.get("retry-after")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:  # an HTTP date instead of seconds: the caller has no wait to honor
        return None
    return seconds if seconds is not None and math.isfinite(seconds) and seconds >= 0 else None


def rerank_error_code(error: ProviderError) -> RerankErrorCode:
    """The ``request_logs.rerank_error`` code of a failed rerank call (Tech §8).

    Anything that is not one of the specific provider errors counts as ``unavailable``: the call
    produced no ranking, and the request goes on in RRF order either way.
    """
    if isinstance(error, ProviderRateLimited):
        return "rate_limited"
    if isinstance(error, ProviderTimeout):
        return "timeout"
    if isinstance(error, ProviderRequestRejected):
        return "rejected"
    if isinstance(error, ProviderBadOutput):
        return "bad_output"
    return "unavailable"


# --- Fake -----------------------------------------------------------------------------------------


class Hang:
    """A ``FakeReranker`` step: the call never returns, so only the caller's timeout ends it."""


type FakeStep = Sequence[float] | ProviderError | Hang


@dataclass(frozen=True)
class RecordedRerank:
    query: str
    documents: tuple[str, ...]


class FakeReranker(ScoringReranker):
    """A ``Reranker`` for tests: no network, no randomness (AGENTS.md §8).

    Each call that reaches ``scores`` is recorded in ``calls`` (also one that raises) and consumes
    the next step of ``script``: a sequence of scores (one per chunk, in input order), a
    ``ProviderError`` to raise, or ``Hang``. With no step left it scores every chunk from a hash of
    the query and its document text, so a bare ``FakeReranker()`` never runs out.
    """

    def __init__(self, script: Sequence[FakeStep] = (), *, model: str = "fake-rerank") -> None:
        super().__init__(model)
        self.calls: list[RecordedRerank] = []
        self._steps: deque[FakeStep] = deque(script)

    async def _scores(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        documents = tuple(rerank_document(chunk) for chunk in chunks)
        self.calls.append(RecordedRerank(query, documents))
        if not self._steps:
            return [fake_score(query, document) for document in documents]
        step = self._steps.popleft()
        if isinstance(step, ProviderError):
            raise step
        if isinstance(step, Hang):
            await asyncio.Event().wait()
            raise AssertionError("a hung call was resumed")  # pragma: no cover
        scores = list(step)
        if len(scores) != len(documents) or not all(0.0 <= score <= 1.0 for score in scores):
            raise ValueError("a scripted step needs one score in [0, 1] per chunk")
        return scores


def fake_score(query: str, document: str) -> float:
    """A stable pseudo-score in ``[0, 1)`` for a query and a document."""
    digest = hashlib.sha256(f"{query}\n{document}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# --- Cache ----------------------------------------------------------------------------------------


def rerank_cache_key(model: str, query: str, chunks: Sequence[RetrievedChunk]) -> str:
    """sha256 of the parts as a JSON array (a ``|`` in a query cannot shift a field).

    The candidates' content hashes are in input order, so the same set in another order is another
    key: the scores are stored by position, and the order is what the caller ranked. The document
    format version covers what the hashes do not; there is no ``top_n`` (see the module docstring).
    """
    parts = [DOCUMENT_FORMAT_VERSION, model, query, [chunk.content_hash for chunk in chunks]]
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode("utf-8")).hexdigest()


class _StoredScores(BaseModel):
    """One cache entry. ``extra="forbid"``: an entry of another layout is a miss, not half-read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scores: list[Annotated[float, Field(ge=0.0, le=1.0)]]


class CachedReranker(ScoringReranker):
    """Replays the stored scores of an identical call and only asks ``inner`` on a miss.

    An entry that no longer validates (a corrupt file, a wrong length) is deleted and counted as
    a miss, so the live scores replace it: one extra call, never a loop. Only a successful call is
    stored; an error from ``inner`` propagates and caches nothing.
    """

    def __init__(self, inner: ScoringReranker, cache: KVCache) -> None:
        super().__init__(inner.model)
        self._inner = inner
        self._cache = cache
        self.hits = 0
        self.misses = 0

    async def _scores(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        key = rerank_cache_key(self.model, query, chunks)
        stored = (await asyncio.to_thread(self._cache.get_many, [key])).get(key)
        if stored is not None:
            scores = self._replay(stored, len(chunks))
            if scores is not None:
                self.hits += 1
                return scores
            # Unusable: delete it, or the live scores could not take its place (first value wins).
            await asyncio.to_thread(self._cache.delete_many, [key])
        self.misses += 1
        scores = await self._inner.scores(query, chunks)
        entry = _StoredScores(scores=scores).model_dump_json().encode("utf-8")
        await asyncio.to_thread(self._cache.put_many, {key: entry})
        return scores

    def _replay(self, stored: bytes, expected: int) -> list[float] | None:
        """The stored scores, or ``None`` if the entry is not valid for ``expected`` chunks."""
        try:
            scores = _StoredScores.model_validate_json(stored).scores
            if len(scores) != expected:
                raise ValueError(f"{len(scores)} scores for {expected} chunks")
        except ValueError as exc:  # a ValidationError is one too
            # The type only: a log line has no business repeating a stored payload.
            logger.warning(
                "unusable rerank cache entry, replacing it", extra={"error": type(exc).__name__}
            )
            return None
        return scores
