"""Text embeddings: the Gemini adapter, a deterministic fake and a persistent cache (Tech.md §5.6).

``GeminiEmbedder`` talks to the API and nothing else; ``CachedEmbedder`` wraps any embedder so a
re-run with unchanged content makes zero API calls. Ingest uses ``CachedEmbedder(GeminiEmbedder)``,
tests use ``FakeEmbedder`` (pytest never reaches the network).

Free-tier limits for ``gemini-embedding-001`` (AI Studio, checked 2026-09-26): 100 RPM, 30K TPM,
1K RPD; RPD resets at midnight Pacific. Max 2,048 input tokens per text, at most 100 texts per
batch call. TPM is the binding limit for a full ingest (~230K tokens → roughly 10 minutes).

This is offline tooling, so the retry loop may sleep; the request path (Phase 3) embeds a single
question per request and never waits out a rate limit.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import random
import re
import struct
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import Any, Literal, Protocol, cast

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.settings import Settings

logger = logging.getLogger(__name__)

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
type Vector = list[float]
type TokenCounter = Callable[[str], int]

# Exponential backoff for retryable errors without a server-given delay: 2 s, 4 s, 8 s, … capped.
_BACKOFF_BASE_S = 2.0
_BACKOFF_MAX_S = 60.0
_WINDOW_S = 60.0  # RPM and TPM are per rolling minute


class Embedder(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def dim(self) -> int: ...

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        """One vector per text, in order, L2-normalized."""
        ...


class EmbeddingInputTooLongError(ValueError):
    """A text is over the model's input limit. Raised before any API call, so nothing is spent;
    the caller adds context (ingest names the chunk's ``section_id``)."""

    def __init__(self, index: int, tokens: int, limit: int) -> None:
        super().__init__(f"text #{index} has ~{tokens} tokens, over the {limit}-token input limit")
        self.index = index
        self.tokens = tokens
        self.limit = limit


def l2_normalize(values: Sequence[float]) -> Vector:
    """Scale to unit length. Truncated (768-dim) Gemini vectors aren't unit length, and the
    cosine distance in pgvector is only a clean ranking signal on normalized vectors."""
    norm = math.sqrt(math.fsum(v * v for v in values))
    if not math.isfinite(norm) or norm == 0.0:
        raise ProviderBadOutput(f"cannot normalize a vector with norm {norm}")
    return [v / norm for v in values]


# --- Gemini ---------------------------------------------------------------------------------------


class _RateWindow:
    """Requests and estimated tokens sent in the last 60 s, to stay under RPM and TPM."""

    def __init__(self, rpm: int, tpm: int, clock: Callable[[], float]) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._clock = clock
        self._sent: deque[tuple[float, int]] = deque()  # (sent_at, tokens), oldest first

    def wait_s(self, tokens: int) -> float:
        """Seconds until a request of ``tokens`` fits in the window (0 if it fits now)."""
        now = self._clock()
        while self._sent and self._sent[0][0] <= now - _WINDOW_S:
            self._sent.popleft()
        count, used = len(self._sent), sum(t for _, t in self._sent)
        if count < self._rpm and used + tokens <= self._tpm:
            return 0.0
        # Drop the oldest requests one by one until both limits have room; the wait is the moment
        # the last dropped one leaves the window.
        for sent_at, sent_tokens in self._sent:
            count, used = count - 1, used - sent_tokens
            if count < self._rpm and used + tokens <= self._tpm:
                return sent_at + _WINDOW_S - now
        raise AssertionError("unreachable: an empty window always has room")  # pragma: no cover

    def record(self, tokens: int) -> None:
        self._sent.append((self._clock(), tokens))


class GeminiEmbedder:
    """``gemini-embedding-001`` over the async ``google-genai`` client.

    Texts are packed into batches (≤ ``batch_size`` texts, ≤ ``tpm`` estimated tokens), paced to
    the RPM/TPM window, and retried on 429/5xx/timeouts with backoff. A daily-quota 429 is raised
    at once: waiting hours inside a CLI run helps nobody, and the cache keeps what was done.

    Token counts are tiktoken *estimates* (``count_tokens``); Gemini's tokenizer differs. The
    Gemini API has no ``auto_truncate`` switch (the SDK allows it on Vertex only), so the length
    check before the call is the only guard: keep real inputs well under the limit.
    """

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str,
        dim: int,
        count_tokens: TokenCounter,
        batch_size: int,
        rpm: int,
        tpm: int,
        max_input_tokens: int,
        max_retries: int,
        timeout_s: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._dim = dim
        self._count_tokens = count_tokens
        self._batch_size = batch_size
        self._tpm = tpm
        self._max_input_tokens = max_input_tokens
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._window = _RateWindow(rpm, tpm, clock)
        self.api_calls = 0  # requests actually sent, retries included (for ingest stats)

    @classmethod
    def from_settings(cls, settings: Settings, count_tokens: TokenCounter) -> GeminiEmbedder:
        if settings.gemini_api_key is None or not settings.embedding_model:
            raise ValueError("GEMINI_API_KEY and EMBEDDING_MODEL must be set to embed")
        client = genai.Client(api_key=settings.gemini_api_key.get_secret_value())
        return cls(
            client,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
            count_tokens=count_tokens,
            batch_size=settings.embedding_batch_size,
            rpm=settings.embedding_rpm,
            tpm=settings.embedding_tpm,
            max_input_tokens=settings.embedding_max_input_tokens,
            max_retries=settings.embedding_max_retries,
            timeout_s=settings.embedding_timeout_s,
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        sizes = [self._count_tokens(text) for text in texts]
        for index, tokens in enumerate(sizes):  # all of them first: fail before spending quota
            if tokens > self._max_input_tokens:
                raise EmbeddingInputTooLongError(index, tokens, self._max_input_tokens)
        vectors: list[Vector] = []
        for start, end in self._batches(sizes):
            batch = list(texts[start:end])
            vectors.extend(await self._embed_batch(batch, sum(sizes[start:end]), task_type))
        return vectors

    def _batches(self, sizes: Sequence[int]) -> Iterator[tuple[int, int]]:
        """``[start, end)`` ranges in order, each within the batch-size and TPM caps."""
        start, tokens = 0, 0
        for end, size in enumerate(sizes):
            if end > start and (end - start == self._batch_size or tokens + size > self._tpm):
                yield start, end
                start, tokens = end, 0
            tokens += size
        if start < len(sizes):
            yield start, len(sizes)

    async def _embed_batch(
        self, batch: list[str], tokens: int, task_type: TaskType
    ) -> list[Vector]:
        attempt = 0
        while True:
            if (wait := self._window.wait_s(tokens)) > 0:
                await self._sleep(wait)
            self._window.record(tokens)  # a rejected request still counts against the limits
            self.api_calls += 1
            try:
                return await self._call(batch, task_type)
            except (ProviderRateLimited, ProviderUnavailable, ProviderTimeout) as exc:
                if isinstance(exc, ProviderRateLimited) and exc.is_quota:
                    raise
                server_delay = exc.retry_after_s if isinstance(exc, ProviderRateLimited) else None
                delay = server_delay or self._backoff_s(attempt)
                error = exc
            attempt += 1
            if attempt > self._max_retries:
                raise error
            logger.warning(
                "embedding batch failed, retrying",
                extra={"error": type(error).__name__, "attempt": attempt, "delay_s": delay},
            )
            await self._sleep(delay)

    def _backoff_s(self, attempt: int) -> float:
        # "Equal jitter": half the delay is fixed, half random, so parallel runs spread out.
        delay = min(_BACKOFF_MAX_S, _BACKOFF_BASE_S * 2**attempt)
        return delay / 2 + self._rng.uniform(0, delay / 2)

    async def _call(self, batch: list[str], task_type: TaskType) -> list[Vector]:
        config = genai_types.EmbedContentConfig(
            task_type=task_type, output_dimensionality=self._dim
        )
        # Part of the SDK's content types refer to PIL (optional, not installed), which leaves the
        # method's signature partially unknown to pyright; pin the one shape we use.
        embed_content = cast(
            Callable[..., Awaitable[genai_types.EmbedContentResponse]],
            self._client.aio.models.embed_content,  # pyright: ignore[reportUnknownMemberType]
        )
        try:
            async with asyncio.timeout(self._timeout_s):
                # A list of strings: one embedding per string, sent as one batch call.
                response = await embed_content(model=self._model, contents=batch, config=config)
        except genai_errors.APIError as exc:
            raise _map_api_error(exc) from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderTimeout(f"embedding call exceeded {self._timeout_s}s") from exc
        except httpx.TransportError as exc:
            raise ProviderUnavailable(f"embedding transport error: {exc}") from exc
        return self._vectors(response, len(batch))

    def _vectors(self, response: genai_types.EmbedContentResponse, expected: int) -> list[Vector]:
        embeddings = response.embeddings or []
        if len(embeddings) != expected:
            raise ProviderBadOutput(f"asked for {expected} embeddings, got {len(embeddings)}")
        vectors: list[Vector] = []
        for embedding in embeddings:
            values = embedding.values or []
            if len(values) != self._dim:
                raise ProviderBadOutput(f"expected {self._dim} dimensions, got {len(values)}")
            vectors.append(l2_normalize(values))
        return vectors


def _map_api_error(exc: genai_errors.APIError) -> Exception:
    message = f"Gemini API {exc.code} {exc.status}: {exc.message}"
    if exc.code == 429:
        # ``details`` is the parsed error body and ``response`` the raw HTTP response; the SDK
        # types both loosely, so read them as Any and check shapes ourselves.
        raw = cast(Any, exc)
        details = _error_details(raw.details)
        headers = getattr(raw.response, "headers", None)
        return ProviderRateLimited(
            message,
            retry_after_s=_retry_after_s(details, headers),
            is_quota=_is_daily_quota(details),
        )
    if isinstance(exc, genai_errors.ServerError):
        return ProviderUnavailable(message)
    return ProviderRequestRejected(message, status_code=exc.code)


def _dicts(value: Any) -> list[dict[str, Any]]:
    """The dict items of ``value`` if it is a list; JSON from the wire has no guaranteed shape."""
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], item) for item in cast(list[Any], value) if isinstance(item, dict)]


def _error_details(details: Any) -> list[dict[str, Any]]:
    """The ``google.rpc`` detail objects of an error body (``{"error": {"details": [...]}}``)."""
    if not isinstance(details, dict):
        return []
    body = cast(dict[str, Any], details)
    inner = body.get("error", body)
    return _dicts(cast(dict[str, Any], inner).get("details")) if isinstance(inner, dict) else []


def _retry_after_s(details: list[dict[str, Any]], headers: object) -> float | None:
    """``google.rpc.RetryInfo.retryDelay`` (e.g. ``"53s"``), else a ``Retry-After`` header."""
    for item in details:
        if str(item.get("@type", "")).endswith("google.rpc.RetryInfo"):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(item.get("retryDelay", "")))
            if match:
                return float(match.group(1))
    value = headers.get("retry-after") if isinstance(headers, httpx.Headers) else None
    try:
        return float(value) if value is not None else None
    except ValueError:  # an HTTP date instead of seconds: fall back to our own backoff
        return None


def _is_daily_quota(details: list[dict[str, Any]]) -> bool:
    """A ``google.rpc.QuotaFailure`` naming a per-day quota (``...PerDay...`` quota ID)."""
    for item in details:
        if str(item.get("@type", "")).endswith("google.rpc.QuotaFailure") and any(
            "PerDay" in str(v.get("quotaId", "")) for v in _dicts(item.get("violations"))
        ):
            return True
    return False


# --- Fake -----------------------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic unit vectors derived from ``sha256(task_type | text)``, no network.

    Records every call so tests can assert how many texts reached "the provider".
    """

    def __init__(self, *, model: str = "fake-embedding", dim: int = 8) -> None:
        self._model = model
        self._dim = dim
        self.calls: list[tuple[list[str], TaskType]] = []

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        self.calls.append((list(texts), task_type))
        return [fake_vector(text, task_type, self._dim) for text in texts]


def fake_vector(text: str, task_type: str, dim: int) -> Vector:
    values: list[float] = []
    block = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{task_type}|{text}|{block}".encode()).digest()
        # 8 values per 32-byte digest, each from 4 bytes mapped to [-1, 1].
        values.extend(n / 2**31 - 1.0 for n in struct.unpack(">8I", digest))
        block += 1
    return l2_normalize(values[:dim])


# --- Cache ----------------------------------------------------------------------------------------


class CachedEmbedder:
    """Serve vectors from a ``KVCache`` and send only the misses to ``inner``.

    Key: ``model | dim | task_type | sha256(text)``; for chunks the text is ``breadcrumb_text +
    "\\n\\n" + content``, whose sha256 is ``chunks.content_hash``. Values are little-endian float32
    (768 dims = 3 KB). Misses go to ``inner`` in slices of ``write_every`` texts and each slice is
    stored before the next one is sent, so a run that dies on a quota error resumes where it
    stopped. Every returned vector went through the float32 round trip, so a fresh run and a
    cached re-run give bit-identical vectors.
    """

    def __init__(self, inner: Embedder, cache: KVCache, *, write_every: int) -> None:
        if write_every < 1:
            raise ValueError("write_every must be >= 1")
        self._inner = inner
        self._cache = cache
        self._write_every = write_every
        self.hits = 0
        self.misses = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dim(self) -> int:
        return self._inner.dim

    def cache_key(self, text: str, task_type: TaskType) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.model}|{self.dim}|{task_type}|{digest}"

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        keys = [self.cache_key(text, task_type) for text in texts]
        blobs = await asyncio.to_thread(self._cache.get_many, list(dict.fromkeys(keys)))
        # Each distinct missing text once, in first-seen order.
        pending = {key: text for key, text in zip(keys, texts, strict=True) if key not in blobs}
        missing = list(pending.items())
        for start in range(0, len(missing), self._write_every):
            part = missing[start : start + self._write_every]
            vectors = await self._inner.embed([text for _, text in part], task_type)
            new = {key: _pack(vector) for (key, _), vector in zip(part, vectors, strict=True)}
            await asyncio.to_thread(self._cache.put_many, new)
            blobs.update(new)
        self.misses += len(missing)
        self.hits += len(texts) - len(missing)
        return [_unpack(blobs[key], self.dim) for key in keys]


def _pack(vector: Vector) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def _unpack(blob: bytes, dim: int) -> Vector:
    if len(blob) != 4 * dim:
        raise ValueError(f"cached vector has {len(blob) // 4} dimensions, expected {dim}")
    return list(struct.unpack(f"<{dim}f", blob))
