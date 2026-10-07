"""Question embedding for the request path (Tech.md §11).

Prod keeps vectors in a small in-memory LRU: the filesystem is ephemeral and a repeated question
is already served by the answer cache, so the LRU only has to absorb bursts within one instance.
Everywhere else the SQLite cache is shared with the evals (same key), so a question embedded by an
eval run costs nothing when asked locally.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import AsyncExitStack

from grounded.infra.kvcache import KVCache
from grounded.ingest.embed import (
    CachedEmbedder,
    Embedder,
    GeminiEmbedder,
    LazyEmbedder,
    TaskType,
    Vector,
)
from grounded.settings import Settings

_EMBEDDINGS_CACHE = "embeddings.sqlite"


class LRUEmbedder:
    """Wraps an ``Embedder`` with an in-memory LRU keyed by ``(task_type, sha256(text))``.

    Model and dim are fixed per instance, so they don't need to be part of the key.
    """

    def __init__(self, inner: Embedder, *, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._inner = inner
        self._max_size = max_size
        self._vectors: OrderedDict[tuple[str, str], Vector] = OrderedDict()

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        keys = [(task_type, hashlib.sha256(text.encode("utf-8")).hexdigest()) for text in texts]
        found = {key: self._vectors[key] for key in keys if key in self._vectors}
        missing = {key: text for key, text in zip(keys, texts, strict=True) if key not in found}
        if missing:
            fresh = await self._inner.embed(list(missing.values()), task_type)
            found.update(zip(missing, fresh, strict=True))
        for key in keys:
            self._vectors[key] = found[key]
            self._vectors.move_to_end(key)
        while len(self._vectors) > self._max_size:
            self._vectors.popitem(last=False)
        return [found[key] for key in keys]


def build_query_embedder(settings: Settings, stack: AsyncExitStack) -> Embedder:
    """The embedder the request path uses: Gemini, built lazily so no key is needed while every
    question is cached, behind the cache Tech §11 names for this environment.

    It is the request-path variant (``GeminiEmbedder.for_request_path``): one attempt, no pacing,
    no sleep. A rate limit reaches the caller at once; ``grounded ask --golden`` waits it out in the
    CLI layer (``evals/ask_batch.py``), never in here.
    """
    gemini = LazyEmbedder(
        settings.embedding_model,
        settings.embedding_dim,
        # Characters stand in for tokens: a question is at most 500 characters (AskRequest), so the
        # estimate only has to stay under the limits, and loading tiktoken (a file read, possibly a
        # download) inside the request path is not worth the precision.
        lambda: GeminiEmbedder.for_request_path(settings, len),
    )
    if settings.app_env == "prod":
        return LRUEmbedder(gemini, max_size=settings.query_embedding_cache_size)
    cache = stack.enter_context(KVCache(settings.cache_dir / _EMBEDDINGS_CACHE))
    return CachedEmbedder(gemini, cache, write_every=settings.embedding_batch_size)
