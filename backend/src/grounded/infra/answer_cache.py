"""The answer cache in Postgres (Tech.md §11, DB.md §4 and §7.2, PRD FR-17).

A repeated question is answered from ``answer_cache`` without embedding, retrieval or an LLM call.
What is stored is the finished ``AskResponse`` **without** ``meta``: ``meta`` is per request (its
``request_id``, latency, tokens and cost describe one request, and a hit has none of the work), so
the pipeline rebuilds it on a hit (``from_stored``).

**The key** is the sha256 of everything that can change the stored answer: the normalized question
(``infra/hashing.py``, the same function as ``request_logs.question_hash``), the prompt version,
the index version id, the retrieval config hash, the generator model, a hash of the confidence
config (``confidence_config_hash``) and a hash of the generation parameters
(``GenerationParams.config_hash``: provider, temperature, max output tokens, thinking level). The
stored response carries server-computed confidence, so a changed ``CONFIDENCE_*`` weight must miss
instead of serving up to 30 days of stale numbers (PRD §12, closed in 3.13), and the same holds for
a changed generation parameter (PRD D47). Neither hash is a column. The parts are serialized as a
JSON array, not joined with ``|``, so a ``|`` inside a question can never shift a field.

**A row that no longer fits** ``AskResponse`` (written before a schema change that left the key
alone) is deleted on the lookup that finds it (``discard``), so the fresh answer of that request can
take its place. ``put`` only overwrites an *expired* row, so without the delete the question would
stay a miss until the row expired.

**Failure policy.** The cache is an optimization, so a database error on a lookup is a miss and on a
write (or a delete) is a skipped write. Both are logged (error type and SQLSTATE only: a Postgres
message can carry the failing row, i.e. the question, AGENTS.md §6.13) and never silent.

**What may be stored** is decided by the caller: only a valid ``AskResponse`` ever reaches ``put``
(``answered``, ``partial`` or ``insufficient_context``); a failed request raises before there is
one, so an error is never cached. The cache is off in eval mode and in ``no_rag``; that is the
pipeline's decision too.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, cast

import psycopg
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from grounded.generation.confidence import ConfidenceConfig
from grounded.generation.params import GenerationParams
from grounded.infra.hashing import normalize_question
from grounded.schemas.api import AskResponse, Meta

logger = logging.getLogger(__name__)

# Retention: the stored question is user text, kept at most 30 days (AGENTS.md §6.13, DB §4).
MAX_TTL_DAYS = 30

# A lookup that is a hit also records it (DB §7.2), in one statement: the row cannot expire between
# a SELECT and the UPDATE, and it is one round trip to Neon instead of two.
_HIT = """
UPDATE answer_cache
SET hit_count = hit_count + 1, last_hit_at = now()
WHERE cache_key = %s AND expires_at > now()
RETURNING response
"""

# DB §7.2 says DO NOTHING; the WHERE makes it "DO NOTHING unless the row has expired". An expired
# row stays in the table until the retention job deletes it, and a plain DO NOTHING would leave the
# question uncacheable until then. A live row (a concurrent request stored it first) is untouched.
_PUT = """
INSERT INTO answer_cache (
    cache_key, normalized_question, response, prompt_version, index_version_id,
    retrieval_config_hash, generator_model, expires_at
) VALUES (
    %(key)s, %(question)s, %(response)s, %(prompt_version)s, %(index_version_id)s,
    %(retrieval_config_hash)s, %(generator_model)s, now() + make_interval(days => %(ttl_days)s)
)
ON CONFLICT (cache_key) DO UPDATE SET
    normalized_question = EXCLUDED.normalized_question,
    response = EXCLUDED.response,
    hit_count = 0,
    created_at = now(),
    last_hit_at = NULL,
    expires_at = EXCLUDED.expires_at
WHERE answer_cache.expires_at <= now()
"""

_DISCARD = "DELETE FROM answer_cache WHERE cache_key = %s"


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Everything the key is made of; ``digest`` is the primary key, the rest are the columns the
    row stores next to it (so a row can be found by version without the question)."""

    question: str  # normalized
    prompt_version: str
    index_version_id: int
    retrieval_config_hash: str
    generator_model: str
    confidence_hash: str
    generation_hash: str

    @classmethod
    def build(
        cls,
        question: str,
        *,
        prompt_version: str,
        index_version_id: int,
        retrieval_config_hash: str,
        generator_model: str,
        confidence: ConfidenceConfig,
        generation: GenerationParams,
    ) -> CacheKey:
        return cls(
            question=normalize_question(question),
            prompt_version=prompt_version,
            index_version_id=index_version_id,
            retrieval_config_hash=retrieval_config_hash,
            generator_model=generator_model,
            confidence_hash=confidence_config_hash(confidence),
            generation_hash=generation.config_hash,
        )

    @property
    def digest(self) -> str:
        material = json.dumps(
            [
                self.question,
                self.prompt_version,
                self.index_version_id,
                self.retrieval_config_hash,
                self.generator_model,
                self.confidence_hash,
                self.generation_hash,
            ],
            ensure_ascii=False,
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


def confidence_config_hash(config: ConfidenceConfig) -> str:
    """A stable hash of the confidence weights and cap (a plain function of the value, like
    ``RetrievalConfig.config_hash``)."""
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def to_stored(response: AskResponse) -> dict[str, Any]:
    """The JSON that goes into ``answer_cache.response``: the response without ``meta``."""
    return response.model_dump(mode="json", exclude={"meta"})


def from_stored(stored: dict[str, Any], meta: Meta) -> AskResponse:
    """A cached response with this request's own ``meta`` attached."""
    return AskResponse.model_validate({**stored, "meta": meta})


class AnswerCache:
    def __init__(self, pool: AsyncConnectionPool, *, ttl_days: int) -> None:
        # Settings keeps the TTL within retention; the clamp keeps this class safe on its own.
        self._pool = pool
        self._ttl_days = min(ttl_days, MAX_TTL_DAYS)

    async def get(self, key: CacheKey) -> dict[str, Any] | None:
        """The stored response of a live row (and count the hit), or ``None`` on a miss. A database
        error is a miss (logged)."""
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(_HIT, (key.digest,))
                row = await cur.fetchone()
        except psycopg.Error as exc:
            _log_failure("answer cache lookup failed", exc)
            return None
        if row is None:
            return None
        stored: object = row[0]
        if not isinstance(stored, dict):
            # ``put`` always writes an object, so the row was changed by hand: drop it like a row
            # that no longer fits the schema (see ``discard``) and answer fresh.
            logger.warning("answer cache row is not a JSON object, treating as a miss")
            await self.discard(key)
            return None
        return cast("dict[str, Any]", stored)

    async def put(self, key: CacheKey, response: AskResponse) -> None:
        """Store a finished response. Never raises on a database error (logged)."""
        params = {
            "key": key.digest,
            "question": key.question,
            "response": Jsonb(to_stored(response)),
            "prompt_version": key.prompt_version,
            "index_version_id": key.index_version_id,
            "retrieval_config_hash": key.retrieval_config_hash,
            "generator_model": key.generator_model,
            "ttl_days": self._ttl_days,
        }
        try:
            async with self._pool.connection() as conn:
                await conn.execute(_PUT, params)
        except psycopg.Error as exc:
            _log_failure("answer cache write failed", exc)

    async def discard(self, key: CacheKey) -> None:
        """Delete the row of ``key`` (one that no longer fits ``AskResponse``). Never raises on a
        database error (logged): the row then stays a miss until it expires."""
        try:
            async with self._pool.connection() as conn:
                await conn.execute(_DISCARD, (key.digest,))
        except psycopg.Error as exc:
            _log_failure("answer cache delete failed", exc)


def _log_failure(message: str, exc: psycopg.Error) -> None:
    logger.warning(message, extra={"error": type(exc).__name__, "sqlstate": exc.sqlstate})
