"""Build one index version from a corpus checkout (Tech.md §5, §5.7; DB.md §4, §7.3).

    discover → parse → chunk ──► config hash ──► ready version with that hash? reuse it
                                      │ no
                                      ▼
             length check → embed (cached) → one transaction: insert, verify, mark ready
                                      ─► (activate)

Two ordering choices carry the failure semantics (Author decision, 2026-09-26):

- **Embedding happens before the database is touched.** A daily-quota stop (the first full
  ingest needs two days of free-tier RPD) leaves the database as it was; the SQLite cache keeps
  every vector already paid for, and a re-run the next day sends only the rest.
- **Insert and verify share one transaction.** If either fails, nothing of the version remains
  except one ``status='failed'`` row whose ``notes`` say why: a record, never a partial index.

The index config hash covers everything that changes the chunks or their vectors: corpus SHA,
embedding model and dimension, the chunking config, the page exclusion list and the parser
version. Re-running with the same hash reuses the ready version instead of building a duplicate.

Offline tooling: sync psycopg for the database (like the migration runner), with the async
embedder driven by ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import class_row
from psycopg.types.json import Jsonb

from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import ProviderRateLimited
from grounded.ingest.chunker import chunk_document
from grounded.ingest.corpus import EXCLUDED_PAGES, discover_pages
from grounded.ingest.embed import CachedEmbedder, Embedder, TaskType, embedding_cache_key
from grounded.ingest.markdown import PARSER_VERSION, parse_page
from grounded.ingest.tokens import TokenCounter
from grounded.ingest.types import Chunk, ChunkingConfig, CorpusCheckout, ParsedDocument

logger = logging.getLogger(__name__)

_TASK: TaskType = "RETRIEVAL_DOCUMENT"
_MAX_NOTES_CHARS = 2000
_MAX_LISTED_SECTIONS = 10


class IngestError(Exception):
    """Ingest stopped; the message says why and what (if anything) is left in the database."""


class EmbeddingQuotaExhaustedError(IngestError):
    """The provider's daily quota ran out mid-run. Nothing was written to the database; the
    vectors embedded so far are cached, so the same command resumes after the quota resets."""

    def __init__(self, *, remaining: int, total: int) -> None:
        super().__init__(
            f"daily embedding quota reached: {total - remaining} of {total} texts are cached, "
            f"{remaining} remain"
        )
        self.remaining = remaining
        self.total = total


# --- Corpus → chunks ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedDocument:
    doc: ParsedDocument
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class PreparedCorpus:
    """Every page of a checkout, parsed and chunked: what an index version will hold."""

    checkout: CorpusCheckout
    chunking: ChunkingConfig
    documents: tuple[PreparedDocument, ...]  # discover order (sorted by path)

    @property
    def chunks(self) -> list[Chunk]:
        return [chunk for prepared in self.documents for chunk in prepared.chunks]


def prepare_corpus(
    checkout: CorpusCheckout, cfg: ChunkingConfig, count_tokens: TokenCounter
) -> PreparedCorpus:
    documents: list[PreparedDocument] = []
    for page in discover_pages(checkout.path):
        doc = parse_page(page, checkout.path)
        documents.append(PreparedDocument(doc, tuple(chunk_document(doc, cfg, count_tokens))))
    if not any(prepared.chunks for prepared in documents):
        raise IngestError(f"no chunks produced from {checkout.path}")
    return PreparedCorpus(checkout, cfg, tuple(documents))


def token_stats(counts: Sequence[int], max_tokens: int) -> dict[str, int]:
    """Chunk size distribution for ``index_versions.token_stats``. Percentiles are nearest-rank
    (always an actual chunk size); ``over_max`` counts chunks allowed past ``max_tokens`` (an
    atomic code block or table larger than the budget)."""
    if not counts:
        raise ValueError("no token counts")
    ordered = sorted(counts)

    def percentile(p: int) -> int:
        return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]

    return {
        "min": ordered[0],
        "p50": percentile(50),
        "p95": percentile(95),
        "max": ordered[-1],
        "total": sum(ordered),
        "over_max": sum(1 for count in ordered if count > max_tokens),
    }


# --- Index identity -----------------------------------------------------------------------------


def index_chunking_config(cfg: ChunkingConfig) -> dict[str, Any]:
    """What ``index_versions.chunking_config`` stores: the chunker settings plus the two other
    things that change which chunks exist, the page exclusion list and the parser version."""
    return {
        **asdict(cfg),
        "excluded_pages": sorted(EXCLUDED_PAGES),
        "parser_version": PARSER_VERSION,
    }


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """The identity of an index version, known before anything is embedded or stored."""

    git_ref: str
    git_sha: str
    embedding_model: str
    embedding_dim: int
    chunking_config: dict[str, Any]
    config_hash: str

    @property
    def label(self) -> str:
        """``<git_ref>@<config_hash[:8]>``, the form responses and logs show (Tech.md §9.4)."""
        return f"{self.git_ref}@{self.config_hash[:8]}"


def index_spec(
    checkout: CorpusCheckout, cfg: ChunkingConfig, *, embedding_model: str, embedding_dim: int
) -> IndexSpec:
    chunking = index_chunking_config(cfg)
    canonical = json.dumps(chunking, sort_keys=True, separators=(",", ":"))
    material = f"{checkout.sha}|{embedding_model}|{embedding_dim}|{canonical}"
    return IndexSpec(
        git_ref=checkout.ref,
        git_sha=checkout.sha,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        chunking_config=chunking,
        config_hash=hashlib.sha256(material.encode("utf-8")).hexdigest(),
    )


def count_uncached(cache: KVCache, spec: IndexSpec, chunks: Sequence[Chunk]) -> int:
    """Distinct chunk texts with no cached vector: what an ingest would send to the provider."""
    keys = list(
        dict.fromkeys(
            embedding_cache_key(spec.embedding_model, spec.embedding_dim, _TASK, c.embedding_text)
            for c in chunks
        )
    )
    return len(keys) - len(cache.get_many(keys))


def check_input_lengths(chunks: Sequence[Chunk], count_tokens: TokenCounter, limit: int) -> None:
    """Refuse texts over the embedding input limit before any call, naming their sections.

    The embedder checks too, but only per slice it sends and without knowing which chunk a text
    came from; this check covers the whole corpus up front.
    """
    too_long = [
        f"{chunk.section_id} (~{tokens} tokens)"
        for chunk in chunks
        if (tokens := count_tokens(chunk.embedding_text)) > limit
    ]
    if too_long:
        listed = ", ".join(too_long[:_MAX_LISTED_SECTIONS])
        more = len(too_long) - _MAX_LISTED_SECTIONS
        suffix = f" and {more} more" if more > 0 else ""
        raise IngestError(
            f"{len(too_long)} chunk(s) over the {limit}-token embedding input limit: "
            f"{listed}{suffix}"
        )


# --- Orchestration ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IngestReport:
    index_version_id: int
    label: str
    created: bool  # False: a ready version with the same config hash already existed
    activated: bool
    document_count: int
    chunk_count: int
    token_stats: dict[str, int]
    cache_hits: int  # texts served from the embedding cache (0 when the version was reused)
    embedded: int  # distinct texts sent to the provider


def ingest(
    conninfo: str,
    corpus: PreparedCorpus,
    spec: IndexSpec,
    *,
    embedder: Embedder,
    cache: KVCache,
    write_every: int,
    count_tokens: TokenCounter,
    max_input_tokens: int,
    activate: bool,
) -> IngestReport:
    """Store ``corpus`` as the index version ``spec`` (or reuse it), optionally activating it.

    ``embedder`` is the provider adapter; it is wrapped in a ``CachedEmbedder`` over ``cache``
    that stores each slice of ``write_every`` new vectors before sending the next.
    """
    if (embedder.model, embedder.dim) != (spec.embedding_model, spec.embedding_dim):
        raise ValueError("embedder model/dim differ from the index spec")
    chunks = corpus.chunks
    stats = token_stats([c.token_count for c in chunks], corpus.chunking.max_tokens)

    with _connect(conninfo) as conn:
        column_dim = _embedding_column_dim(conn)
        if column_dim != spec.embedding_dim:
            raise IngestError(
                f"EMBEDDING_DIM is {spec.embedding_dim} but chunks.embedding is "
                f"vector({column_dim}); a different dimension needs a migration and a re-index"
            )
        existing = _find_ready_version(conn, spec.config_hash)
        if existing is not None:
            logger.info("index version already built", extra={"index_version_id": existing.id})
            if activate:
                activate_index_version(conn, existing.id)
            return IngestReport(
                index_version_id=existing.id,
                label=spec.label,
                created=False,
                activated=activate or existing.is_active,
                document_count=existing.document_count or 0,
                chunk_count=existing.chunk_count or 0,
                token_stats=existing.token_stats or stats,
                cache_hits=0,
                embedded=0,
            )

    check_input_lengths(chunks, count_tokens, max_input_tokens)
    cached = CachedEmbedder(embedder, cache, write_every=write_every)
    texts = [chunk.embedding_text for chunk in chunks]
    try:
        vectors = asyncio.run(cached.embed(texts, _TASK))
    except ProviderRateLimited as exc:
        if not exc.is_quota:
            raise
        remaining = count_uncached(cache, spec, chunks)
        total = len(set(texts))
        raise EmbeddingQuotaExhaustedError(remaining=remaining, total=total) from exc
    logger.info("chunks embedded", extra={"cache_hits": cached.hits, "embedded": cached.misses})

    with _connect(conninfo) as conn:
        try:
            version_id = _store(conn, corpus, spec, vectors, stats)
        except Exception as exc:
            failed_id = _record_failure(conn, spec, exc)
            where = (
                f" (recorded as failed index version {failed_id})" if failed_id is not None else ""
            )
            raise IngestError(f"storing the index failed{where}: {exc}") from exc
        if activate:
            activate_index_version(conn, version_id)

    return IngestReport(
        index_version_id=version_id,
        label=spec.label,
        created=True,
        activated=activate,
        document_count=len(corpus.documents),
        chunk_count=len(chunks),
        token_stats=stats,
        cache_hits=cached.hits,
        embedded=cached.misses,
    )


def activate_index_version(conn: psycopg.Connection[Any], index_version_id: int) -> None:
    """Make ``index_version_id`` the only active version, atomically (DB.md §7.3).

    Only a ``ready`` version can be activated; the table's partial unique index guarantees at most
    one active row even if two activations race.
    """
    with conn.transaction():
        conn.execute(_DEACTIVATE_OTHERS, (index_version_id,))
        cur = conn.execute(_ACTIVATE, (index_version_id,))
        if cur.rowcount != 1:
            raise IngestError(f"index version {index_version_id} does not exist or is not ready")
    logger.info("index version activated", extra={"index_version_id": index_version_id})


# --- Database -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IndexVersionRow:
    id: int
    git_ref: str
    git_sha: str
    embedding_model: str
    embedding_dim: int
    config_hash: str
    status: str
    is_active: bool
    document_count: int | None
    chunk_count: int | None
    token_stats: dict[str, int] | None
    notes: str | None
    created_at: datetime


_SELECT_VERSIONS = """
SELECT id, git_ref, git_sha, embedding_model, embedding_dim, config_hash, status, is_active,
       document_count, chunk_count, token_stats, notes, created_at
FROM index_versions
"""

_INSERT_VERSION = """
INSERT INTO index_versions
    (git_ref, git_sha, embedding_model, embedding_dim, chunking_config, config_hash, status, notes)
VALUES
    (%(git_ref)s, %(git_sha)s, %(embedding_model)s, %(embedding_dim)s, %(chunking_config)s,
     %(config_hash)s, %(status)s, %(notes)s)
RETURNING id
"""

_INSERT_DOCUMENT = """
INSERT INTO documents (index_version_id, source_path, url, title, content_hash)
VALUES (%s, %s, %s, %s, %s)
RETURNING id
"""

_INSERT_CHUNK = """
INSERT INTO chunks
    (index_version_id, document_id, ordinal, section_id, anchor_path, breadcrumb, breadcrumb_text,
     heading_level, url, content, token_count, content_hash, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_VERIFY = """
SELECT
    (SELECT count(*) FROM documents WHERE index_version_id = %(id)s) AS documents,
    count(*) AS chunks,
    count(*) FILTER (WHERE embedding IS NULL) AS null_embeddings,
    count(*) FILTER (WHERE vector_dims(embedding) <> %(dim)s) AS wrong_dims
FROM chunks
WHERE index_version_id = %(id)s
"""

_MARK_READY = """
UPDATE index_versions
SET status = 'ready', ready_at = now(),
    document_count = %(documents)s, chunk_count = %(chunks)s, token_stats = %(token_stats)s
WHERE id = %(id)s
"""

_DEACTIVATE_OTHERS = "UPDATE index_versions SET is_active = false WHERE is_active AND id <> %s"
_ACTIVATE = "UPDATE index_versions SET is_active = true WHERE id = %s AND status = 'ready'"

# typmod of a vector(n) column is n (pgvector stores the dimension there).
_EMBEDDING_COLUMN_DIM = """
SELECT atttypmod FROM pg_attribute
WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
"""


def list_index_versions(conninfo: str) -> list[IndexVersionRow]:
    with _connect(conninfo) as conn:
        cur = conn.cursor(row_factory=class_row(IndexVersionRow))
        return cur.execute(_SELECT_VERSIONS + "ORDER BY id DESC").fetchall()


def _connect(conninfo: str) -> psycopg.Connection[Any]:
    # autocommit: every write below runs inside an explicit ``conn.transaction()`` block.
    conn = psycopg.connect(conninfo, autocommit=True)
    try:
        register_vector(conn)
    except psycopg.ProgrammingError as exc:
        conn.close()
        raise IngestError("the vector extension is missing: run `grounded migrate` first") from exc
    return conn


def _embedding_column_dim(conn: psycopg.Connection[Any]) -> int:
    row = conn.execute(_EMBEDDING_COLUMN_DIM).fetchone()
    if row is None:
        raise IngestError("chunks.embedding not found: run `grounded migrate` first")
    return int(row[0])


def _find_ready_version(conn: psycopg.Connection[Any], config_hash: str) -> IndexVersionRow | None:
    cur = conn.cursor(row_factory=class_row(IndexVersionRow))
    query = (
        _SELECT_VERSIONS + "WHERE config_hash = %s AND status = 'ready' ORDER BY id DESC LIMIT 1"
    )
    return cur.execute(query, (config_hash,)).fetchone()


def _store(
    conn: psycopg.Connection[Any],
    corpus: PreparedCorpus,
    spec: IndexSpec,
    vectors: Sequence[Sequence[float]],
    stats: dict[str, int],
) -> int:
    """Insert the version, its documents and chunks, verify, mark ready: all or nothing."""
    chunk_count = len(corpus.chunks)
    if len(vectors) != chunk_count:
        raise IngestError(f"{len(vectors)} vectors for {chunk_count} chunks")
    with conn.transaction(), conn.cursor() as cur:
        version_id = _insert_version(conn, spec, status="building", notes=None)

        cur.executemany(
            _INSERT_DOCUMENT,
            [
                (version_id, p.doc.source_path, p.doc.url, p.doc.title, p.doc.content_hash)
                for p in corpus.documents
            ],
            returning=True,
        )
        document_ids: list[int] = []
        for _ in cur.results():
            row = cur.fetchone()
            assert row is not None  # every INSERT ... RETURNING yields one row
            document_ids.append(int(row[0]))

        rows: list[tuple[Any, ...]] = []
        remaining = iter(vectors)
        for document_id, prepared in zip(document_ids, corpus.documents, strict=True):
            for chunk in prepared.chunks:
                rows.append(_chunk_row(version_id, document_id, chunk, next(remaining)))
        cur.executemany(_INSERT_CHUNK, rows)

        verified = _verify(conn, version_id, spec, len(corpus.documents), chunk_count)
        conn.execute(_MARK_READY, {**verified, "token_stats": Jsonb(stats), "id": version_id})
    logger.info(
        "index version stored",
        extra={
            "index_version_id": version_id,
            "documents": len(document_ids),
            "chunks": chunk_count,
        },
    )
    return version_id


def _chunk_row(
    version_id: int, document_id: int, chunk: Chunk, vector: Sequence[float]
) -> tuple[Any, ...]:
    return (
        version_id,
        document_id,
        chunk.ordinal,
        chunk.section_id,
        list(chunk.anchor_path),
        list(chunk.breadcrumb),
        chunk.breadcrumb_text,
        chunk.heading_level,
        chunk.url,
        chunk.content,
        chunk.token_count,
        chunk.content_hash,
        Vector(list(vector)),
    )


def _verify(
    conn: psycopg.Connection[Any],
    version_id: int,
    spec: IndexSpec,
    expected_documents: int,
    expected_chunks: int,
) -> dict[str, int]:
    """Read back what was written (Tech.md §5.7). Golden-set label checks live in
    ``golden validate --against-index``, since the golden set is versioned separately."""
    row = conn.execute(_VERIFY, {"id": version_id, "dim": spec.embedding_dim}).fetchone()
    assert row is not None  # an aggregate query always returns one row
    documents, chunks, null_embeddings, wrong_dims = (int(value) for value in row)
    problems: list[str] = []
    if chunks == 0:
        problems.append("no chunks")
    if (documents, chunks) != (expected_documents, expected_chunks):
        problems.append(
            f"stored {documents} documents / {chunks} chunks, "
            f"expected {expected_documents} / {expected_chunks}"
        )
    if null_embeddings:
        problems.append(f"{null_embeddings} chunks without an embedding")
    if wrong_dims:
        problems.append(f"{wrong_dims} embeddings without {spec.embedding_dim} dimensions")
    if problems:
        raise IngestError("verification failed: " + "; ".join(problems))
    return {"documents": documents, "chunks": chunks}


def _insert_version(
    conn: psycopg.Connection[Any], spec: IndexSpec, *, status: str, notes: str | None
) -> int:
    params = {
        "git_ref": spec.git_ref,
        "git_sha": spec.git_sha,
        "embedding_model": spec.embedding_model,
        "embedding_dim": spec.embedding_dim,
        "chunking_config": Jsonb(spec.chunking_config),
        "config_hash": spec.config_hash,
        "status": status,
        "notes": notes,
    }
    row = conn.execute(_INSERT_VERSION, params).fetchone()
    assert row is not None  # INSERT ... RETURNING yields one row
    return int(row[0])


def _record_failure(conn: psycopg.Connection[Any], spec: IndexSpec, exc: Exception) -> int | None:
    """Leave a ``failed`` row (no documents or chunks) explaining the failure. Best effort: if the
    connection itself is gone this can fail too, and the original error is what matters."""
    notes = f"{type(exc).__name__}: {exc}"[:_MAX_NOTES_CHARS]
    try:
        with conn.transaction():
            return _insert_version(conn, spec, status="failed", notes=notes)
    except psycopg.Error as record_exc:
        logger.warning(
            "could not record the failed index version", extra={"error": str(record_exc)}
        )
        return None
