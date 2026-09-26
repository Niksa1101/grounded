"""Ingest into the real pgvector test database with a fake embedder (Tech.md §5.7, DB.md §7.3)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, LiteralString

import psycopg
import pytest

from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import ProviderRateLimited, ProviderUnavailable
from grounded.ingest.embed import FakeEmbedder, TaskType, Vector, fake_vector
from grounded.ingest.pipeline import (
    EmbeddingQuotaExhaustedError,
    IngestError,
    IngestReport,
    PreparedCorpus,
    PreparedDocument,
    activate_index_version,
    index_spec,
    ingest,
    list_index_versions,
    prepare_corpus,
)
from grounded.ingest.types import ChunkingConfig, CorpusCheckout

pytestmark = pytest.mark.integration

CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"
CHECKOUT = CorpusCheckout(path=CORPUS_MINI, ref="0.0.1", sha="a" * 40)
CFG = ChunkingConfig(max_tokens=60, overlap_tokens=10, min_tokens=5, tokenizer="words")
DIM = 768  # chunks.embedding is vector(768)


def words(text: str) -> int:
    return len(text.split())


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    """The test database with no index versions before and after the test."""

    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    yield test_database_url
    wipe()


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[KVCache]:
    with KVCache(tmp_path / "embeddings.sqlite") as kv:
        yield kv


@pytest.fixture
def corpus() -> PreparedCorpus:
    return prepare_corpus(CHECKOUT, CFG, words)


def run(
    db: str,
    corpus: PreparedCorpus,
    embedder: Any,
    cache: KVCache,
    *,
    activate: bool = False,
    write_every: int = 100,
    max_input_tokens: int = 2048,
) -> IngestReport:
    spec = index_spec(
        corpus.checkout, corpus.chunking, embedding_model=embedder.model, embedding_dim=embedder.dim
    )
    return ingest(
        db,
        corpus,
        spec,
        embedder=embedder,
        cache=cache,
        write_every=write_every,
        count_tokens=words,
        max_input_tokens=max_input_tokens,
        activate=activate,
    )


def fetch(db: str, query: LiteralString, *params: Any) -> list[tuple[Any, ...]]:
    with psycopg.connect(db) as conn:
        return conn.execute(query, params).fetchall()


def sent_texts(embedder: FakeEmbedder) -> int:
    return sum(len(texts) for texts, _ in embedder.calls)


class QuotaAfter:
    """Embeds ``allowed`` texts, then fails like a spent daily quota."""

    def __init__(self, allowed: int) -> None:
        self.allowed = allowed
        self.inner = FakeEmbedder(dim=DIM)

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def dim(self) -> int:
        return self.inner.dim

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        if sent_texts(self.inner) + len(texts) > self.allowed:
            raise ProviderRateLimited("quota", retry_after_s=None, is_quota=True)
        return await self.inner.embed(texts, task_type)


# --- Happy path ------------------------------------------------------------------------------


def test_ingest_stores_a_ready_verified_version(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    embedder = FakeEmbedder(dim=DIM)
    report = run(db, corpus, embedder, cache)

    chunks = corpus.chunks
    assert report.created
    assert not report.activated
    assert (report.document_count, report.chunk_count) == (len(corpus.documents), len(chunks))
    assert (report.embedded, report.cache_hits) == (len(chunks), 0)
    assert embedder.calls[0][1] == "RETRIEVAL_DOCUMENT"

    [row] = list_index_versions(db)
    assert row.id == report.index_version_id
    assert (row.status, row.is_active) == ("ready", False)
    assert (row.git_ref, row.git_sha, row.embedding_model, row.embedding_dim) == (
        "0.0.1", "a" * 40, "fake-embedding", DIM
    )  # fmt: skip
    assert (row.document_count, row.chunk_count) == (len(corpus.documents), len(chunks))
    assert row.token_stats == report.token_stats
    assert report.token_stats["total"] == sum(c.token_count for c in chunks)

    [(config, ready_at)] = fetch(
        db, "SELECT chunking_config, ready_at FROM index_versions WHERE id = %s", row.id
    )
    assert config["parser_version"] >= 1
    assert "release-notes.md" in config["excluded_pages"]
    assert ready_at is not None


def test_ingest_stores_chunks_exactly_as_chunked(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    report = run(db, corpus, FakeEmbedder(dim=DIM), cache)
    rows = fetch(
        db,
        """
        SELECT d.source_path, c.ordinal, c.section_id, c.anchor_path, c.breadcrumb,
               c.breadcrumb_text, c.heading_level, c.url, c.content, c.token_count,
               c.content_hash, c.embedding::text, c.tsv <> ''::tsvector
        FROM chunks c JOIN documents d ON d.id = c.document_id
        WHERE c.index_version_id = %s
        ORDER BY d.source_path, c.ordinal
        """,
        report.index_version_id,
    )
    expected = sorted(
        ((p.doc.source_path, chunk) for p in corpus.documents for chunk in p.chunks),
        key=lambda item: (item[0], item[1].ordinal),
    )
    assert len(rows) == len(expected)
    for row, (source_path, chunk) in zip(rows, expected, strict=True):
        vector = [float(v) for v in row[11].strip("[]").split(",")]
        want = fake_vector(chunk.embedding_text, "RETRIEVAL_DOCUMENT", DIM)
        assert row[:11] == (
            source_path,
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
        )
        assert vector == pytest.approx(want, abs=1e-6)  # stored as float32
        assert row[12], "generated tsv is filled"

    documents = fetch(
        db,
        "SELECT source_path, url, title, content_hash FROM documents "
        "WHERE index_version_id = %s ORDER BY source_path",
        report.index_version_id,
    )
    assert documents == [
        (p.doc.source_path, p.doc.url, p.doc.title, p.doc.content_hash) for p in corpus.documents
    ]


# --- Re-runs and activation ------------------------------------------------------------------


def test_rerun_with_the_same_config_reuses_the_ready_version(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    embedder = FakeEmbedder(dim=DIM)
    first = run(db, corpus, embedder, cache)
    calls = len(embedder.calls)

    again = run(db, corpus, embedder, cache, activate=True)
    assert not again.created
    assert again.index_version_id == first.index_version_id
    assert again.activated
    assert (again.chunk_count, again.token_stats) == (first.chunk_count, first.token_stats)
    assert len(embedder.calls) == calls  # nothing embedded
    [row] = list_index_versions(db)
    assert row.is_active


def test_rebuild_is_served_from_the_embedding_cache(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    embedder = FakeEmbedder(dim=DIM)
    first = run(db, corpus, embedder, cache)
    with psycopg.connect(db, autocommit=True) as conn:  # e.g. retired by `index prune`
        conn.execute(
            "UPDATE index_versions SET status = 'retired' WHERE id = %s", (first.index_version_id,)
        )
    calls = len(embedder.calls)

    second = run(db, corpus, embedder, cache)
    assert second.created
    assert second.index_version_id != first.index_version_id
    assert (second.embedded, second.cache_hits) == (0, len(corpus.chunks))
    assert len(embedder.calls) == calls  # FR-3: zero embedding calls


def test_activation_leaves_exactly_one_active_version(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    embedder = FakeEmbedder(dim=DIM)
    first = run(db, corpus, embedder, cache, activate=True)
    other = dataclasses.replace(corpus, chunking=dataclasses.replace(CFG, max_tokens=61))
    second = run(db, other, embedder, cache, activate=True)
    assert second.index_version_id != first.index_version_id

    active = fetch(db, "SELECT id FROM index_versions WHERE is_active")
    assert active == [(second.index_version_id,)]

    with psycopg.connect(db, autocommit=True) as conn:
        activate_index_version(conn, first.index_version_id)
    assert fetch(db, "SELECT id FROM index_versions WHERE is_active") == [(first.index_version_id,)]


def test_only_a_ready_version_can_be_activated(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    ready = run(db, corpus, FakeEmbedder(dim=DIM), cache, activate=True)
    with psycopg.connect(db, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO index_versions (git_ref, git_sha, embedding_model, embedding_dim,"
            " chunking_config, config_hash, status)"
            " VALUES ('x', %s, 'm', 768, '{}', 'h', 'failed')",
            ("b" * 40,),
        )
        [(failed_id,)] = conn.execute("SELECT id FROM index_versions WHERE status = 'failed'")
        with pytest.raises(IngestError, match="not ready"):
            activate_index_version(conn, failed_id)
    # The failed attempt rolled back: the previous active version is still active.
    assert fetch(db, "SELECT id FROM index_versions WHERE is_active") == [(ready.index_version_id,)]


# --- Failures --------------------------------------------------------------------------------


def test_daily_quota_stop_writes_nothing_and_resumes_from_the_cache(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    total = len(corpus.chunks)
    assert total > 6
    with pytest.raises(EmbeddingQuotaExhaustedError) as excinfo:
        run(db, corpus, QuotaAfter(allowed=6), cache, write_every=3)
    assert (excinfo.value.remaining, excinfo.value.total) == (total - 6, total)
    assert fetch(db, "SELECT count(*) FROM index_versions") == [(0,)]

    # "The next day": only the rest is sent.
    resumed = QuotaAfter(allowed=total)
    report = run(db, corpus, resumed, cache, write_every=3)
    assert report.created
    assert (report.embedded, report.cache_hits) == (total - 6, 6)
    assert sent_texts(resumed.inner) == total - 6


def test_provider_failure_writes_nothing(db: str, corpus: PreparedCorpus, cache: KVCache) -> None:
    class Down(FakeEmbedder):
        async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
            raise ProviderUnavailable("503")

    with pytest.raises(ProviderUnavailable):
        run(db, corpus, Down(dim=DIM), cache)
    assert fetch(db, "SELECT count(*) FROM index_versions") == [(0,)]


def test_store_failure_leaves_only_a_failed_record(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    # A chunk the schema rejects (token_count must be > 0) fails the insert mid-transaction.
    first, *rest = corpus.documents
    broken_chunk = dataclasses.replace(first.chunks[0], token_count=0)
    broken = dataclasses.replace(
        corpus,
        documents=(PreparedDocument(first.doc, (broken_chunk, *first.chunks[1:])), *rest),
    )
    with pytest.raises(IngestError, match="recorded as failed index version") as excinfo:
        run(db, broken, FakeEmbedder(dim=DIM), cache, activate=True)
    assert isinstance(excinfo.value.__cause__, psycopg.errors.CheckViolation)

    [row] = list_index_versions(db)
    assert (row.status, row.is_active, row.chunk_count) == ("failed", False, None)
    assert row.notes is not None
    assert "CheckViolation" in row.notes
    assert fetch(db, "SELECT count(*) FROM documents") == [(0,)]
    assert fetch(db, "SELECT count(*) FROM chunks") == [(0,)]


def test_dimension_mismatch_fails_before_embedding(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    embedder = FakeEmbedder(dim=8)
    with pytest.raises(IngestError, match=r"vector\(768\)"):
        run(db, corpus, embedder, cache)
    assert embedder.calls == []
    assert fetch(db, "SELECT count(*) FROM index_versions") == [(0,)]


def test_too_long_input_fails_before_embedding(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    embedder = FakeEmbedder(dim=DIM)
    with pytest.raises(IngestError, match="embedding input limit"):
        run(db, corpus, embedder, cache, max_input_tokens=5)
    assert embedder.calls == []
    assert fetch(db, "SELECT count(*) FROM index_versions") == [(0,)]


def test_list_index_versions_is_newest_first(
    db: str, corpus: PreparedCorpus, cache: KVCache
) -> None:
    embedder = FakeEmbedder(dim=DIM)
    first = run(db, corpus, embedder, cache)
    second = run(
        db,
        dataclasses.replace(corpus, chunking=dataclasses.replace(CFG, max_tokens=61)),
        embedder,
        cache,
    )
    assert [row.id for row in list_index_versions(db)] == [
        second.index_version_id,
        first.index_version_id,
    ]
