"""Test helpers importable from any test module (conftest.py is for fixtures only)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import psycopg
from pgvector import Vector

from grounded.generation.providers.base import LLMProvider
from grounded.generation.providers.fake import StubLLMProvider
from grounded.ingest.embed import Embedder, FakeEmbedder
from grounded.main import create_app
from grounded.settings import Settings

EMBEDDING_DIM = 768


class NetworkBlockedError(RuntimeError):
    """Raised by the conftest network guard when a test tries to reach a non-local host."""


def make_settings(**overrides: Any) -> Settings:
    """Settings for tests: ignores any developer .env, but still honors real env vars (CI)."""
    return Settings(_env_file=None, **{"app_env": "test", **overrides})  # pyright: ignore[reportCallIssue]


@asynccontextmanager
async def app_client(
    settings: Settings,
    *,
    embedder: Embedder | None = None,
    provider: LLMProvider | None = None,
    raise_app_exceptions: bool = True,
) -> AsyncGenerator[httpx.AsyncClient]:
    """An app built from ``settings`` with its lifespan running, plus an in-process HTTP client.

    The embedder and provider default to fakes, so no test builds a real one. A real server turns an
    unhandled error into a 500 response; pass ``raise_app_exceptions=False`` to see that here
    instead of having the exception re-raised in the test.
    """
    app = create_app(
        settings,
        embedder=embedder or FakeEmbedder(dim=settings.embedding_dim),
        provider=provider or StubLLMProvider(),
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=raise_app_exceptions)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


# --- Hand-built index rows (retrieval tests) -----------------------------------------------------
# Plain INSERTs into the real schema, so the generated ``tsv`` column and the constraints behave as
# they do in production. Each builder returns the new row's id.


@dataclass(frozen=True, slots=True)
class DocumentRow:
    id: int
    source_path: str
    url: str


def insert_index_version(
    conn: psycopg.Connection[Any], *, active: bool, config_hash: str, dim: int = EMBEDDING_DIM
) -> int:
    row = conn.execute(
        """
        INSERT INTO index_versions (git_ref, git_sha, embedding_model, embedding_dim,
                                    chunking_config, config_hash, status, is_active)
        VALUES ('0.0.1', %s, 'fake-embedding', %s, '{}', %s, 'ready', %s)
        RETURNING id
        """,
        ("a" * 40, dim, config_hash, active),
    ).fetchone()
    assert row is not None
    return int(row[0])


def insert_document(
    conn: psycopg.Connection[Any], version: int, *, source_path: str, url: str, title: str
) -> DocumentRow:
    row = conn.execute(
        "INSERT INTO documents (index_version_id, source_path, url, title, content_hash) "
        "VALUES (%s, %s, %s, %s, 'h') RETURNING id",
        (version, source_path, url, title),
    ).fetchone()
    assert row is not None
    return DocumentRow(id=int(row[0]), source_path=source_path, url=url)


def unit_vector(axis: int = 0) -> list[float]:
    """A unit vector along one axis: a valid, normalized embedding for tests that ignore vectors."""
    vector = [0.0] * EMBEDDING_DIM
    vector[axis] = 1.0
    return vector


def insert_chunk(
    conn: psycopg.Connection[Any],
    version: int,
    document: DocumentRow,
    *,
    ordinal: int,
    breadcrumb: Sequence[str],
    anchors: Sequence[str],
    content: str,
    embedding: Sequence[float] | None = None,
) -> int:
    """One chunk. ``anchors`` is the ``anchor_path`` (empty for a page intro, whose ``section_id``
    ends in ``#`` and whose ``url`` is the page's). ``embedding`` defaults to ``unit_vector()``."""
    anchor = anchors[-1] if anchors else ""
    row = conn.execute(
        """
        INSERT INTO chunks (index_version_id, document_id, ordinal, section_id, anchor_path,
                            breadcrumb, breadcrumb_text, heading_level, url, content,
                            token_count, content_hash, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
        RETURNING id
        """,
        (
            version,
            document.id,
            ordinal,
            f"{document.source_path}#{anchor}",
            list(anchors),
            list(breadcrumb),
            " > ".join(breadcrumb),
            len(anchors) + 1,
            f"{document.url}#{anchor}" if anchor else document.url,
            content,
            max(1, len(content.split())),
            f"hash-{version}-{document.id}-{ordinal}",
            Vector(list(embedding or unit_vector())).to_text(),
        ),
    ).fetchone()
    assert row is not None
    return int(row[0])
