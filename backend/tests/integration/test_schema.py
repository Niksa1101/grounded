"""0001_init applies cleanly and its constraints hold (DB.md §4)."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from grounded.infra.migrations import (
    DEFAULT_MIGRATIONS_DIR,
    ChecksumMismatchError,
    migrate,
)

pytestmark = pytest.mark.integration

_EXPECTED_TABLES = {
    "schema_migrations",
    "index_versions",
    "documents",
    "chunks",
    "answer_cache",
    "daily_usage",
    "request_logs",
    "eval_runs",
}

_INSERT_INDEX_VERSION = """
INSERT INTO index_versions
    (git_ref, git_sha, embedding_model, embedding_dim, chunking_config, config_hash,
     status, is_active)
VALUES (%s, %s, 'test-embedding', 768, '{}', %s, %s, %s)
RETURNING id
"""


@pytest.fixture
def conn(test_database_url: str) -> Iterator[psycopg.Connection]:
    """A connection whose work is rolled back, so tests don't leak rows into each other."""
    with psycopg.connect(test_database_url) as connection:
        yield connection
        connection.rollback()


def _insert_index_version(
    conn: psycopg.Connection, *, status: str = "ready", is_active: bool = False, tag: str = "a"
) -> int:
    row = conn.execute(
        _INSERT_INDEX_VERSION, (f"0.0.{tag}", tag * 40, f"hash-{tag}", status, is_active)
    ).fetchone()
    assert row is not None
    return row[0]


def test_all_tables_exist(conn: psycopg.Connection) -> None:
    rows = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
    ).fetchall()
    assert {name for (name,) in rows} >= _EXPECTED_TABLES


def test_pgvector_extension_installed(conn: psycopg.Connection) -> None:
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    assert row is not None


def test_migrate_is_idempotent(test_database_url: str) -> None:
    report = migrate(test_database_url)
    assert report.applied == []
    assert "0001_init" in report.already_applied


def test_edited_applied_migration_aborts_against_real_db(
    test_database_url: str, tmp_path: Path
) -> None:
    edited_dir = tmp_path / "migrations"
    shutil.copytree(DEFAULT_MIGRATIONS_DIR, edited_dir)
    init = edited_dir / "0001_init.sql"
    init.write_text(init.read_text(encoding="utf-8") + "\n-- edited\n", encoding="utf-8")
    with pytest.raises(ChecksumMismatchError):
        migrate(test_database_url, edited_dir)


def test_at_most_one_active_index_version(conn: psycopg.Connection) -> None:
    _insert_index_version(conn, is_active=True, tag="a")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_index_version(conn, is_active=True, tag="b")


def test_active_index_version_must_be_ready(conn: psycopg.Connection) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_index_version(conn, status="building", is_active=True)


def test_chunk_tsv_is_generated_with_heading_weight(conn: psycopg.Connection) -> None:
    version_id = _insert_index_version(conn)
    doc = conn.execute(
        """
        INSERT INTO documents (index_version_id, source_path, url, title, content_hash)
        VALUES (%s, 'docs/en/docs/tutorial/x.md', 'https://fastapi.tiangolo.com/tutorial/x/',
                'X', 'h')
        RETURNING id
        """,
        (version_id,),
    ).fetchone()
    assert doc is not None
    row = conn.execute(
        """
        INSERT INTO chunks (index_version_id, document_id, ordinal, section_id, anchor_path,
                            breadcrumb, breadcrumb_text, url, content, token_count, content_hash,
                            embedding)
        VALUES (%s, %s, 0, 'docs/en/docs/tutorial/x.md#', '{}', '{Tutorial,Background Tasks}',
                'Tutorial > Background Tasks', 'https://fastapi.tiangolo.com/tutorial/x/',
                'Run a function after returning a response.', 7, 'c',
                array_fill(0.0::real, ARRAY[768])::vector)
        RETURNING tsv::text
        """,
        (version_id, doc[0]),
    ).fetchone()
    assert row is not None
    tsv = row[0]
    # tsvector text looks like "'background':2A ... 'respons':9B": lexeme, positions, weight.
    # Breadcrumb lexemes carry weight A, body lexemes weight B.
    assert re.search(r"'background':[\d,]+A", tsv), tsv
    assert re.search(r"'respons':[\d,]+B", tsv), tsv
