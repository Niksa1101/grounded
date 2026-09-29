"""The migrations apply cleanly and their constraints hold (DB.md §4)."""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

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


# --- 0002_index_integrity --------------------------------------------------------------------

_INSERT_DOCUMENT = """
INSERT INTO documents (index_version_id, source_path, url, title, content_hash)
VALUES (%s, %s, 'https://fastapi.tiangolo.com/x/', 'X', 'h')
RETURNING id
"""

_INSERT_CHUNK = """
INSERT INTO chunks (index_version_id, document_id, ordinal, section_id, anchor_path, breadcrumb,
                    breadcrumb_text, url, content, token_count, content_hash, embedding)
VALUES (%s, %s, 0, 'docs/en/docs/x.md#', '{}', '{X}', 'X', 'https://fastapi.tiangolo.com/x/',
        'text', 1, 'c', array_fill(0.0::real, ARRAY[768])::vector)
"""


def _insert_document(conn: psycopg.Connection, version_id: int) -> int:
    row = conn.execute(_INSERT_DOCUMENT, (version_id, "docs/en/docs/x.md")).fetchone()
    assert row is not None
    return row[0]


def test_a_second_ready_version_with_the_same_config_hash_is_rejected(
    conn: psycopg.Connection,
) -> None:
    _insert_index_version(conn, tag="a")
    with pytest.raises(psycopg.errors.UniqueViolation) as excinfo:
        conn.execute(_INSERT_INDEX_VERSION, ("0.0.b", "b" * 40, "hash-a", "ready", False))
    assert excinfo.value.diag.constraint_name == "index_versions_one_ready_per_config"


@pytest.mark.parametrize("status", ["building", "failed", "retired"])
def test_only_ready_versions_compete_for_a_config_hash(
    conn: psycopg.Connection, status: str
) -> None:
    # A failed attempt, one still building, and a retired version can sit beside the ready one.
    _insert_index_version(conn, tag="a")
    conn.execute(_INSERT_INDEX_VERSION, ("0.0.b", "b" * 40, "hash-a", status, False))


def test_retiring_a_version_frees_its_config_for_a_rebuild(conn: psycopg.Connection) -> None:
    first = _insert_index_version(conn, tag="a")
    conn.execute("UPDATE index_versions SET status = 'retired' WHERE id = %s", (first,))
    _insert_index_version(conn, tag="a")  # same config, ready again


def test_a_chunk_cannot_point_at_a_document_of_another_version(conn: psycopg.Connection) -> None:
    version_a = _insert_index_version(conn, tag="a")
    version_b = _insert_index_version(conn, tag="b")
    document_of_a = _insert_document(conn, version_a)

    conn.execute(_INSERT_CHUNK, (version_a, document_of_a))  # the matching pair is fine
    with pytest.raises(psycopg.errors.ForeignKeyViolation) as excinfo:
        conn.execute(_INSERT_CHUNK, (version_b, document_of_a))
    assert excinfo.value.diag.constraint_name == "chunks_document_same_version_fkey"


def test_deleting_a_version_still_cascades_to_documents_and_chunks(
    conn: psycopg.Connection,
) -> None:
    version = _insert_index_version(conn, tag="a")
    conn.execute(_INSERT_CHUNK, (version, _insert_document(conn, version)))
    conn.execute("DELETE FROM index_versions WHERE id = %s", (version,))
    assert conn.execute("SELECT count(*) FROM documents").fetchone() == (0,)
    assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (0,)


def test_the_single_column_document_foreign_key_is_gone(conn: psycopg.Connection) -> None:
    names = {
        name
        for (name,) in conn.execute(
            "SELECT conname FROM pg_constraint WHERE conrelid = 'chunks'::regclass"
        ).fetchall()
    }
    assert "chunks_document_id_fkey" not in names
    assert "chunks_document_same_version_fkey" in names


def test_0002_applies_over_a_database_that_already_holds_v1_data(
    test_database_url: str, tmp_path: Path
) -> None:
    upgrade_db = "grounded_upgrade_test"
    params = conninfo_to_dict(test_database_url)
    assert params.get("host") in {"localhost", "127.0.0.1", "::1"}
    admin_url = make_conninfo(test_database_url, dbname="postgres")
    url = make_conninfo(test_database_url, dbname=upgrade_db)
    only_0001 = tmp_path / "migrations"
    only_0001.mkdir()
    shutil.copy(DEFAULT_MIGRATIONS_DIR / "0001_init.sql", only_0001)

    drop = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(upgrade_db))
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(drop)
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(upgrade_db)))
    try:
        assert migrate(url, only_0001).applied == ["0001_init"]
        with psycopg.connect(url) as conn:
            version = _insert_index_version(conn, tag="a", is_active=True)
            conn.execute(_INSERT_CHUNK, (version, _insert_document(conn, version)))
            conn.commit()

        assert migrate(url).applied == ["0002_index_integrity"]

        with psycopg.connect(url) as conn:
            assert conn.execute("SELECT count(*) FROM index_versions").fetchone() == (1,)
            assert conn.execute("SELECT count(*) FROM documents").fetchone() == (1,)
            assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (1,)
            with pytest.raises(psycopg.errors.UniqueViolation):
                _insert_index_version(conn, tag="a")
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(drop)
