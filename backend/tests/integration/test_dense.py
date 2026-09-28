"""Dense retrieval and the active index version on a hand-built index (DB.md §6.1, §7.3).

Every chunk vector is a unit vector at a known angle from the query in one plane, so its cosine
distance is ``1 - cos(angle)`` and the expected order is known without an embedding model.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator
from typing import Any

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import (  # pyright: ignore[reportMissingTypeStubs]
    register_vector,
    register_vector_async,
)

from grounded.retrieval.dense import dense_search
from grounded.retrieval.index import NoActiveIndexError, active_index_version
from grounded.retrieval.types import IndexVersion

pytestmark = pytest.mark.integration

DIM = 768
PAGE = "docs/en/docs/tutorial/page.md"
URL = "https://fastapi.tiangolo.com/tutorial/page/"
QUERY = [1.0] + [0.0] * (DIM - 1)
# 30 angles 0.0 … 2.9 rad, inserted out of order (7 is coprime with 30, so this is a permutation).
# cos falls monotonically on [0, π], so a bigger angle is a bigger distance.
STEPS = [(7 * i) % 30 for i in range(30)]
TIE_STEP = 0.5  # two extra chunks with the same vector, between steps 0 and 1


def at_angle(step: float) -> list[float]:
    angle = step * 0.1
    return [math.cos(angle), math.sin(angle)] + [0.0] * (DIM - 2)


def section(step: float) -> str:
    return f"{PAGE}#s{step:g}"


def insert_version(conn: psycopg.Connection[Any], *, active: bool, config_hash: str) -> int:
    row = conn.execute(
        """
        INSERT INTO index_versions (git_ref, git_sha, embedding_model, embedding_dim,
                                    chunking_config, config_hash, status, is_active)
        VALUES ('0.0.1', %s, 'fake-embedding', %s, '{}', %s, 'ready', %s)
        RETURNING id
        """,
        ("a" * 40, DIM, config_hash, active),
    ).fetchone()
    assert row is not None
    return int(row[0])


def insert_chunks(conn: psycopg.Connection[Any], version: int, steps: list[float]) -> None:
    doc = conn.execute(
        "INSERT INTO documents (index_version_id, source_path, url, title, content_hash) "
        "VALUES (%s, %s, %s, 'Page', 'h') RETURNING id",
        (version, PAGE, URL),
    ).fetchone()
    assert doc is not None
    for ordinal, step in enumerate(steps):
        anchor = f"s{step:g}"
        conn.execute(
            """
            INSERT INTO chunks (index_version_id, document_id, ordinal, section_id, anchor_path,
                                breadcrumb, breadcrumb_text, heading_level, url, content,
                                token_count, content_hash, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 2, %s, %s, %s, %s, %s)
            """,
            (
                version,
                doc[0],
                ordinal,
                section(step),
                ["h2", anchor],
                ["Tutorial", "Page", anchor],
                f"Tutorial > Page > {anchor}",
                f"{URL}#{anchor}",
                f"content {anchor}",
                10 + ordinal,
                f"hash-{version}-{ordinal}",
                Vector(at_angle(step)),
            ),
        )


@pytest.fixture
def index(test_database_url: str) -> Iterator[tuple[int, int]]:
    """Version 1 (active): 30 chunks plus a tied pair. Version 2 (inactive): one chunk that points
    exactly at the query, so a missing version filter would put it first."""

    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        register_vector(conn)
        active = insert_version(conn, active=True, config_hash="1" * 64)
        insert_chunks(conn, active, [*STEPS, TIE_STEP, TIE_STEP])
        other = insert_version(conn, active=False, config_hash="2" * 64)
        insert_chunks(conn, other, [0])
    yield active, other
    wipe()


@pytest.fixture
async def aconn(test_database_url: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    async with await psycopg.AsyncConnection.connect(test_database_url) as conn:
        await register_vector_async(conn)
        yield conn


async def test_nearest_first_with_ranks_and_distances(
    index: tuple[int, int], aconn: psycopg.AsyncConnection[Any]
) -> None:
    active, _ = index
    chunks = await dense_search(aconn, QUERY, index_version_id=active, k=5)

    assert [c.section_id for c in chunks] == [
        section(0),
        section(TIE_STEP),
        section(TIE_STEP),
        section(1),
        section(2),
    ]
    assert [c.dense_rank for c in chunks] == [1, 2, 3, 4, 5]
    distances = [c.dense_distance for c in chunks]
    expected = [1 - math.cos(step * 0.1) for step in (0, TIE_STEP, TIE_STEP, 1, 2)]
    assert distances == pytest.approx(expected, abs=1e-6)
    assert all(
        c.fts_rank is None and c.rrf_score is None and c.rerank_score is None for c in chunks
    )


async def test_ties_break_by_chunk_id(
    index: tuple[int, int], aconn: psycopg.AsyncConnection[Any]
) -> None:
    active, _ = index
    chunks = await dense_search(aconn, QUERY, index_version_id=active, k=3)
    tied = chunks[1:3]
    assert tied[0].dense_distance == tied[1].dense_distance
    assert tied[0].chunk_id < tied[1].chunk_id


async def test_searches_only_the_given_index_version(
    index: tuple[int, int], aconn: psycopg.AsyncConnection[Any]
) -> None:
    active, other = index
    in_active = await dense_search(aconn, QUERY, index_version_id=active, k=40)
    assert len(in_active) == 32  # every chunk of the version, no more (k beyond the rows is fine)
    in_other = await dense_search(aconn, QUERY, index_version_id=other, k=40)
    assert [c.section_id for c in in_other] == [section(0)]
    assert {c.chunk_id for c in in_active}.isdisjoint(c.chunk_id for c in in_other)


async def test_full_order_follows_the_angles(
    index: tuple[int, int], aconn: psycopg.AsyncConnection[Any]
) -> None:
    active, _ = index
    chunks = await dense_search(aconn, QUERY, index_version_id=active, k=40)
    expected = [0, TIE_STEP, TIE_STEP, *range(1, 30)]
    assert [c.section_id for c in chunks] == [section(step) for step in expected]


async def test_chunk_fields_are_mapped(
    index: tuple[int, int], aconn: psycopg.AsyncConnection[Any]
) -> None:
    active, _ = index
    [top] = await dense_search(aconn, QUERY, index_version_id=active, k=1)
    assert top.anchor_path == ("h2", "s0")
    assert top.breadcrumb_text == "Tutorial > Page > s0"
    assert top.url == f"{URL}#s0"
    assert top.content == "content s0"
    assert top.token_count == 10  # step 0 is ordinal 0
    assert top.content_hash == f"hash-{active}-0"


async def test_k_must_be_positive(aconn: psycopg.AsyncConnection[Any]) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        await dense_search(aconn, QUERY, index_version_id=1, k=0)


async def test_active_index_version(
    index: tuple[int, int], aconn: psycopg.AsyncConnection[Any]
) -> None:
    active, _ = index
    version = await active_index_version(aconn)
    assert version == IndexVersion(
        id=active,
        git_ref="0.0.1",
        git_sha="a" * 40,
        embedding_model="fake-embedding",
        embedding_dim=DIM,
        config_hash="1" * 64,
    )
    assert version.label == "0.0.1@11111111"

    await aconn.execute("UPDATE index_versions SET is_active = false")
    with pytest.raises(NoActiveIndexError, match="grounded ingest --activate"):
        await active_index_version(aconn)
    await aconn.rollback()
