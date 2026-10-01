"""Spec for the hybrid query (DB.md §6.3), written before the Author implements it (ticket 2.06).

The index is hand-made (``tests/hybrid_corpus.py``): every chunk has a known angle from the query
vector and some repeat a rare word, so the dense list, the lexical list and the fused order are all
known without running any query. The numbers below are worked out by hand with ``rrf_k = 60``:
a chunk at rank ``r`` in a list adds ``1 / (60 + r)``; a chunk in no list adds nothing.
Each test pins one point of the contract, which is the docstring of ``hybrid_search``.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Iterator, Mapping
from itertools import pairwise
from typing import Any, NamedTuple

import psycopg
import pytest
from pgvector import Vector
from pgvector.psycopg import (  # pyright: ignore[reportMissingTypeStubs]
    register_vector_async,
)

from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.dense import dense_search
from grounded.retrieval.hybrid import hybrid_search
from grounded.retrieval.lexical import lexical_search
from grounded.retrieval.types import RetrievedChunk
from tests.hybrid_corpus import CORPUS, PAGE, RADIANS_PER_STEP, Spec, at_angle, insert_page
from tests.support import insert_index_version, unit_vector

pytestmark = pytest.mark.integration

QUERY = unit_vector(0)
# The whole fused order for "quokka" with the default config, worked out below.
QUOKKA_FULL_ORDER = [
    "q_top",
    "q_third",
    "q_fourth",
    "q_second",
    "q_fifth",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "w_top",
    "w_second",
]
# The dense order of the whole index: ``step`` ascending.
DENSE_ORDER = [spec.key for spec in sorted(CORPUS, key=lambda s: s.step)]


class Index(NamedTuple):
    active: int
    other: int  # an inactive version whose only chunk is the best match for everything
    ids: dict[str, int]  # corpus key -> chunk id in the active version
    other_chunk: int


@pytest.fixture
def index(test_database_url: str) -> Iterator[Index]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        active = insert_index_version(conn, active=True, config_hash="1" * 64)
        ids = insert_page(conn, active, CORPUS)
        other = insert_index_version(conn, active=False, config_hash="2" * 64)
        # Nearest to the query and the most "quokka" there is: it would win both lists if it leaked.
        [other_chunk] = insert_page(conn, other, (Spec("other", 0, "quokka", 9),)).values()
    yield Index(active, other, ids, other_chunk)
    wipe()


@pytest.fixture
async def aconn(test_database_url: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    async with await psycopg.AsyncConnection.connect(test_database_url) as conn:
        await register_vector_async(conn)
        yield conn


def cfg(**overrides: int) -> RetrievalConfig:
    values: dict[str, Any] = {
        "mode": "hybrid",
        "k_dense": 20,
        "k_fts": 20,
        "k_fused": 40,
        "k_context": 5,
        "rrf_k": 60,
    }
    return RetrievalConfig(**{**values, **overrides})


async def search(
    aconn: psycopg.AsyncConnection[Any],
    index: Index,
    question: str,
    **config: int,
) -> list[RetrievedChunk]:
    return await hybrid_search(
        aconn,
        question,
        QUERY,
        index_version_id=index.active,
        cfg=cfg(**config),
    )


def keys(index: Index, chunks: list[RetrievedChunk]) -> list[str]:
    """Corpus keys of the result, in order (an unknown id shows up as ``?<id>``)."""
    by_id = {chunk_id: key for key, chunk_id in index.ids.items()}
    return [by_id.get(c.chunk_id, f"?{c.chunk_id}") for c in chunks]


def by_key(index: Index, chunks: list[RetrievedChunk]) -> dict[str, RetrievedChunk]:
    return dict(zip(keys(index, chunks), chunks, strict=True))


def rrf(*ranks: int, k: int = 60) -> float:
    """Σ 1/(k + rank) over the ranks the chunk has: the DB.md §6.3 formula, for hand-made rows."""
    return sum(1 / (k + rank) for rank in ranks)


# --- RRF math -------------------------------------------------------------------------------------


async def test_rank_one_in_both_lists_scores_two_over_61(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """``q_top`` is the nearest chunk and has the most "quokka": 1/61 + 1/61."""
    chunks = by_key(index, await search(aconn, index, "quokka"))
    top = chunks["q_top"]
    assert (top.dense_rank, top.fts_rank) == (1, 1)
    assert top.rrf_score == pytest.approx(2 / 61)


async def test_rank_one_in_one_list_only_scores_one_over_61(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """With k_dense = 3 the dense list is q_top, f1, q_third. "wombat" matches only w_top and
    w_second, which are far from the query: q_top is rank 1 in dense only, w_top is rank 1 in
    lexical only. A missing rank contributes 0, not a penalty."""
    chunks = by_key(index, await search(aconn, index, "wombat", k_dense=3))
    assert (chunks["q_top"].dense_rank, chunks["q_top"].fts_rank) == (1, None)
    assert chunks["q_top"].rrf_score == pytest.approx(1 / 61)
    assert (chunks["w_top"].dense_rank, chunks["w_top"].fts_rank) == (None, 1)
    assert chunks["w_top"].rrf_score == pytest.approx(1 / 61)


# (dense rank, lexical rank) of every chunk for "quokka" with the default config (20/20/40). The
# index has 13 chunks, so the dense list holds all of them; the lexical list is the five "quokka"
# chunks. A rank missing from a list is None.
QUOKKA_RANKS: dict[str, tuple[int | None, int | None]] = {
    "q_top": (1, 1),
    "f1": (2, None),
    "q_third": (3, 3),
    "f2": (4, None),
    "f3": (5, None),
    "f4": (6, None),
    "q_fourth": (7, 4),
    "f5": (8, None),
    "f6": (9, None),
    "q_fifth": (10, 5),
    "w_top": (11, None),
    "w_second": (12, None),
    "q_second": (13, 2),
}


@pytest.mark.parametrize("key", list(QUOKKA_RANKS))
async def test_the_score_is_the_sum_over_the_lists_a_chunk_is_in(
    index: Index, aconn: psycopg.AsyncConnection[Any], key: str
) -> None:
    dense_rank, fts_rank = QUOKKA_RANKS[key]
    chunk = by_key(index, await search(aconn, index, "quokka"))[key]
    assert (chunk.dense_rank, chunk.fts_rank) == (dense_rank, fts_rank)
    assert chunk.rrf_score == pytest.approx(rrf(*(r for r in (dense_rank, fts_rank) if r)))


async def test_the_whole_fused_order_for_one_question(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """Scores, best first: q_top 2/61, q_third 2/63, q_fourth 1/67 + 1/64, q_second 1/73 + 1/62,
    q_fifth 1/70 + 1/65, then the dense-only chunks f1 ... f6 (1/62 ... 1/69) and w_top 1/71,
    w_second 1/72. A chunk in both lists can fall behind one with a better rank in one list."""
    chunks = await search(aconn, index, "quokka")
    assert keys(index, chunks) == QUOKKA_FULL_ORDER


async def test_rrf_k_comes_from_the_config(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    chunks = by_key(index, await search(aconn, index, "quokka", rrf_k=10))
    assert chunks["q_top"].rrf_score == pytest.approx(2 / 11)
    assert chunks["q_third"].rrf_score == pytest.approx(2 / 13)
    assert chunks["f1"].rrf_score == pytest.approx(1 / 12)


# --- Full outer join ------------------------------------------------------------------------------


async def test_a_chunk_found_only_lexically_appears_without_a_dense_rank(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """k_dense = 5 leaves q_second (the farthest chunk) out of the dense list, but it is the
    second best "quokka" match: it must still come back, with no dense signal."""
    chunks = by_key(index, await search(aconn, index, "quokka", k_dense=5))
    chunk = chunks["q_second"]
    assert chunk.dense_rank is None
    assert chunk.dense_distance is None
    assert chunk.fts_rank == 2
    assert chunk.fts_score is not None
    assert chunk.rrf_score == pytest.approx(1 / 62)


async def test_a_chunk_found_only_densely_appears_without_a_lexical_rank(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """f1 has no "quokka" in it but is the second nearest chunk."""
    chunks = by_key(index, await search(aconn, index, "quokka"))
    chunk = chunks["f1"]
    assert chunk.fts_rank is None
    assert chunk.fts_score is None
    assert chunk.dense_rank == 2
    assert chunk.dense_distance == pytest.approx(1 - math.cos(1 * RADIANS_PER_STEP), abs=1e-6)
    assert chunk.rrf_score == pytest.approx(1 / 62)


async def test_the_signals_of_a_chunk_in_both_lists(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    chunk = by_key(index, await search(aconn, index, "quokka"))["q_third"]
    assert chunk.dense_rank == 3
    assert chunk.dense_distance == pytest.approx(1 - math.cos(2 * RADIANS_PER_STEP), abs=1e-6)
    assert chunk.fts_rank == 3
    assert chunk.fts_score is not None
    assert chunk.fts_score > 0
    assert chunk.rerank_score is None


# --- Cutting the lists and the output -------------------------------------------------------------


async def test_the_dense_list_is_cut_to_k_dense_before_fusion(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """k_dense = 3 keeps q_top, f1, q_third. q_fourth is the 7th nearest chunk, so it must get no
    dense rank and no dense share of the score, only its lexical 1/64. f2 (4th nearest) is cut and
    has no lexical match: it is not in the result at all."""
    chunks = await search(aconn, index, "quokka", k_dense=3)
    assert keys(index, chunks) == ["q_top", "q_third", "q_second", "f1", "q_fourth", "q_fifth"]
    by = by_key(index, chunks)
    assert by["q_fourth"].dense_rank is None
    assert by["q_fourth"].rrf_score == pytest.approx(1 / 64)
    assert by["q_fifth"].rrf_score == pytest.approx(1 / 65)
    assert by["q_third"].rrf_score == pytest.approx(2 / 63)


async def test_the_lexical_list_is_cut_to_k_fts_before_fusion(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """k_fts = 2 keeps q_top and q_second. q_third is the 3rd best match, so it must get no
    lexical rank and keeps only its dense 1/63."""
    chunks = by_key(index, await search(aconn, index, "quokka", k_fts=2))
    assert chunks["q_third"].fts_rank is None
    assert chunks["q_third"].rrf_score == pytest.approx(1 / 63)
    assert chunks["q_fourth"].fts_rank is None
    assert chunks["q_fifth"].fts_rank is None
    assert chunks["q_second"].fts_rank == 2
    assert chunks["q_second"].rrf_score == pytest.approx(1 / 73 + 1 / 62)
    assert chunks["q_top"].rrf_score == pytest.approx(2 / 61)


async def test_the_output_is_cut_to_k_fused_unique_chunks(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """With k_dense = k_fts = 5 the lists have ten entries but only eight distinct chunks
    (q_top and q_third are in both). The result is those eight, once each, and nothing else."""
    chunks = await search(aconn, index, "quokka", k_dense=5, k_fts=5)
    assert len(chunks) == len({c.chunk_id for c in chunks}) == 8
    assert sorted(keys(index, chunks)) == sorted(
        ["q_top", "q_third", "q_second", "q_fourth", "q_fifth", "f1", "f2", "f3"]
    )


async def test_k_fused_keeps_the_best_chunks_and_never_exceeds_k(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    full = await search(aconn, index, "quokka")
    assert len(full) == 13  # fewer than k_fused = 40: every chunk of either list, once
    for k in (1, 3, 8):
        top = await search(aconn, index, "quokka", k_fused=k)
        assert keys(index, top) == QUOKKA_FULL_ORDER[:k]


# --- Tie-break ------------------------------------------------------------------------------------


async def test_equal_scores_come_back_ordered_by_chunk_id(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """With k_dense = 3, "wombat" gives two ties between a lexical-only and a dense-only chunk:
    w_top (lexical rank 1) with q_top (dense rank 1), and f1 (dense rank 2) with w_second (lexical
    rank 2). In the first tie the lexical-only chunk has the lower id, in the second the
    dense-only one does, so a tie-break that favors one list would get one of them wrong."""
    assert index.ids["w_top"] < index.ids["q_top"]
    assert index.ids["f1"] < index.ids["w_second"]
    chunks = await search(aconn, index, "wombat", k_dense=3)
    assert keys(index, chunks) == ["w_top", "q_top", "f1", "w_second", "q_third"]
    scores = [c.rrf_score for c in chunks]
    assert all(score is not None for score in scores)
    first, second, third, fourth, fifth = (score or 0.0 for score in scores)
    assert first == second
    assert third == fourth
    assert second > third > fifth


async def test_a_tie_is_broken_before_the_cut_to_k_fused(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """With k_dense = k_fts = 5, q_second and f1 tie for the 3rd place (1/62 each). q_second has
    the lower id, so k_fused = 3 must keep it and drop f1, on every run."""
    assert index.ids["q_second"] < index.ids["f1"]
    for _ in range(3):
        chunks = await search(aconn, index, "quokka", k_dense=5, k_fts=5, k_fused=3)
        assert keys(index, chunks) == ["q_top", "q_third", "q_second"]


async def test_the_whole_order_is_score_descending_then_id_ascending(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    configs: list[dict[str, int]] = [{"k_dense": 5, "k_fts": 5}, {"k_dense": 3}, {}]
    for config in configs:
        chunks = await search(aconn, index, "quokka", **config)
        for before, after in pairwise(chunks):
            assert before.rrf_score is not None
            assert after.rrf_score is not None
            assert before.rrf_score >= after.rrf_score
            if before.rrf_score == after.rrf_score:
                assert before.chunk_id < after.chunk_id


async def test_the_result_is_reproducible(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    first = await search(aconn, index, "quokka wombat", k_dense=5, k_fts=5)
    second = await search(aconn, index, "quokka wombat", k_dense=5, k_fts=5)
    assert first == second != []


# --- A question without lexemes -------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    ["how do I do it", "the a an of to", "", "   ", "?! ... --", "zeppelin"],
    ids=["stop-words", "more-stop-words", "empty", "blank", "punctuation", "no-match"],
)
async def test_nothing_to_search_for_gives_the_dense_ranking(
    index: Index, aconn: psycopg.AsyncConnection[Any], question: str
) -> None:
    """RRF over one list: the dense order, every score 1/(60 + dense rank), no lexical signal.
    "zeppelin" has lexemes but no chunk contains them, which is the same situation."""
    chunks = await search(aconn, index, question)
    assert keys(index, chunks) == DENSE_ORDER
    assert [c.dense_rank for c in chunks] == list(range(1, 14))
    assert [c.rrf_score for c in chunks] == pytest.approx([1 / (60 + r) for r in range(1, 14)])
    assert all(c.fts_rank is None and c.fts_score is None for c in chunks)


async def test_the_dense_ranking_is_still_cut_to_k_dense_and_k_fused(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    assert keys(index, await search(aconn, index, "how do I do it", k_dense=4)) == DENSE_ORDER[:4]
    assert keys(index, await search(aconn, index, "how do I do it", k_fused=2)) == DENSE_ORDER[:2]


# --- Filtering and the vector ---------------------------------------------------------------------


async def test_searches_only_the_given_index_version(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """The other version's chunk is the nearest and has "quokka" nine times, so it would be rank
    1 in both lists (and push q_top to rank 2) if it leaked in."""
    in_active = await search(aconn, index, "quokka")
    assert keys(index, in_active) == QUOKKA_FULL_ORDER
    assert index.other_chunk not in {c.chunk_id for c in in_active}
    assert in_active[0].dense_rank == 1
    assert in_active[0].rrf_score == pytest.approx(2 / 61)

    in_other = await hybrid_search(aconn, "quokka", QUERY, index_version_id=index.other, cfg=cfg())
    assert [c.chunk_id for c in in_other] == [index.other_chunk]
    assert in_other[0].rrf_score == pytest.approx(2 / 61)
    assert {c.chunk_id for c in in_active}.isdisjoint(c.chunk_id for c in in_other)


async def test_the_query_vector_decides_the_dense_list(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """A vector at the far end of the corpus reverses the dense order, so q_second (the farthest
    chunk from ``QUERY``) becomes the nearest one."""
    chunks = await hybrid_search(aconn, "", at_angle(40), index_version_id=index.active, cfg=cfg())
    assert keys(index, chunks)[0] == "q_second"
    assert chunks[0].dense_rank == 1
    assert chunks[0].dense_distance == pytest.approx(0, abs=1e-6)


def _as_text(query: object) -> str:
    if isinstance(query, bytes):
        return query.decode()
    as_string = getattr(query, "as_string", None)  # psycopg.sql.Composable
    return str(as_string(None)) if callable(as_string) else str(query)


@pytest.fixture
async def recording_conn(
    test_database_url: str,
) -> AsyncIterator[tuple[psycopg.AsyncConnection[Any], list[tuple[str, object]]]]:
    """A connection that records every statement it runs, as (SQL text, bound parameters)."""
    log: list[tuple[str, object]] = []

    class RecordingCursor(psycopg.AsyncCursor[Any]):
        async def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
            log.append((_as_text(query), params))
            return await super().execute(query, params, **kwargs)

    async with await psycopg.AsyncConnection.connect(
        test_database_url, cursor_factory=RecordingCursor
    ) as conn:
        await register_vector_async(conn)
        log.clear()  # the adapter registration looked up types: not our business
        yield conn, log


async def test_one_statement_and_every_value_is_bound(
    index: Index,
    recording_conn: tuple[psycopg.AsyncConnection[Any], list[tuple[str, object]]],
) -> None:
    """DB.md §6.3: one statement with the CTEs dense, lexical and fused; the question and the
    vector are parameters, never part of the SQL text (AGENTS.md §6.11)."""
    conn, log = recording_conn
    question = "quokka'; DROP TABLE chunks; --"
    vector = at_angle(0.123456)  # digits that would show if the vector were formatted into SQL
    await hybrid_search(conn, question, vector, index_version_id=index.active, cfg=cfg())

    assert len(log) == 1
    sql, params = log[0]
    lowered = sql.lower()
    assert "with" in lowered
    for cte in ("dense", "lexical", "fused"):
        assert cte in lowered
    assert "drop table" not in lowered
    assert repr(vector[0]) not in sql
    assert repr(vector[1]) not in sql
    values = params.values() if isinstance(params, Mapping) else params
    assert isinstance(values, (list, tuple, type({}.values())))
    assert any(isinstance(v, Vector) for v in values)


# --- Hostile input --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "'",
        "'; DROP TABLE chunks; --",
        "a & | b",
        "((((",
        "\\",
        "quokka:*",
        "http://x.com/it's/a(b)!c|d\\e",
        "日本語のテスト",
        "x" * 5000,
    ],
    ids=lambda q: q[:24],
)
async def test_special_characters_never_raise(
    index: Index, aconn: psycopg.AsyncConnection[Any], question: str
) -> None:
    chunks = await search(aconn, index, question)
    assert all(isinstance(c, RetrievedChunk) for c in chunks)
    # The connection is still usable: a failed statement would have aborted the transaction.
    assert await search(aconn, index, "wombat") != []


async def test_operators_and_nul_bytes_are_words_not_syntax(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """The same words give the same result with or without punctuation or NUL bytes between
    them: ``!`` must not exclude, ``&`` must not require."""
    plain = await search(aconn, index, "quokka wombat", k_dense=5)
    assert plain != []
    for question in ("quokka:* & !(wombat | 'x') \\", "quokka\x00wombat", "\x00quokka\x00 wombat"):
        chunks = await search(aconn, index, question, k_dense=5)
        assert [(c.chunk_id, c.rrf_score) for c in chunks] == [
            (c.chunk_id, c.rrf_score) for c in plain
        ]


# --- Consistency oracle ---------------------------------------------------------------------------
# Fusing what ``dense_search`` and ``lexical_search`` return, in Python, must agree with the single
# statement. This checks two code paths against each other, not the SQL itself: both lists must be
# produced the same way in either path, and the arithmetic and tie-break must be the same.


def fuse(
    dense: list[RetrievedChunk], lexical: list[RetrievedChunk], *, rrf_k: int, k_fused: int
) -> list[tuple[int, int | None, int | None, float]]:
    """(chunk id, dense rank, fts rank, rrf score), best first: score descending, id ascending."""
    dense_ranks = {c.chunk_id: c.dense_rank for c in dense}
    fts_ranks = {c.chunk_id: c.fts_rank for c in lexical}
    fused: list[tuple[int, int | None, int | None, float]] = []
    for chunk_id in dense_ranks.keys() | fts_ranks.keys():
        d, f = dense_ranks.get(chunk_id), fts_ranks.get(chunk_id)
        fused.append((chunk_id, d, f, sum(1 / (rrf_k + r) for r in (d, f) if r is not None)))
    fused.sort(key=lambda row: (-row[3], row[0]))
    return fused[:k_fused]


@pytest.mark.parametrize(
    ("question", "config"),
    [
        ("quokka", {}),
        ("wombat", {"k_dense": 3}),
        ("quokka wombat", {}),
        ("quokka wombat plain", {"k_dense": 4, "k_fts": 3, "k_fused": 6, "rrf_k": 20}),
        ("plain filler paragraph quokka", {"k_dense": 6, "k_fts": 6, "rrf_k": 5}),
        ("how do I do it", {"k_dense": 7}),
        ("zeppelin", {}),
    ],
    ids=["quokka", "wombat-k3", "two-words", "small-ks", "many-words", "stop-words", "no-match"],
)
async def test_agrees_with_fusing_dense_and_lexical_search_in_python(
    index: Index,
    aconn: psycopg.AsyncConnection[Any],
    question: str,
    config: dict[str, int],
) -> None:
    config_ = cfg(**config)
    dense = await dense_search(aconn, QUERY, index_version_id=index.active, k=config_.k_dense)
    lexical = await lexical_search(aconn, question, index_version_id=index.active, k=config_.k_fts)
    expected = fuse(dense, lexical, rrf_k=config_.rrf_k, k_fused=config_.k_fused)
    assert expected != []

    actual = await hybrid_search(aconn, question, QUERY, index_version_id=index.active, cfg=config_)
    assert [(c.chunk_id, c.dense_rank, c.fts_rank) for c in actual] == [
        (chunk_id, d, f) for chunk_id, d, f, _ in expected
    ]
    assert [c.rrf_score for c in actual] == pytest.approx([score for *_, score in expected])

    distances = {c.chunk_id: c.dense_distance for c in dense}
    fts_scores = {c.chunk_id: c.fts_score for c in lexical}
    for chunk in actual:
        assert chunk.dense_distance == pytest.approx(distances.get(chunk.chunk_id), abs=1e-9)
        assert chunk.fts_score == pytest.approx(fts_scores.get(chunk.chunk_id), abs=1e-6)


# --- Result mapping -------------------------------------------------------------------------------


async def test_chunk_fields_are_mapped(index: Index, aconn: psycopg.AsyncConnection[Any]) -> None:
    source_path, url, _title = PAGE
    ordinal = [spec.key for spec in CORPUS].index("q_top")
    spec = next(s for s in CORPUS if s.key == "q_top")
    chunk = by_key(index, await search(aconn, index, "quokka"))["q_top"]
    assert chunk.chunk_id == index.ids["q_top"]
    assert chunk.section_id == f"{source_path}#part-{ordinal}"
    assert chunk.anchor_path == (f"part-{ordinal}",)
    assert chunk.breadcrumb_text == f"Hybrid > Part {ordinal}"
    assert chunk.url == f"{url}#part-{ordinal}"
    assert chunk.content == spec.content
    assert chunk.token_count == len(spec.content.split())
    # insert_chunk builds "hash-<version>-<document id>-<ordinal>"
    assert chunk.content_hash.startswith(f"hash-{index.active}-")
    assert chunk.content_hash.endswith(f"-{ordinal}")
    assert chunk.rerank_score is None
