"""Spec for the lexical query (DB.md §6.2), written before it was implemented (tickets 2.02, 2.03).

The corpus is hand-made (``tests/retrieval_corpus.py``): each group of chunks shares a rare word, so
the expected matches and the expected order are known without running any query. Nothing here
recomputes a rank; the tests pin the observable contract: which chunks come back, in which order,
with which fields, and that the question is never interpreted as query syntax.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from itertools import pairwise
from typing import Any, NamedTuple

import psycopg
import pytest

from grounded.retrieval.lexical import lexical_search
from grounded.retrieval.types import RetrievedChunk
from tests.retrieval_corpus import (
    CAPYBARA_COUNT,
    CAPYBARA_KEYS,
    CORPUS,
    PENGUINS,
    TIE_KEYS,
    ZOO,
    build_text_index,
)
from tests.support import insert_chunk, insert_document, insert_index_version

pytestmark = [pytest.mark.integration]

QUOKKA_KEYS = ["or_one", "or_two", "or_three"]


class Index(NamedTuple):
    active: int
    other: int  # an inactive version whose only chunk says "quokka" four times
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
        ids = build_text_index(conn, active)
        other = insert_index_version(conn, active=False, config_hash="2" * 64)
        source_path, url, title = ZOO
        document = insert_document(conn, other, source_path=source_path, url=url, title=title)
        other_chunk = insert_chunk(
            conn,
            other,
            document,
            ordinal=0,
            breadcrumb=(title, "Quokkas"),
            anchors=("quokkas",),
            content="quokka quokka quokka quokka",
        )
    yield Index(active, other, ids, other_chunk)
    wipe()


@pytest.fixture
async def aconn(test_database_url: str) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    async with await psycopg.AsyncConnection.connect(test_database_url) as conn:
        yield conn


async def search(
    aconn: psycopg.AsyncConnection[Any], index: Index, question: str, *, k: int = 50
) -> list[RetrievedChunk]:
    return await lexical_search(aconn, question, index_version_id=index.active, k=k)


def keys(index: Index, chunks: list[RetrievedChunk]) -> list[str]:
    """Corpus keys of the result, in order (an unknown id shows up as ``?<id>``)."""
    by_id = {chunk_id: key for key, chunk_id in index.ids.items()}
    return [by_id.get(c.chunk_id, f"?{c.chunk_id}") for c in chunks]


def scores(chunks: list[RetrievedChunk]) -> list[float]:
    values = [c.fts_score for c in chunks]
    assert all(v is not None for v in values)
    return [v for v in values if v is not None]


# --- Matching ------------------------------------------------------------------------------------


async def test_or_semantics_matches_a_chunk_with_only_some_of_the_words(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """DB.md §6.2 pitfall: AND semantics would return nothing for this question, because no chunk
    holds all of quokka, narwhal, pangolin and compare."""
    question = "How does the quokka compare with the narwhal and the pangolin?"
    chunks = await search(aconn, index, question)
    assert sorted(keys(index, chunks)) == sorted(QUOKKA_KEYS)


async def test_stop_words_in_the_question_are_not_searched(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    chunks = await search(aconn, index, "how do I do it with the quokka")
    assert sorted(keys(index, chunks)) == sorted(QUOKKA_KEYS)


@pytest.mark.parametrize(
    "question",
    ["how do I do it", "the a an of to", "", "   ", "?! ... --"],
    ids=["stop-words", "more-stop-words", "empty", "blank", "punctuation"],
)
async def test_nothing_to_search_for_returns_an_empty_list(
    index: Index, aconn: psycopg.AsyncConnection[Any], question: str
) -> None:
    assert await search(aconn, index, question) == []


async def test_a_question_no_chunk_matches_returns_an_empty_list(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    assert await search(aconn, index, "zeppelin") == []


@pytest.mark.parametrize("question", ["running", "run", "runs", "RUNNING"])
async def test_stemming_matches_other_forms_of_the_word(
    index: Index, aconn: psycopg.AsyncConnection[Any], question: str
) -> None:
    """The index has "runs"; the question forms differ, and so does their case."""
    chunks = await search(aconn, index, question)
    assert keys(index, chunks) == ["stem_runs"]


async def test_matching_ignores_case(index: Index, aconn: psycopg.AsyncConnection[Any]) -> None:
    lower = await search(aconn, index, "quokka narwhal")
    upper = await search(aconn, index, "QUOKKA Narwhal")
    assert [c.chunk_id for c in upper] == [c.chunk_id for c in lower] != []


# --- Ranking -------------------------------------------------------------------------------------


async def test_a_term_in_the_breadcrumb_outranks_the_same_term_in_the_body(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """Weight A (breadcrumb) vs B (content). The body chunk has the lower id, so a query that
    ignores the weights would put it first on the tie-break alone."""
    assert index.ids["weight_body"] < index.ids["weight_title"]
    chunks = await search(aconn, index, "wombat")
    assert keys(index, chunks) == ["weight_title", "weight_body"]
    assert scores(chunks)[0] > scores(chunks)[1]


async def test_more_occurrences_rank_higher_and_ties_break_by_chunk_id(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """``tie_strong`` has the highest id but the most matches; the identical ``tie_*`` chunks
    score the same, so only the id can order them."""
    chunks = await search(aconn, index, "ocelot")
    assert keys(index, chunks) == ["tie_strong", *TIE_KEYS]
    tied = scores(chunks)[1:]
    assert tied[0] == tied[1] == tied[2]
    assert scores(chunks)[0] > tied[0]


async def test_ranks_are_dense_and_scores_never_increase(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    chunks = await search(aconn, index, "quokka narwhal pangolin wombat ocelot capybara")
    assert len(chunks) == 3 + 2 + 4 + CAPYBARA_COUNT
    assert [c.fts_rank for c in chunks] == list(range(1, len(chunks) + 1))
    values = scores(chunks)
    assert all(v > 0 for v in values)
    assert values == sorted(values, reverse=True)
    for before, after in pairwise(chunks):
        if before.fts_score == after.fts_score:
            assert before.chunk_id < after.chunk_id


async def test_the_result_is_reproducible(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    question = "quokka narwhal pangolin wombat ocelot capybara"
    first = await search(aconn, index, question)
    second = await search(aconn, index, question)
    assert first == second


# --- Filtering and k -----------------------------------------------------------------------------


async def test_searches_only_the_given_index_version(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """The other version's chunk says "quokka" four times, so it would win if it leaked in."""
    in_active = await search(aconn, index, "quokka")
    assert sorted(keys(index, in_active)) == sorted(QUOKKA_KEYS)

    in_other = await lexical_search(aconn, "quokka", index_version_id=index.other, k=50)
    assert [c.chunk_id for c in in_other] == [index.other_chunk]


async def test_k_limits_the_result_and_keeps_the_best_rows(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """The twelve capybara chunks are identical, so the ones that survive k are the lowest ids."""
    top = await search(aconn, index, "capybara", k=5)
    assert keys(index, top) == list(CAPYBARA_KEYS[:5])
    assert [c.fts_rank for c in top] == [1, 2, 3, 4, 5]

    everything = await search(aconn, index, "capybara", k=100)  # k beyond the matches is fine
    assert keys(index, everything) == list(CAPYBARA_KEYS)


@pytest.mark.parametrize("k", [0, -1])
async def test_k_must_be_positive(aconn: psycopg.AsyncConnection[Any], k: int) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        await lexical_search(aconn, "quokka", index_version_id=1, k=k)


# --- Hostile input -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "'",
        "''''",
        "'; DROP TABLE chunks; --",
        "a & | b",
        "!!!",
        "((((",
        "))))",
        "\\",
        "quokka:*",
        "'quokka'",
        "quokka <-> narwhal",
        "quokka & !narwhal",
        "a:b:c",
        # Words whose lexeme itself holds query syntax (Postgres keeps URLs and host:port whole).
        "example.com:8080/path?x=1&y=2",
        "http://x.com/a:b&c=d(e)",
        "http://x.com/it's/a(b)!c|d\\e",
        "日本語のテスト",
        "🦘 quokka 🦘",
        "x" * 5000,
    ],
    ids=lambda q: q[:24],
)
async def test_special_characters_never_raise(
    index: Index, aconn: psycopg.AsyncConnection[Any], question: str
) -> None:
    chunks = await search(aconn, index, question)
    assert isinstance(chunks, list)
    assert all(isinstance(c, RetrievedChunk) for c in chunks)
    # The connection is still usable: a failed statement would have aborted the transaction.
    assert await search(aconn, index, "wombat") != []


async def test_operators_in_the_question_are_words_not_syntax(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """``!`` would exclude, ``&`` would require, ``|`` would group. None of that may happen: the
    result equals the one for the same words without any punctuation, in the same order."""
    plain = await search(aconn, index, "quokka narwhal pangolin")
    hostile = await search(aconn, index, "quokka:* & !(narwhal | 'pangolin') \\")
    assert sorted(keys(index, plain)) == sorted(QUOKKA_KEYS)
    assert [(c.chunk_id, c.fts_score) for c in hostile] == [
        (c.chunk_id, c.fts_score) for c in plain
    ]


async def test_a_word_that_contains_special_characters_still_matches(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    """Postgres keeps ``example.com:8080`` as one lexeme, colon included, and keeps a whole URL
    with ``?`` and ``&`` as another. A query built by joining them with ``|`` breaks here."""
    chunks = await search(aconn, index, "why does example.com:8080/path?x=1&y=2 fail")
    assert "host_port" in keys(index, chunks)


async def test_non_ascii_words_match(index: Index, aconn: psycopg.AsyncConnection[Any]) -> None:
    chunks = await search(aconn, index, "où est le café ?")
    assert keys(index, chunks) == ["accent"]


# --- Result mapping ------------------------------------------------------------------------------


async def test_a_page_intro_is_returned_with_its_fields(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    [top] = await search(aconn, index, "emperor")
    source_path, url, _title = PENGUINS
    assert top.chunk_id == index.ids["intro"]
    assert top.anchor_path == ()
    assert top.section_id == f"{source_path}#"
    assert top.breadcrumb_text == "Penguins"
    assert top.url == url


async def test_chunk_fields_and_signals_are_mapped(
    index: Index, aconn: psycopg.AsyncConnection[Any]
) -> None:
    [top] = await search(aconn, index, "smiles")
    spec = next(s for s in CORPUS if s.key == "or_one")
    source_path, url, _title = ZOO
    assert top.chunk_id == index.ids["or_one"]
    assert top.section_id == f"{source_path}#marsupials"
    assert top.anchor_path == ("marsupials",)
    assert top.breadcrumb_text == "Zoo > Marsupials"
    assert top.url == f"{url}#marsupials"
    assert top.content == spec.content
    assert top.token_count == len(spec.content.split())
    assert top.content_hash.startswith(f"hash-{index.active}-")
    # Only the lexical signals are set.
    assert top.fts_rank == 1
    assert top.fts_score is not None
    assert top.fts_score > 0
    assert top.dense_rank is None
    assert top.dense_distance is None
    assert top.rrf_score is None
    assert top.rerank_score is None
