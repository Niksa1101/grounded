"""Lexical retrieval: full-text search over one index version (DB.md §6.2).

The whole query is one SQL statement, so the question is bound exactly once and never becomes SQL
text or tsquery syntax. Its steps:

1. ``to_tsvector('english', question)`` normalizes the question the same way ``chunks.tsv`` was
   built (DB.md §4): same stemmer, same stop words, so "running" meets "run".
2. ``tsvector_to_array`` turns that into plain lexemes. A lexeme can itself hold tsquery syntax
   (Postgres keeps ``example.com:8080`` and whole URLs as single lexemes), so each one is wrapped
   in single quotes with ``'`` and ``\\`` escaped: inside quotes nothing is an operator.
3. The quoted lexemes are joined with ``|`` and parsed by ``to_tsquery('simple', …)``.

**Why OR.** ``plainto_tsquery`` and ``websearch_to_tsquery`` are AND: a natural-language question
with eight content words practically never fits into one chunk. With OR any shared word matches and
``ts_rank_cd`` decides the order (more and closer matches, and breadcrumb weight A over body B).

**Why the ``simple`` config.** The lexemes are already stemmed by ``english``; running them through
``english`` again would stem twice ("comput" is not "compute"). ``simple`` only lowercases and
keeps every token.

An empty question, or one made only of stop words, has no lexemes: ``string_agg`` over zero rows is
NULL, ``to_tsquery`` of NULL is NULL, ``tsv @@ NULL`` is never true, and the result is empty without
a special case in Python.
"""

from __future__ import annotations

from typing import Any

from psycopg import AsyncConnection

from grounded.retrieval.types import RetrievedChunk

# ``query`` is a scalar sub-select, so the planner sees a single value and can use the GIN index on
# ``chunks.tsv`` (a join against the CTE would not). The chunk id breaks score ties: eval
# reproducibility depends on it (AGENTS.md §9). All values are bound parameters (AGENTS.md §6.11).
_LEXICAL = r"""
WITH lexemes AS (
    SELECT unnest(tsvector_to_array(to_tsvector('english', %(question)s))) AS lexeme
),
query AS (
    SELECT to_tsquery(
               'simple',
               string_agg('''' || replace(replace(lexeme, '\', '\\'), '''', '''''') || '''', ' | ')
           ) AS q
    FROM lexemes
)
SELECT id, section_id, anchor_path, breadcrumb_text, url, content, token_count, content_hash,
       ts_rank_cd(tsv, (SELECT q FROM query)) AS score
FROM chunks
WHERE index_version_id = %(index_version_id)s
  AND tsv @@ (SELECT q FROM query)
ORDER BY score DESC, id
LIMIT %(k)s
"""


async def lexical_search(
    conn: AsyncConnection[Any],
    question: str,
    *,
    index_version_id: int,
    k: int,
) -> list[RetrievedChunk]:
    """The (up to) ``k`` chunks of ``index_version_id`` that best match ``question``, best first.

    A chunk matches when it contains *any* content word of the question (OR semantics). The question
    is data, never query syntax: any text is safe (a NUL byte counts as a space). A question with
    nothing to search for (empty, only stop words or punctuation) returns ``[]``, like one that no
    chunk matches.

    Order is ``ts_rank_cd`` descending, ties by chunk id ascending. ``fts_rank`` is 1-based and
    dense; ``fts_score`` is the ``ts_rank_cd`` value, only comparable within this one query. The
    dense and later signals stay ``None``. ``k < 1`` raises ``ValueError``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # Postgres text cannot hold NUL and psycopg raises on it; a JSON body can carry one. It splits
    # words like whitespace does.
    params = {
        "question": question.replace("\x00", " "),
        "index_version_id": index_version_id,
        "k": k,
    }
    cur = await conn.execute(_LEXICAL, params)
    rows: list[tuple[Any, ...]] = await cur.fetchall()
    return [
        RetrievedChunk(
            chunk_id=int(row[0]),
            section_id=str(row[1]),
            anchor_path=tuple(row[2]),
            breadcrumb_text=str(row[3]),
            url=str(row[4]),
            content=str(row[5]),
            token_count=int(row[6]),
            content_hash=str(row[7]),
            fts_rank=rank,
            fts_score=float(row[8]),
        )
        for rank, row in enumerate(rows, start=1)
    ]
