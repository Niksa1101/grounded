"""Hybrid retrieval: dense and lexical lists fused with Reciprocal Rank Fusion (DB.md §6.3).

The query is one SQL statement. ``dense`` and ``lexical`` each rank the chunks of one index
version and are cut to their own K; ``fused`` full-outer-joins them on chunk id and adds
``1 / (rrf_k + rank)`` per list a chunk is in; the final select cuts to ``k_fused`` and attaches the
chunk fields. The question becomes a tsquery through the CTEs ``lexical_search`` uses
(``LEXICAL_QUERY_CTES``), so both modes read it the same way.

RRF fuses *ranks*, not scores: cosine distance and ``ts_rank_cd`` live on unrelated scales, so
there is nothing to normalize and no weight to tune, only ``rrf_k`` (how fast the credit decays
with rank).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pgvector import Vector
from psycopg import AsyncConnection

from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.lexical import LEXICAL_QUERY_CTES
from grounded.retrieval.types import RetrievedChunk

# - ``row_number()`` gives the dense 1-based rank; the chunk id is in both ORDER BYs, so ranks (and
#   the cut at K) never depend on physical row order.
# - ``rrf_k`` is cast to float8 so the division is floating point (``1 / 61`` in integers is 0).
# - ``USING (id)`` merges the join keys, so ``id`` is the chunk from either side. A rank missing on
#   one side is NULL there, and COALESCE turns its credit into 0.
# - The tie-break ``rrf_score DESC, id`` is applied before ``LIMIT %(k_fused)s`` and once more
#   after the join to ``chunks``, because a join does not promise to keep the order.
# - Every value, the vector included, is a bound parameter (AGENTS.md §6.11).
_HYBRID = (
    "WITH"
    + LEXICAL_QUERY_CTES
    + r""",
dense AS (
    SELECT id,
           embedding <=> %(query)s AS distance,
           row_number() OVER (ORDER BY embedding <=> %(query)s, id) AS rank
    FROM chunks
    WHERE index_version_id = %(index_version_id)s
    ORDER BY rank
    LIMIT %(k_dense)s
),
lexical AS (
    SELECT id,
           ts_rank_cd(tsv, (SELECT q FROM query)) AS score,
           row_number() OVER (ORDER BY ts_rank_cd(tsv, (SELECT q FROM query)) DESC, id) AS rank
    FROM chunks
    WHERE index_version_id = %(index_version_id)s
      AND tsv @@ (SELECT q FROM query)
    ORDER BY rank
    LIMIT %(k_fts)s
),
fused AS (
    SELECT id,
           dense.rank AS dense_rank,
           dense.distance AS dense_distance,
           lexical.rank AS fts_rank,
           lexical.score AS fts_score,
           COALESCE(1.0::float8 / (%(rrf_k)s::float8 + lexical.rank), 0) AS rrf_score
    FROM dense
    FULL OUTER JOIN lexical USING (id)
    ORDER BY rrf_score DESC, id
    LIMIT %(k_fused)s
)
SELECT c.id, c.section_id, c.anchor_path, c.breadcrumb_text, c.url, c.content, c.token_count,
       c.content_hash, f.dense_rank, f.dense_distance, f.fts_rank, f.fts_score, f.rrf_score
FROM fused AS f
JOIN chunks AS c ON c.id = f.id
ORDER BY f.rrf_score DESC, c.id
"""
)


async def hybrid_search(
    conn: AsyncConnection[Any],
    question: str,
    query_vector: Sequence[float],
    *,
    index_version_id: int,
    cfg: RetrievalConfig,
) -> list[RetrievedChunk]:
    """The (up to) ``cfg.k_fused`` chunks of ``index_version_id`` that best match ``question``
    by dense *and* lexical evidence, best first.

    Contract (the spec tests pin each point):

    - **Two ranked lists.** The dense list is the ``cfg.k_dense`` chunks nearest to
      ``query_vector`` (cosine distance, as in ``dense_search``). The lexical list is the
      ``cfg.k_fts`` best matches of ``question`` (OR semantics and ``ts_rank_cd`` order, as in
      ``lexical_search``). Each list is **cut to its own K before fusion**: a chunk outside a
      list's cut has no rank there and gets nothing from it, even if it would have ranked in a
      longer list.
    - **Reciprocal Rank Fusion.** A chunk's ``rrf_score`` is the sum, over the lists it is in, of
      ``1 / (cfg.rrf_k + rank)`` with a 1-based rank within that list. A missing rank contributes
      0. The chunks of the two lists are joined on chunk id with a full outer join, so a chunk
      found by one list only still appears (``rrf_score = 1 / (rrf_k + rank)``).
    - **Output.** At most ``cfg.k_fused`` *unique* chunks, ordered by ``rrf_score`` descending,
      ties broken by chunk id ascending (eval reproducibility, AGENTS.md §9). The tie-break
      applies before the cut to ``k_fused``, so the same chunks survive every time.
    - **Signals.** ``rrf_score`` is always set. ``dense_rank`` and ``dense_distance`` are set when
      the chunk is in the dense list, ``fts_rank`` and ``fts_score`` (the ``ts_rank_cd`` value,
      same as ``lexical_search``) when it is in the lexical list, and are ``None`` otherwise.
      Ranks belong to their own list, not to the fused order (that is the position in the result).
      ``rerank_score`` stays ``None``. The chunk fields are those of ``dense_search``.
    - **No lexemes is not an error.** A question with nothing to search for (only stop words,
      empty, punctuation) or that no chunk matches leaves the lexical list empty: the result is
      the dense ranking (RRF over one list) and ``fts_rank`` is ``None`` everywhere.
    - **The question is data, never query syntax**, exactly as in ``lexical_search`` (a NUL byte
      counts as a space). Only chunks of ``index_version_id`` are considered. An index version
      without chunks gives ``[]``.
    - **One SQL statement** with the CTEs ``dense``, ``lexical`` and ``fused`` (DB.md §6.3). Every
      value is a bound parameter, the vector as a pgvector ``Vector`` (AGENTS.md §6.11).
    - Only ``cfg.k_dense``, ``cfg.k_fts``, ``cfg.k_fused`` and ``cfg.rrf_k`` are read: the mode and
      the rerank fields are the caller's business. ``rrf_k`` is config, never a literal.

    ``conn`` is an async psycopg connection with the pgvector adapters registered
    (``register_vector_async``), like for ``dense_search``. The vector's dimension must match the
    index's, or Postgres raises.
    """
    params = {
        # A NUL cannot live in Postgres text; it splits words like whitespace (see lexical_search).
        "question": question.replace("\x00", " "),
        "query": Vector(list(query_vector)),
        "index_version_id": index_version_id,
        "k_dense": cfg.k_dense,
        "k_fts": cfg.k_fts,
        "k_fused": cfg.k_fused,
        "rrf_k": cfg.rrf_k,
    }
    cur = await conn.execute(_HYBRID, params)
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
            dense_rank=None if row[8] is None else int(row[8]),
            dense_distance=None if row[9] is None else float(row[9]),
            fts_rank=None if row[10] is None else int(row[10]),
            fts_score=None if row[11] is None else float(row[11]),
            rrf_score=float(row[12]),
        )
        for row in rows
    ]
