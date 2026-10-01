"""Hybrid retrieval: dense and lexical lists fused with Reciprocal Rank Fusion (DB.md §6.3).

**Author-owned** (AGENTS.md §3): the query is written in ticket 2.06, this file only fixes the
contract. The spec tests are in ``tests/integration/test_hybrid.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from psycopg import AsyncConnection

from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.types import RetrievedChunk


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
    raise NotImplementedError("Author implements the hybrid query in ticket 2.06")
