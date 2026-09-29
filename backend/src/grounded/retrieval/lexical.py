"""Lexical retrieval: full-text search over one index version (DB.md §6.2).

**Author-owned** (AGENTS.md §3): the query is written in ticket 2.03, this file only fixes the
contract. The spec tests are in ``tests/integration/test_lexical.py``.
"""

from __future__ import annotations

from typing import Any

from psycopg import AsyncConnection

from grounded.retrieval.types import RetrievedChunk


async def lexical_search(
    conn: AsyncConnection[Any],
    question: str,
    *,
    index_version_id: int,
    k: int,
) -> list[RetrievedChunk]:
    """The (up to) ``k`` chunks of ``index_version_id`` that best match ``question``, best first.

    Contract (the spec tests pin each point):

    - **OR semantics.** A chunk matches when it contains *any* content word of the question, not
      all of them. ``plainto_tsquery`` / ``websearch_to_tsquery`` are AND, so a long
      natural-language question would match almost nothing; DB.md §6.2 describes the approach that
      avoids it.
    - **The question is data, never query syntax.** Whatever it contains (quotes, ``:``, ``&``,
      ``|``, ``!``, parentheses, backslashes, non-ASCII, URLs) must not raise a syntax error and
      must not act as a tsquery operator. The same words give the same result with or without such
      punctuation around them.
    - **Nothing to search for is not an error.** A question with no lexemes after stop-word
      removal ("how do I do it", empty, only punctuation) returns ``[]``. So does a question that
      no chunk matches.
    - **Same language as the index.** The lexemes must line up with how ``chunks.tsv`` was built
      (``english`` config; breadcrumb weight A, content weight B, DB.md §4), so "running" matches
      "run" and a term in the breadcrumb outranks the same term only in the body.
    - **Order.** ``ts_rank_cd`` descending, ties broken by chunk id ascending (eval reproducibility,
      AGENTS.md §9). Only chunks of ``index_version_id`` are considered.
    - **Result.** ``fts_rank`` is 1-based and dense in the returned order; ``fts_score`` is the
      ``ts_rank_cd`` value (decided 2026-09-29: kept, next to the rank, for debugging and the
      confidence heuristic). The dense and later signals stay ``None``. Fewer than ``k`` rows if
      fewer chunks match.
    - ``k < 1`` raises ``ValueError`` (same as ``dense_search``). SQL parameters are always bound
      (AGENTS.md §6.11).

    ``conn`` is an async psycopg connection; no vector adapters are needed for this query.
    """
    raise NotImplementedError("Author implements the lexical query in ticket 2.03")
