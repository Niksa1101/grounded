"""Dense retrieval: exact cosine scan over one index version (DB.md §6.1).

No ANN index on purpose. With a few thousand rows an exact scan takes milliseconds and has 100%
recall, while an HNSW index combined with ``WHERE index_version_id = …`` filters *after* the
approximate search and can quietly return fewer than ``k`` rows.

Vectors must be L2-normalized (the embedder does it), so cosine distance ranks the same way as
inner product; ``<=>`` is pgvector's cosine distance, in [0, 2].
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pgvector import Vector
from psycopg import AsyncConnection

from grounded.retrieval.types import RetrievedChunk

# The chunk id breaks distance ties, so equal vectors always come back in the same order: eval
# reproducibility depends on it. The vector is a bound parameter like any other (AGENTS.md §6.11).
_DENSE = """
SELECT id, section_id, anchor_path, breadcrumb_text, url, content, token_count, content_hash,
       embedding <=> %(query)s AS distance
FROM chunks
WHERE index_version_id = %(index_version_id)s
ORDER BY distance, id
LIMIT %(k)s
"""


async def dense_search(
    conn: AsyncConnection[Any],
    query_vector: Sequence[float],
    *,
    index_version_id: int,
    k: int,
) -> list[RetrievedChunk]:
    """The ``k`` chunks of ``index_version_id`` closest to ``query_vector``, nearest first.

    ``conn`` needs the pgvector adapters registered (``register_vector_async``; the runtime pool
    does it per connection). The vector's dimension must match the index's, or Postgres raises.
    Fewer than ``k`` rows only if the index version has fewer chunks.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    params = {
        "query": Vector(list(query_vector)),
        "index_version_id": index_version_id,
        "k": k,
    }
    cur = await conn.execute(_DENSE, params)
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
            dense_rank=rank,
            dense_distance=float(row[8]),
        )
        for rank, row in enumerate(rows, start=1)
    ]
