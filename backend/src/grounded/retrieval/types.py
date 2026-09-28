"""Value objects returned by retrieval (Tech.md §7).

Frozen and slotted, like the ingest value objects: evals depend on a retrieved list being exactly
what the query returned.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class IndexVersion:
    """The index version a query runs against (the active one), and what built it."""

    id: int
    git_ref: str
    git_sha: str
    embedding_model: str
    embedding_dim: int
    config_hash: str

    @property
    def label(self) -> str:
        """``"<git_ref>@<config_hash[:8]>"``, the ``meta.index_version`` shown per answer."""
        return f"{self.git_ref}@{self.config_hash[:8]}"


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One ranked chunk with the fields later stages need (context building, citations, metrics).

    Each retrieval mode fills the signals it has and leaves the others ``None``: dense sets
    ``dense_rank``/``dense_distance``; FTS, RRF and rerank arrive in Phases 2 and 6. The
    confidence heuristic (Phase 3) reads these signals, so they are kept, not recomputed.
    """

    chunk_id: int
    section_id: str  # "<source_path>#<deepest anchor>" ("...md#" for a page intro)
    anchor_path: tuple[str, ...]  # outermost first; () for a page intro
    breadcrumb_text: str
    url: str
    content: str
    token_count: int
    content_hash: str
    dense_rank: int | None = None  # 1-based
    dense_distance: float | None = None  # cosine distance, 0 = same direction
    fts_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None
