"""``RetrievalConfig``: every setting that changes which chunks a query returns (Tech.md §4, §7).

One frozen value object with a stable hash, so a result can always say which retrieval produced it:
eval results and baseline rows now, request logs, ``eval_runs`` and the answer-cache key later
(``retrieval_config_hash`` in DB.md §4). Like ``ChunkingConfig`` it is built from ``Settings``,
which stays the only reader of the environment (AGENTS.md §6.7).

The hash covers every field, whatever the mode uses: a K a mode ignores still changes it. That keeps
the hash a plain function of the value; the cost is that tuning ``K_FTS`` also changes the ``dense``
hash. The rerank fields are part of it from the start, so turning rerank on in Phase 6 changes the
hash without changing its format.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from grounded.settings import RerankProvider

if TYPE_CHECKING:
    from grounded.settings import Settings

# Phase 6 adds "hybrid_rerank".
RetrievalMode = Literal["dense", "fts", "hybrid"]


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: RetrievalMode
    k_dense: int = Field(gt=0)
    k_fts: int = Field(gt=0)
    k_fused: int = Field(gt=0)
    k_context: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    rerank_provider: RerankProvider = "none"
    rerank_model: str | None = None

    @model_validator(mode="after")
    def _check_rerank(self) -> Self:
        # "No rerank" has one spelling, so it has one hash.
        if self.rerank_provider == "none" and self.rerank_model is not None:
            raise ValueError("rerank_model needs a rerank_provider")
        return self

    @classmethod
    def from_settings(cls, settings: Settings, mode: RetrievalMode) -> RetrievalConfig:
        rerank_on = settings.rerank_provider != "none"
        return cls(
            mode=mode,
            k_dense=settings.k_dense,
            k_fts=settings.k_fts,
            k_fused=settings.k_fused,
            k_context=settings.k_context,
            rrf_k=settings.rrf_k,
            rerank_provider=settings.rerank_provider,
            # A model named while rerank is off is not applied, so it must not change the hash.
            rerank_model=settings.rerank_model if rerank_on else None,
        )

    def canonical_json(self) -> str:
        """Key-sorted, whitespace-free JSON: the same config always hashes the same."""
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        """sha256 hex of ``canonical_json()``: the ``retrieval_config_hash`` that gets recorded."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
