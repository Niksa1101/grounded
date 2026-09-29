"""Retrieval eval runner (Tech.md §15.2): golden set → retrieval → metrics → results file.

For each retrieval mode, every answerable question is embedded (``RETRIEVAL_QUERY``, through the
embedding cache, so a re-run makes no API calls), retrieved against the active index, and scored
with ``evals/metrics.py``. Unanswerable questions are skipped and counted, never scored 0. Means
are reported with ``n``. No LLM calls.

The results file holds per-question rows (ranks, retrieved sections) so two runs can be diffed.
A baseline row is only ever copied from a run (AGENTS.md §7), never typed.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import psycopg
from pgvector.psycopg import register_vector_async  # pyright: ignore[reportMissingTypeStubs]
from pydantic import TypeAdapter

from grounded.evals.golden import GOLDEN_DIR
from grounded.evals.metrics import ChunkRef, label_ranks, mrr, ndcg_at_k, recall_at_k
from grounded.ingest.embed import Embedder, Vector
from grounded.retrieval.dense import dense_search
from grounded.retrieval.index import active_index_version
from grounded.schemas.eval import (
    GoldenItem,
    RetrievalBaselineEntry,
    RetrievalConfigResult,
    RetrievalQuestionResult,
    RetrievalRun,
    RetrievalRunInfo,
)

EVAL_DIR: Final = GOLDEN_DIR.parent
RESULTS_DIR: Final = EVAL_DIR / "results"
RETRIEVAL_BASELINE: Final = EVAL_DIR / "baselines" / "retrieval.json"

# The cutoffs the metrics are reported at. They are part of the metric definitions (PRD §8 targets
# Recall@5 and nDCG@5; @10 shows what a bigger context would add), not a retrieval setting.
METRIC_CUTOFFS: Final = (5, 10)

RetrievalMode = Literal["dense"]  # Phase 2 adds "fts" and "hybrid"

# One question → its ranked chunks. Gets the question text and its query vector; a mode uses
# whichever it needs (dense the vector, FTS the text, hybrid both).
type Search = Callable[[str, Vector], Awaitable[Sequence[ChunkRef]]]

_BASELINE_FILE: Final = TypeAdapter(dict[str, RetrievalBaselineEntry])
_VERSIONED_NAME: Final = re.compile(r"^golden_set\.(v\d+)\.jsonl$")


class RetrievalEvalError(Exception):
    """The eval can't run as asked (wrong index, bad golden-set file name, ...)."""


def metric_names() -> list[str]:
    """Metric keys in report order: recall@5, recall@10, mrr, ndcg@5, ndcg@10."""
    return [f"recall@{k}" for k in METRIC_CUTOFFS] + ["mrr"] + [f"ndcg@{k}" for k in METRIC_CUTOFFS]


def score_question(item: GoldenItem, retrieved: Sequence[ChunkRef]) -> RetrievalQuestionResult:
    """Metrics for one answerable question, from its ranked chunks."""
    relevant = item.relevant
    ranks = label_ranks(retrieved, relevant)
    metrics = {f"recall@{k}": recall_at_k(ranks, relevant, k) for k in METRIC_CUTOFFS}
    metrics["mrr"] = mrr(ranks, relevant)
    metrics |= {f"ndcg@{k}": ndcg_at_k(ranks, relevant, k) for k in METRIC_CUTOFFS}
    return RetrievalQuestionResult(
        id=item.id,
        type=item.type,
        metrics=metrics,
        ranks={label: ranks.get(label) for label in relevant},
        retrieved=[chunk.section_id for chunk in retrieved],
    )


async def evaluate(
    items: Sequence[GoldenItem],
    *,
    config: str,
    k: int,
    embedder: Embedder,
    search: Search,
) -> RetrievalConfigResult:
    """Score every answerable item with ``search`` and average the metrics over them.

    All questions are embedded in one call (one batch, and one cache lookup), then retrieved one
    by one in golden-set order.
    """
    scored = [item for item in items if item.answerable]
    if not scored:
        raise RetrievalEvalError("the golden set has no answerable questions to score")
    vectors = await embedder.embed([item.question for item in scored], "RETRIEVAL_QUERY")
    questions = [
        score_question(item, await search(item.question, vector))
        for item, vector in zip(scored, vectors, strict=True)
    ]
    return RetrievalConfigResult(
        config=config,
        k=k,
        n=len(questions),
        skipped_unanswerable=len(items) - len(scored),
        metrics=_means(questions),
        questions=questions,
    )


def _means(questions: Sequence[RetrievalQuestionResult]) -> dict[str, float]:
    # fsum is exactly rounded, so the mean doesn't depend on the order of the questions.
    return {
        name: math.fsum(q.metrics[name] for q in questions) / len(questions)
        for name in metric_names()
    }


async def run_retrieval_eval(
    conninfo: str,
    items: Sequence[GoldenItem],
    *,
    modes: Sequence[RetrievalMode],
    embedder: Embedder,
    k_dense: int,
    golden_set_version: str,
    now: datetime,
    repo: tuple[str | None, bool | None] = (None, None),
) -> RetrievalRun:
    """Run each mode against the active index of ``conninfo``.

    The embedder must be the one that built the index: vectors of another model (or dimension)
    would be compared with the index's, which is meaningless, so that is an error, not a warning.
    ``repo`` is ``(git_sha, git_dirty)`` of the code under test (``repo_state``).
    """
    async with await psycopg.AsyncConnection.connect(conninfo) as conn:
        await register_vector_async(conn)
        index = await active_index_version(conn)
        if (embedder.model, embedder.dim) != (index.embedding_model, index.embedding_dim):
            raise RetrievalEvalError(
                f"index {index.label} was built with {index.embedding_model} "
                f"({index.embedding_dim} dims), but queries would be embedded with "
                f"{embedder.model} ({embedder.dim} dims)"
            )

        async def dense(question: str, vector: Vector) -> Sequence[ChunkRef]:
            return await dense_search(conn, vector, index_version_id=index.id, k=k_dense)

        searches: dict[RetrievalMode, tuple[Search, int]] = {"dense": (dense, k_dense)}
        configs: dict[str, RetrievalConfigResult] = {}
        for mode in dict.fromkeys(modes):  # each mode once, in the order given
            search, k = searches[mode]
            configs[mode] = await evaluate(
                items, config=mode, k=k, embedder=embedder, search=search
            )

    git_sha, git_dirty = repo
    info = RetrievalRunInfo(
        date=now,
        git_sha=git_sha,
        git_dirty=git_dirty,
        golden_set_version=golden_set_version,
        index_version_id=index.id,
        index_config_hash=index.config_hash,
        fastapi_ref=index.git_ref,
        fastapi_sha=index.git_sha,
        embedding_model=index.embedding_model,
        embedding_dim=index.embedding_dim,
    )
    return RetrievalRun(info=info, configs=configs)


# --- Files ----------------------------------------------------------------------------------------


def golden_set_version(path: Path) -> str:
    """``"v1"`` for ``golden_set.v1.jsonl``. Results and baselines must say which version they
    scored, so an unversioned file name is refused."""
    match = _VERSIONED_NAME.match(path.name)
    if match is None:
        raise RetrievalEvalError(
            f"{path.name}: the golden-set file must be named golden_set.v<N>.jsonl"
        )
    return match.group(1)


def results_path(now: datetime) -> Path:
    """``eval/results/<UTC timestamp>-retrieval.json``."""
    return RESULTS_DIR / f"{now.strftime('%Y%m%dT%H%M%SZ')}-retrieval.json"


def write_run(path: Path, run: RetrievalRun) -> None:
    _write_json(path, run.model_dump(mode="json"))


def read_run(path: Path) -> RetrievalRun:
    return RetrievalRun.model_validate_json(path.read_bytes())


def update_baseline(path: Path, run: RetrievalRun) -> dict[str, RetrievalBaselineEntry]:
    """Replace the rows of the configs in ``run`` in the baseline file (created if missing) and
    keep the other rows, so ablation configs can be added one run at a time."""
    rows = _BASELINE_FILE.validate_json(path.read_bytes()) if path.exists() else {}
    for name, result in run.configs.items():
        rows[name] = RetrievalBaselineEntry(
            metrics=result.metrics,
            n=result.n,
            k=result.k,
            golden_set_version=run.info.golden_set_version,
            index_config_hash=run.info.index_config_hash,
            fastapi_ref=run.info.fastapi_ref,
            fastapi_sha=run.info.fastapi_sha,
            embedding_model=run.info.embedding_model,
            embedding_dim=run.info.embedding_dim,
            git_sha=run.info.git_sha,
            date=run.info.date,
        )
    _write_json(path, _BASELINE_FILE.dump_python(rows, mode="json"))
    return rows


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bytes, not text: LF line endings on every OS, like the rest of the committed files.
    path.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def repo_state(root: Path = EVAL_DIR.parent) -> tuple[str | None, bool | None]:
    """``(HEAD sha, tracked files changed)``, or ``(None, None)`` outside a git checkout."""
    try:
        sha = _git(root, "rev-parse", "HEAD")
        # Untracked files (scratch scripts, notes) don't change the code under test.
        dirty = _git(root, "status", "--porcelain", "--untracked-files=no") != ""
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, dirty


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return done.stdout.strip()
