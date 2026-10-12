"""Retrieval eval runner (Tech.md §15.2): golden set → retrieval → metrics → results file.

For each retrieval mode, every answerable question is embedded (``RETRIEVAL_QUERY``, through the
embedding cache, so a re-run makes no API calls), retrieved against the active index, and scored
with ``evals/metrics.py``. Unanswerable questions are skipped and counted, never scored 0. Means
are reported with ``n``. No LLM calls.

``hybrid_rerank`` (6.04, run by hand, never in CI) is the ``hybrid`` list reranked: the metrics are
computed over the reranker's *whole* reranked list, the same ``K_FUSED`` candidates as ``hybrid``,
so ``MRR`` and ``Recall@10`` compare like with like. A reranker error is not a result: the first
``ProviderError`` ends the run (``RerankRunError``) and nothing is scored on the RRF order instead.

The results file holds per-question rows (ranks, retrieved sections) so two runs can be diffed.
A baseline row is only ever copied from a run (AGENTS.md §7), never typed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import psycopg
from pgvector.psycopg import register_vector_async  # pyright: ignore[reportMissingTypeStubs]
from pydantic import TypeAdapter, ValidationError

from grounded.evals.golden import GOLDEN_DIR
from grounded.evals.metrics import ChunkRef, label_ranks, mrr, ndcg_at_k, recall_at_k
from grounded.infra.provider_errors import ProviderError
from grounded.ingest.embed import Embedder, Vector
from grounded.retrieval.config import RetrievalConfig, RetrievalMode
from grounded.retrieval.dense import dense_search
from grounded.retrieval.hybrid import hybrid_search
from grounded.retrieval.index import active_index_version
from grounded.retrieval.lexical import lexical_search
from grounded.retrieval.rerank import Reranker, rerank_error_code, rerank_usage
from grounded.schemas.eval import (
    GoldenItem,
    RerankRunStats,
    RetrievalBaselineEntry,
    RetrievalConfigResult,
    RetrievalQuestionResult,
    RetrievalRun,
    RetrievalRunInfo,
)

if TYPE_CHECKING:
    from grounded.settings import Settings

EVAL_DIR: Final = GOLDEN_DIR.parent
RESULTS_DIR: Final = EVAL_DIR / "results"
RETRIEVAL_BASELINE: Final = EVAL_DIR / "baselines" / "retrieval.json"

# The cutoffs the metrics are reported at. They are part of the metric definitions (PRD §8 targets
# Recall@5 and nDCG@5; @10 shows what a bigger context would add), not a retrieval setting.
METRIC_CUTOFFS: Final = (5, 10)

# One question → its ranked chunks. Gets the question text and its query vector; a mode uses
# whichever it needs (dense the vector, FTS the text, hybrid both).
type Search = Callable[[str, Vector], Awaitable[Sequence[ChunkRef]]]

_BASELINE_FILE: Final = TypeAdapter(dict[str, RetrievalBaselineEntry])
_VERSIONED_NAME: Final = re.compile(r"^golden_set\.(v\d+)\.jsonl$")


class RetrievalEvalError(Exception):
    """The eval can't run as asked (wrong index, bad golden-set file name, ...)."""


class RerankRunError(RetrievalEvalError):
    """The reranker failed on a question, so the run is not a result (Tech §8, §15.2).

    The question would have had to be scored on the RRF order instead, and a ``hybrid_rerank``
    number that is partly ``hybrid`` is worse than none. ``question_id`` is the golden id (never the
    text) and ``error`` the typed provider error, whose ``rerank_error_code`` is ``code``. The CLI
    ends the run on it without waiting out a ``Retry-After`` (AGENTS.md §6.15).
    """

    def __init__(self, question_id: str, error: ProviderError) -> None:
        self.question_id = question_id
        self.error = error
        self.code = rerank_error_code(error)
        text = " ".join(str(error).split())
        super().__init__(
            f"rerank failed at {question_id} ({self.code}, {type(error).__name__})"
            + (f": {text}" if text else "")
        )


class BaselineMismatchError(Exception):
    """The baseline file has rows from a different setup than this run; nothing was written."""


def metric_names() -> list[str]:
    """Metric keys in report order: recall@5, recall@10, mrr, ndcg@5, ndcg@10."""
    return [f"recall@{k}" for k in METRIC_CUTOFFS] + ["mrr"] + [f"ndcg@{k}" for k in METRIC_CUTOFFS]


def retrieval_configs(settings: Settings, modes: Sequence[RetrievalMode]) -> list[RetrievalConfig]:
    """The configs to score, one per mode, from the settings.

    A plain mode is scored as what it is, with no rerank fields, even when
    ``RERANK_PROVIDER=cohere`` is in ``.env`` for the ``hybrid_rerank`` run next to it: its
    ``retrieval_config_hash`` is then the one of the committed ``hybrid`` row, not a "hybrid with a
    reranker" it never used.
    ``hybrid_rerank`` keeps them, and without a provider and a model it is a ``ValueError``
    (``RetrievalConfig`` refuses it) instead of a ``hybrid`` run under another name.
    """
    configs: list[RetrievalConfig] = []
    for mode in modes:
        config = RetrievalConfig.from_settings(settings, mode)
        if mode != "hybrid_rerank" and config.rerank_provider != "none":
            config = config.model_copy(update={"rerank_provider": "none", "rerank_model": None})
        configs.append(config)
    return configs


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
    config: RetrievalConfig,
    k: int,
    embedder: Embedder,
    search: Search,
) -> RetrievalConfigResult:
    """Score every answerable item with ``search`` and average the metrics over them.

    ``config`` is the retrieval setup ``search`` runs; the result records it with its hash. All
    questions are embedded in one call (one batch, and one cache lookup), then retrieved one by
    one in golden-set order.
    """
    scored = [item for item in items if item.answerable]
    if not scored:
        raise RetrievalEvalError("the golden set has no answerable questions to score")
    vectors = await embedder.embed([item.question for item in scored], "RETRIEVAL_QUERY")
    questions: list[RetrievalQuestionResult] = []
    for item, vector in zip(scored, vectors, strict=True):
        try:
            found = await search(item.question, vector)
        except ProviderError as exc:
            # The questions were embedded above, so only the reranker can raise a provider error
            # here. Not skipped, not scored on the RRF order: the run ends (see RerankRunError).
            if config.mode != "hybrid_rerank":
                raise
            raise RerankRunError(item.id, exc) from exc
        questions.append(score_question(item, found))
    return RetrievalConfigResult(
        config=config.mode,
        retrieval_config=config,
        retrieval_config_hash=config.config_hash,
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


def _rerank_stats(config: RetrievalConfig, reranker: Reranker) -> RerankRunStats | None:
    """Live calls, cache hits and live latency of a run, read off the eval layers of ``reranker``
    (``rerank_usage``); ``None`` for a reranker that counts nothing (a bare fake)."""
    usage = rerank_usage(reranker)
    if usage is None or config.rerank_model is None:
        return None
    return RerankRunStats(
        model=config.rerank_model,
        live_calls=usage.live_calls,
        cache_hits=usage.cache_hits,
        latency_p50_ms=_percentile(usage.latencies_ms, 50),
        latency_p95_ms=_percentile(usage.latencies_ms, 95),
    )


def _percentile(values: Sequence[float], percent: int) -> float | None:
    """Nearest-rank percentile (the rule of the generation gate, Tech §15.5), in integers so no
    float error moves the rank; ``None`` for no values."""
    if not values:
        return None
    rank = -(-percent * len(values) // 100)  # ceil(percent / 100 * n)
    return sorted(values)[max(rank, 1) - 1]


async def run_retrieval_eval(
    conninfo: str,
    items: Sequence[GoldenItem],
    *,
    configs: Sequence[RetrievalConfig],
    embedder: Embedder,
    golden_set_version: str,
    golden_set_sha256: str,
    now: datetime,
    repo: tuple[str | None, bool | None] = (None, None),
    reranker: Reranker | None = None,
) -> RetrievalRun:
    """Run each config's mode against the active index of ``conninfo``.

    The embedder must be the one that built the index: vectors of another model (or dimension)
    would be compared with the index's, which is meaningless, so that is an error, not a warning.
    ``repo`` is ``(git_sha, git_dirty)`` of the code under test (``repo_state``). ``reranker`` is
    what a ``hybrid_rerank`` config reranks with (``runtime.build_reranker``; a fake in tests); a
    run that names that mode without one fails here, before any embedding or database work, and
    never falls back to ``hybrid``.
    """
    if reranker is None and any(config.mode == "hybrid_rerank" for config in configs):
        raise RetrievalEvalError(
            "hybrid_rerank needs a reranker: set RERANK_PROVIDER=cohere (with COHERE_API_KEY and "
            "RERANK_MODEL); it never falls back to hybrid"
        )
    async with await psycopg.AsyncConnection.connect(conninfo) as conn:
        await register_vector_async(conn)
        index = await active_index_version(conn)
        if (embedder.model, embedder.dim) != (index.embedding_model, index.embedding_dim):
            raise RetrievalEvalError(
                f"index {index.label} was built with {index.embedding_model} "
                f"({index.embedding_dim} dims), but queries would be embedded with "
                f"{embedder.model} ({embedder.dim} dims)"
            )

        def build_search(config: RetrievalConfig) -> tuple[Search, int]:
            """The search for ``config.mode`` and how many chunks it retrieves."""
            if config.mode == "dense":

                async def dense(question: str, vector: Vector) -> Sequence[ChunkRef]:
                    return await dense_search(
                        conn, vector, index_version_id=index.id, k=config.k_dense
                    )

                return dense, config.k_dense
            if config.mode == "fts":

                async def fts(question: str, vector: Vector) -> Sequence[ChunkRef]:
                    return await lexical_search(
                        conn, question, index_version_id=index.id, k=config.k_fts
                    )

                return fts, config.k_fts
            if config.mode == "hybrid":

                async def hybrid(question: str, vector: Vector) -> Sequence[ChunkRef]:
                    return await hybrid_search(
                        conn, question, vector, index_version_id=index.id, cfg=config
                    )

                # MRR runs over the fused list, so k is its length, not either input list's.
                return hybrid, config.k_fused
            if config.mode == "hybrid_rerank" and reranker is not None:
                ranker = reranker

                async def hybrid_rerank(question: str, vector: Vector) -> Sequence[ChunkRef]:
                    candidates = await hybrid_search(
                        conn, question, vector, index_version_id=index.id, cfg=config
                    )
                    if not candidates:  # nothing to rank: no call (the request path does the same)
                        return candidates
                    # The whole reranked list, not K_CONTEXT: MRR and Recall@10 then run over the
                    # same K_FUSED candidates as `hybrid`, only in the reranker's order.
                    return await ranker.rerank(question, candidates, top_n=len(candidates))

                return hybrid_rerank, config.k_fused
            raise RetrievalEvalError(f"retrieval mode {config.mode!r} is not implemented yet")

        results: dict[str, RetrievalConfigResult] = {}
        for config in configs:
            if config.mode in results:  # each mode once, in the order given
                continue
            search, k = build_search(config)
            result = await evaluate(items, config=config, k=k, embedder=embedder, search=search)
            if config.mode == "hybrid_rerank" and reranker is not None:
                result = result.model_copy(update={"rerank": _rerank_stats(config, reranker)})
            results[config.mode] = result

    git_sha, git_dirty = repo
    info = RetrievalRunInfo(
        date=now,
        git_sha=git_sha,
        git_dirty=git_dirty,
        golden_set_version=golden_set_version,
        golden_set_sha256=golden_set_sha256,
        index_version_id=index.id,
        index_config_hash=index.config_hash,
        fastapi_ref=index.git_ref,
        fastapi_sha=index.git_sha,
        embedding_model=index.embedding_model,
        embedding_dim=index.embedding_dim,
    )
    return RetrievalRun(info=info, configs=results)


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


def golden_set_digest(path: Path) -> str:
    """sha256 of the golden-set file with CRLF read as LF, like the migration checksum: a Windows
    checkout must not look like an edited file, while any content change still does."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def results_path(now: datetime) -> Path:
    """``eval/results/<UTC timestamp>-retrieval.json``."""
    return RESULTS_DIR / f"{now.strftime('%Y%m%dT%H%M%SZ')}-retrieval.json"


def write_run(path: Path, run: RetrievalRun) -> None:
    write_json(path, run.model_dump(mode="json"))


def read_run(path: Path) -> RetrievalRun:
    return RetrievalRun.model_validate_json(path.read_bytes())


def read_baseline(path: Path) -> dict[str, RetrievalBaselineEntry]:
    """The committed baseline file, validated: a row missing a required field is an error."""
    return _BASELINE_FILE.validate_json(path.read_bytes())


def update_baseline(path: Path, run: RetrievalRun) -> dict[str, RetrievalBaselineEntry]:
    """Replace the rows of the configs in ``run`` in the baseline file (created if missing) and
    keep the other rows, so ablation configs can be added one run at a time.

    A kept row must come from the same setup as the run (golden-set version and bytes, index
    config): otherwise the file would compare configs scored on different data, and the gate would
    read that as a difference between the configs. A mismatch raises ``BaselineMismatchError`` and
    leaves the file untouched; the fix is one run of all configs together.
    """
    git_dirty = run.info.git_dirty
    if git_dirty is None:
        # A baseline row must say whether its numbers came from committed code (the gate needs it).
        raise RetrievalEvalError(
            "this run has no git state (not a git checkout), so it can't become a baseline row"
        )
    try:
        rows = read_baseline(path) if path.exists() else {}
    except ValidationError as exc:
        # E.g. a row written before the golden-set hash and git state became required.
        raise BaselineMismatchError(
            f"{path.name} has rows that no longer validate ({exc.error_count()} errors, first: "
            f"{exc.errors()[0]['loc']}). Rows scored on an older setup must be rewritten together: "
            "delete the file (keep a copy of its thresholds) and run all configs with "
            "`--write-baseline`. The baseline file was not changed."
        ) from exc
    kept = {name: row for name, row in rows.items() if name not in run.configs}
    _check_same_setup(kept, run, joint=[*dict.fromkeys([*kept, *run.configs])])
    for name, result in run.configs.items():
        previous = rows.get(name)
        rows[name] = RetrievalBaselineEntry(
            metrics=result.metrics,
            # Thresholds are hand-written policy: refreshing the numbers must not erase them.
            thresholds=previous.thresholds if previous else {},
            n=result.n,
            k=result.k,
            golden_set_version=run.info.golden_set_version,
            golden_set_sha256=run.info.golden_set_sha256,
            index_config_hash=run.info.index_config_hash,
            retrieval_config_hash=result.retrieval_config_hash,
            fastapi_ref=run.info.fastapi_ref,
            fastapi_sha=run.info.fastapi_sha,
            embedding_model=run.info.embedding_model,
            embedding_dim=run.info.embedding_dim,
            git_sha=run.info.git_sha,
            git_dirty=git_dirty,
            date=run.info.date,
        )
    data: dict[str, Any] = _BASELINE_FILE.dump_python(rows, mode="json")
    for row in data.values():
        # A row without a gate is committed without the key (as the generation writer does), so
        # adding another config's row never rewrites the rows that were already there (6.04).
        if not row["thresholds"]:
            del row["thresholds"]
    write_json(path, data)
    return rows


def _check_same_setup(
    kept: Mapping[str, RetrievalBaselineEntry], run: RetrievalRun, *, joint: Sequence[str]
) -> None:
    """Raise if any kept row differs from ``run`` in golden-set version/bytes or index config.

    The index config hash already covers the FastAPI SHA, the embedding model and dimension and
    the chunking config, so those aren't compared one by one; the model is only named in the
    message when it differs, to make the mismatch readable.
    """
    info = run.info
    problems: list[str] = []
    for name, row in kept.items():
        differences: list[str] = []
        for field, row_value, run_value in (
            ("golden_set_version", row.golden_set_version, info.golden_set_version),
            ("golden_set_sha256", row.golden_set_sha256, info.golden_set_sha256),
            ("index_config_hash", row.index_config_hash, info.index_config_hash),
        ):
            if row_value != run_value:
                differences.append(f"{field}: row {_short(row_value)}, run {_short(run_value)}")
        if differences and row.embedding_model != info.embedding_model:
            differences.append(
                f"embedding_model: row {row.embedding_model}, run {info.embedding_model}"
            )
        if differences:
            problems.append(f"row {name!r} differs in " + "; ".join(differences))
    if problems:
        command = " ".join(f"--config {name}" for name in joint)
        raise BaselineMismatchError(
            "; ".join(problems) + f". Rows scored on different data must not share a file: run "
            f"`grounded eval retrieval {command} --write-baseline` to rewrite them together. "
            "The baseline file was not changed."
        )


def _short(value: str | None) -> str:
    """A hash as 8 characters; ``None`` = a row written before the field existed."""
    return "unset" if value is None else value[:8] if len(value) > 12 else value


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """The one way eval files are written (results and both baselines): UTF-8, LF, indent 2."""
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
