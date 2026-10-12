"""`grounded eval retrieval` end to end: corpus_mini indexed with a fake embedder, real pgvector."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import psycopg
import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import EXIT_EMBEDDINGS_UNAVAILABLE, EXIT_RERANK_UNAVAILABLE, app
from grounded.evals.retrieval_runner import golden_set_digest, read_run
from grounded.infra.kvcache import KVCache
from grounded.infra.provider_errors import (
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.ingest.embed import (
    FakeEmbedder,
    GeminiEmbedder,
    TaskType,
    TokenCounter,
    Vector,
)
from grounded.ingest.pipeline import index_spec, ingest, prepare_corpus
from grounded.ingest.types import ChunkingConfig, CorpusCheckout
from grounded.retrieval.config import RetrievalConfig
from grounded.retrieval.rerank import CachedReranker, PacedReranker, Reranker, ScoringReranker
from grounded.retrieval.types import RetrievedChunk
from grounded.settings import Settings
from tests.support import make_settings

pytestmark = pytest.mark.integration

runner = CliRunner()
CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"
BG = "docs/en/docs/tutorial/background-tasks.md"
MODEL = "fake-embedding"


class StubGemini(FakeEmbedder):
    """Stands in for GeminiEmbedder in the CLI; counts the texts that reach "the API"."""

    instances: list[StubGemini] = []  # noqa: RUF012  (test bookkeeping)

    @classmethod
    def from_settings(cls, settings: Settings, count_tokens: TokenCounter) -> StubGemini:
        assert settings.embedding_model is not None
        stub = cls(model=settings.embedding_model, dim=settings.embedding_dim)
        cls.instances.append(stub)
        return stub


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    yield test_database_url
    wipe()


def build_index(db: str, tmp_path: Path, *, activate: bool = True) -> None:
    def words(text: str) -> int:
        return len(text.split())

    checkout = CorpusCheckout(path=CORPUS_MINI, ref="0.0.1", sha="a" * 40)
    cfg = ChunkingConfig(max_tokens=60, overlap_tokens=10, min_tokens=5, tokenizer="words")
    corpus = prepare_corpus(checkout, cfg, words)
    spec = index_spec(checkout, cfg, embedding_model=MODEL, embedding_dim=768)
    with KVCache(tmp_path / "ingest-cache.sqlite") as cache:
        ingest(
            db,
            corpus,
            spec,
            embedder=FakeEmbedder(model=MODEL, dim=768),
            cache=cache,
            write_every=100,
            count_tokens=words,
            max_input_tokens=2048,
            activate=activate,
        )


@pytest.fixture
def baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "baselines" / "retrieval.json"
    monkeypatch.setattr(grounded.cli, "RETRIEVAL_BASELINE", path)
    return path


@pytest.fixture
def settings(db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = make_settings(
        database_url=db, cache_dir=tmp_path / "cache", embedding_model=MODEL, k_dense=4
    )
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.cli, "GeminiEmbedder", StubGemini)
    StubGemini.instances.clear()
    return settings


def golden_file(tmp_path: Path, name: str = "golden_set.v1.jsonl") -> Path:
    rows: list[dict[str, Any]] = [
        {
            "id": "q001",
            "question": "How do I run a function after returning a response?",
            "type": "how_to",
            "answerable": True,
            "reference_answer": "Use BackgroundTasks.",
            "relevant_sections": [{"section": f"{BG}#using-backgroundtasks", "grade": 2}],
        },
        {
            "id": "q002",
            "question": "How do I configure Django middleware?",
            "type": "unanswerable",
            "answerable": False,
            "reference_answer": "Not covered.",
            "relevant_sections": [],
        },
    ]
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def invoke(*args: str) -> Any:
    return runner.invoke(app, ["eval", "retrieval", *args])


def test_eval_writes_results_and_reuses_cached_query_vectors(
    db: str, tmp_path: Path, baseline: Path, settings: Settings
) -> None:
    build_index(db, tmp_path)
    golden, out = golden_file(tmp_path), tmp_path / "results" / "run.json"

    result = invoke("--golden", str(golden), "--out", str(out))
    assert result.exit_code == 0, result.output
    assert "golden set v1 (2 items)" in result.stdout
    assert "1 questions embedded, 0 from cache" in result.stdout
    assert "Skipped 1 unanswerable questions (not scored)." in result.stdout
    [row] = [line for line in result.stdout.splitlines() if line.startswith("dense ")]
    assert row.split()[:3] == ["dense", "1", "4"]
    assert not baseline.exists()

    run = read_run(out)
    assert list(run.configs) == ["dense"]
    dense = run.configs["dense"]
    assert (dense.n, dense.k, dense.skipped_unanswerable) == (1, 4, 1)
    expected = RetrievalConfig.from_settings(settings, "dense")
    assert dense.retrieval_config == expected
    assert dense.retrieval_config_hash == expected.config_hash
    [question] = dense.questions
    assert question.id == "q001"
    assert len(question.retrieved) == 4
    info = run.info
    assert (info.golden_set_version, info.fastapi_ref, info.fastapi_sha) == (
        "v1",
        "0.0.1",
        "a" * 40,
    )
    assert (info.embedding_model, info.embedding_dim) == (MODEL, 768)
    assert info.git_sha is not None
    assert info.golden_set_sha256 == golden_set_digest(golden)

    again = invoke("--golden", str(golden), "--out", str(out), "--write-baseline")
    assert again.exit_code == 0, again.output
    assert "0 questions embedded, 1 from cache" in again.stdout
    assert sum(len(texts) for stub in StubGemini.instances for texts, _ in stub.calls) == 1
    assert read_run(out).configs["dense"].metrics == dense.metrics  # deterministic

    rows = json.loads(baseline.read_bytes())
    assert list(rows) == ["dense"]
    assert rows["dense"]["metrics"] == dense.metrics
    assert rows["dense"]["n"] == 1
    assert rows["dense"]["index_config_hash"] == info.index_config_hash
    assert rows["dense"]["golden_set_sha256"] == info.golden_set_sha256
    assert rows["dense"]["git_dirty"] == info.git_dirty
    assert rows["dense"]["retrieval_config_hash"] == expected.config_hash


def test_fts_and_dense_fts_score_the_active_index_without_new_embeddings(
    db: str, tmp_path: Path, baseline: Path, settings: Settings
) -> None:
    build_index(db, tmp_path)
    golden = golden_file(tmp_path)
    stop_words = {
        "id": "q003",
        "question": "the and or",
        "type": "how_to",
        "answerable": True,
        "reference_answer": "Use BackgroundTasks.",
        "relevant_sections": [{"section": f"{BG}#using-backgroundtasks", "grade": 2}],
    }
    with golden.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stop_words) + "\n")
    out = tmp_path / "results" / "run.json"

    first = invoke("--config", "fts", "--golden", str(golden), "--out", str(out))
    assert first.exit_code == 0, first.output
    assert "2 questions embedded, 0 from cache" in first.stdout
    fts_only = read_run(out)
    assert list(fts_only.configs) == ["fts"]
    fts = fts_only.configs["fts"]
    assert (fts.n, fts.k, fts.skipped_unanswerable) == (2, settings.k_fts, 1)
    assert fts.retrieval_config_hash == RetrievalConfig.from_settings(settings, "fts").config_hash
    stop_result = next(question for question in fts.questions if question.id == "q003")
    assert stop_result.retrieved == []
    assert all(rank is None for rank in stop_result.ranks.values())
    assert all(value == 0 for value in stop_result.metrics.values())

    again = invoke(
        "--config", "dense", "--config", "fts", "--golden", str(golden), "--out", str(out)
    )
    assert again.exit_code == 0, again.output
    assert "0 questions embedded, 4 from cache" in again.stdout
    assert sum(len(texts) for stub in StubGemini.instances for texts, _ in stub.calls) == 2
    combined = read_run(out)
    assert list(combined.configs) == ["dense", "fts"]
    assert combined.configs["fts"].metrics == fts.metrics
    assert combined.configs["dense"].k == settings.k_dense
    assert not baseline.exists()


def test_hybrid_scores_the_fused_list_next_to_its_two_inputs(
    db: str, tmp_path: Path, baseline: Path, settings: Settings
) -> None:
    build_index(db, tmp_path)
    golden, out = golden_file(tmp_path), tmp_path / "results" / "run.json"

    result = invoke(
        *("--config", "dense", "--config", "fts", "--config", "hybrid"),
        *("--golden", str(golden), "--out", str(out), "--write-baseline"),
    )
    assert result.exit_code == 0, result.output
    # The first mode embeds the question, the other two read it from the cache.
    assert "1 questions embedded, 2 from cache" in result.stdout
    assert sum(len(texts) for stub in StubGemini.instances for texts, _ in stub.calls) == 1
    run = read_run(out)
    assert list(run.configs) == ["dense", "fts", "hybrid"]
    hybrid = run.configs["hybrid"]
    expected = RetrievalConfig.from_settings(settings, "hybrid")
    assert hybrid.retrieval_config == expected
    assert hybrid.retrieval_config_hash == expected.config_hash
    assert hybrid.k == settings.k_fused  # MRR is taken over the fused list
    [question] = hybrid.questions
    assert 0 < len(question.retrieved) <= settings.k_fused

    rows = json.loads(baseline.read_bytes())
    assert list(rows) == ["dense", "fts", "hybrid"]
    assert rows["hybrid"]["metrics"] == hybrid.metrics
    assert rows["hybrid"]["k"] == settings.k_fused
    assert rows["hybrid"]["retrieval_config_hash"] == expected.config_hash


@pytest.mark.usefixtures("settings")
def test_eval_without_an_active_index(db: str, tmp_path: Path) -> None:
    build_index(db, tmp_path, activate=False)
    result = invoke("--golden", str(golden_file(tmp_path)), "--out", str(tmp_path / "r.json"))
    assert result.exit_code == 1
    assert "no active index version" in result.output


def test_eval_refuses_another_embedding_model(
    db: str, tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_index(db, tmp_path)
    other = settings.model_copy(update={"embedding_model": "other-model"})
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: other)
    result = invoke("--golden", str(golden_file(tmp_path)), "--out", str(tmp_path / "r.json"))
    assert result.exit_code == 1
    assert "was built with fake-embedding" in result.output
    assert "other-model" in result.output


@pytest.mark.usefixtures("settings")
@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            ["--config", "rerank"],
            "Unknown retrieval config 'rerank'; available: dense, fts, hybrid, hybrid_rerank.",
        ),
        (["--golden", "{tmp}/golden.jsonl"], "golden_set.v<N>.jsonl"),
    ],
)
def test_eval_rejects_bad_arguments(tmp_path: Path, args: list[str], message: str) -> None:
    golden_file(tmp_path, "golden.jsonl")
    result = invoke(*(arg.replace("{tmp}", str(tmp_path)) for arg in args))
    assert result.exit_code == 1
    assert message in result.output


def without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """From here on the real ``GeminiEmbedder`` has no key, so using it fails the command."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(grounded.cli, "GeminiEmbedder", GeminiEmbedder)


@pytest.mark.usefixtures("settings")
def test_eval_with_every_query_vector_cached_needs_no_key(
    db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_index(db, tmp_path)
    args = ("--golden", str(golden_file(tmp_path)), "--out", str(tmp_path / "r.json"))
    assert invoke(*args).exit_code == 0

    without_a_key(monkeypatch)
    again = invoke(*args)
    assert again.exit_code == 0, again.output
    assert "0 questions embedded, 1 from cache" in again.stdout


@pytest.mark.usefixtures("settings")
def test_eval_on_a_cold_cache_without_a_key_fails_cleanly(
    db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_index(db, tmp_path)
    without_a_key(monkeypatch)
    result = invoke("--golden", str(golden_file(tmp_path)), "--out", str(tmp_path / "r.json"))
    assert result.exit_code == EXIT_EMBEDDINGS_UNAVAILABLE
    assert "GEMINI_API_KEY must be set to embed: 1 questions are not cached." in result.output
    assert isinstance(result.exception, SystemExit)


class QuotaGemini(StubGemini):
    """The daily embedding quota is already spent."""

    async def embed(self, texts: Sequence[str], task_type: TaskType) -> list[Vector]:
        raise ProviderRateLimited("quota", retry_after_s=None, is_quota=True)


@pytest.mark.usefixtures("settings")
def test_eval_stopped_by_the_daily_quota_is_not_a_plain_failure(
    db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_index(db, tmp_path)
    monkeypatch.setattr(grounded.cli, "GeminiEmbedder", QuotaGemini)
    result = invoke("--golden", str(golden_file(tmp_path)), "--out", str(tmp_path / "r.json"))
    assert result.exit_code == EXIT_EMBEDDINGS_UNAVAILABLE
    assert "Query embedding failed (ProviderRateLimited)" in result.output


@pytest.mark.usefixtures("settings")
def test_write_baseline_refuses_to_mix_setups_but_still_writes_the_results(
    db: str, tmp_path: Path, baseline: Path
) -> None:
    build_index(db, tmp_path)
    golden, out = golden_file(tmp_path), tmp_path / "results" / "run.json"
    assert invoke("--golden", str(golden), "--out", str(out), "--write-baseline").exit_code == 0

    # A row of another config that was scored on different golden-set bytes.
    rows = json.loads(baseline.read_bytes())
    rows["fts"] = rows["dense"] | {"golden_set_sha256": "b" * 64}
    baseline.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    before = baseline.read_bytes()
    out.unlink()

    # dense is re-run alone: the fts row is kept, and it disagrees with this run.
    result = invoke("--golden", str(golden), "--out", str(out), "--write-baseline")
    assert result.exit_code == 1
    assert "Baseline not updated: row 'fts' differs in golden_set_sha256" in result.output
    assert "--config fts --config dense --write-baseline" in result.output
    assert isinstance(result.exception, SystemExit)
    assert out.exists()  # the run itself is kept
    assert baseline.read_bytes() == before


# --- hybrid_rerank (6.04): the reranker is a fake, the database and the metrics are real ----------

LABEL = f"{BG}#using-backgroundtasks"
RERANK_MODEL = "rerank-v4.0-pro"


class ReverseReranker(ScoringReranker):
    """Scores that rise with the RRF position, so the reranked order is the RRF order reversed.
    ``fail_on`` maps a 1-based call number to the error that call raises."""

    def __init__(self, fail_on: dict[int, ProviderError] | None = None) -> None:
        super().__init__(RERANK_MODEL)
        self.calls = 0
        self._fail_on = fail_on or {}

    async def _scores(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[float]:
        self.calls += 1
        if self.calls in self._fail_on:
            raise self._fail_on[self.calls]
        return [(i + 1) / (len(chunks) + 1) for i in range(len(chunks))]


@pytest.fixture
def rerank_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """The settings of a manual hybrid_rerank run: rerank on in `.env`, no real key needed because
    the tests replace the builder."""
    on = settings.model_copy(
        update={
            "rerank_provider": "cohere",
            "rerank_model": RERANK_MODEL,
            "cohere_api_key": SecretStr("test-key"),
            "eval_rerank_min_interval_s": 0.0,
        }
    )
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: on)
    return on


def use_reranker(monkeypatch: pytest.MonkeyPatch, reranker: ScoringReranker) -> list[bool]:
    """Make the CLI build ``cache(pacer(reranker))`` like ``runtime.build_reranker`` does, and
    return the list that records the ``paced`` flag of every build."""
    seen: list[bool] = []

    def build(settings: Settings, stack: AsyncExitStack, *, paced: bool = False) -> Reranker:
        seen.append(paced)
        cache = stack.enter_context(KVCache(settings.cache_dir / "rerank.sqlite"))
        return CachedReranker(PacedReranker(reranker, min_interval_s=0.0), cache)

    monkeypatch.setattr(grounded.cli, "build_reranker", build)
    return seen


def two_question_golden(tmp_path: Path) -> Path:
    golden = golden_file(tmp_path)
    second = {
        "id": "q003",
        "question": "the and or",
        "type": "how_to",
        "answerable": True,
        "reference_answer": "Use BackgroundTasks.",
        "relevant_sections": [{"section": LABEL, "grade": 2}],
    }
    with golden.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(second) + "\n")
    return golden


def test_hybrid_rerank_scores_the_reranked_list_of_the_same_candidates_as_hybrid(
    db: str,
    tmp_path: Path,
    baseline: Path,
    rerank_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_index(db, tmp_path)
    reranker = ReverseReranker()
    paced = use_reranker(monkeypatch, reranker)
    golden, out = golden_file(tmp_path), tmp_path / "results" / "run.json"

    result = invoke(
        "--config",
        "hybrid",
        "--config",
        "hybrid_rerank",
        "--golden",
        str(golden),
        "--out",
        str(out),
    )

    assert result.exit_code == 0, result.output
    assert paced == [True]  # the eval asks for the pacer
    assert reranker.calls == 1  # one answerable question, one live call
    assert "1 live calls, 0 from the rerank cache; live latency p50 " in result.stdout
    run = read_run(out)
    assert list(run.configs) == ["hybrid", "hybrid_rerank"]
    plain, reranked = run.configs["hybrid"], run.configs["hybrid_rerank"]

    # Same candidates, the reranker's order: the whole list, so MRR and Recall@10 are over K_FUSED.
    assert reranked.k == plain.k == rerank_settings.k_fused
    [before], [after] = plain.questions, reranked.questions
    assert len(before.retrieved) > 1
    assert after.retrieved == before.retrieved[::-1]
    # The ranks are those of the reranked order (a label at position p of n moves to n + 1 - p).
    n = len(before.retrieved)
    moved = min(n + 1 - p for p, section in enumerate(before.retrieved, 1) if section == LABEL)
    assert after.ranks == {LABEL: moved}
    assert reranked.metrics["mrr"] == pytest.approx(1 / moved)

    # What each row says it ran: hybrid without the reranker `.env` has, hybrid_rerank with it.
    without = rerank_settings.model_copy(update={"rerank_provider": "none", "rerank_model": None})
    assert plain.retrieval_config == RetrievalConfig.from_settings(without, "hybrid")
    assert plain.retrieval_config.rerank_provider == "none"
    assert reranked.retrieval_config.rerank_provider == "cohere"
    assert reranked.retrieval_config.rerank_model == RERANK_MODEL
    assert reranked.retrieval_config_hash != plain.retrieval_config_hash
    assert plain.rerank is None
    stats = reranked.rerank
    assert stats is not None
    assert (stats.model, stats.live_calls, stats.cache_hits) == (RERANK_MODEL, 1, 0)
    assert stats.latency_p50_ms is not None
    assert stats.latency_p95_ms is not None
    assert stats.latency_p50_ms >= 0


def test_a_rerun_is_answered_from_the_rerank_cache_without_a_live_call(
    db: str,
    tmp_path: Path,
    baseline: Path,
    rerank_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_index(db, tmp_path)
    golden, out = golden_file(tmp_path), tmp_path / "results" / "run.json"
    first = ReverseReranker()
    use_reranker(monkeypatch, first)
    args = ("--config", "hybrid_rerank", "--golden", str(golden), "--out", str(out))
    assert invoke(*args).exit_code == 0
    before = read_run(out).configs["hybrid_rerank"]

    again = ReverseReranker()
    use_reranker(monkeypatch, again)
    result = invoke(*args)

    assert result.exit_code == 0, result.output
    assert again.calls == 0  # a hit asks Cohere nothing
    assert "0 live calls, 1 from the rerank cache; no live call, so no latency" in result.stdout
    after = read_run(out).configs["hybrid_rerank"]
    assert after.metrics == before.metrics
    assert after.questions == before.questions
    assert after.rerank is not None
    assert (after.rerank.live_calls, after.rerank.cache_hits) == (0, 1)
    assert (after.rerank.latency_p50_ms, after.rerank.latency_p95_ms) == (None, None)


def test_the_rerank_row_is_added_without_touching_a_single_byte_of_the_others(
    db: str,
    tmp_path: Path,
    baseline: Path,
    rerank_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_index(db, tmp_path)
    golden, out = golden_file(tmp_path), tmp_path / "results" / "run.json"
    use_reranker(monkeypatch, ReverseReranker())
    plain_args = ("--golden", str(golden), "--out", str(out), "--write-baseline")
    plain = invoke("--config", "dense", "--config", "fts", "--config", "hybrid", *plain_args)
    assert plain.exit_code == 0, plain.output
    # The committed file: hybrid is gated, dense and fts are not and have no `thresholds` key.
    rows = json.loads(baseline.read_bytes())
    assert all("thresholds" not in row for row in rows.values())
    gated = {"metrics": rows["hybrid"].pop("metrics"), "thresholds": {"mrr": {"tolerance": 0.04}}}
    rows["hybrid"] = gated | rows["hybrid"]
    baseline.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    committed = baseline.read_text(encoding="utf-8")

    result = invoke("--config", "hybrid_rerank", *plain_args)

    assert result.exit_code == 0, result.output
    rewritten = baseline.read_text(encoding="utf-8")
    # Every existing row is the same bytes; the new row is appended after them.
    assert rewritten.startswith(committed[: committed.rindex("\n}")] + ",\n")
    after = json.loads(rewritten)
    assert list(after) == ["dense", "fts", "hybrid", "hybrid_rerank"]
    assert after["hybrid"]["thresholds"] == {"mrr": {"tolerance": 0.04}}
    assert "thresholds" not in after["hybrid_rerank"]  # a reranked row is ungated (D55)
    assert after["hybrid_rerank"]["k"] == rerank_settings.k_fused
    reranked = RetrievalConfig.from_settings(rerank_settings, "hybrid_rerank")
    assert after["hybrid_rerank"]["retrieval_config_hash"] == reranked.config_hash


@pytest.mark.parametrize(
    ("error", "code", "hint"),
    [
        (
            ProviderRateLimited("Too many requests", retry_after_s=30.0, is_quota=False),
            "rate_limited",
            "Cohere asked to wait 30s; the run does not wait out a limit",
        ),
        (
            ProviderRateLimited("Too many requests", retry_after_s=None, is_quota=False),
            "rate_limited",
            "the run does not wait out a limit",
        ),
        (ProviderTimeout("rerank call exceeded 3s"), "timeout", "Nothing was scored or written"),
        (ProviderUnavailable("Cohere rerank 503"), "unavailable", "Nothing was scored or written"),
        (
            ProviderRequestRejected("Cohere rerank 404: model not found", status_code=404),
            "rejected",
            "Check COHERE_API_KEY and RERANK_MODEL",
        ),
    ],
)
def test_a_reranker_error_ends_the_run_with_nothing_written_and_exit_3(
    db: str,
    tmp_path: Path,
    baseline: Path,
    rerank_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    error: ProviderError,
    code: str,
    hint: str,
) -> None:
    build_index(db, tmp_path)
    reranker = ReverseReranker(fail_on={2: error})  # q001 is reranked, q003 is not
    use_reranker(monkeypatch, reranker)
    golden, out = two_question_golden(tmp_path), tmp_path / "results" / "run.json"

    result = invoke(
        "--config", "hybrid_rerank", "--golden", str(golden), "--out", str(out), "--write-baseline"
    )

    assert result.exit_code == EXIT_RERANK_UNAVAILABLE == EXIT_EMBEDDINGS_UNAVAILABLE
    assert isinstance(result.exception, SystemExit)
    assert f"Eval stopped: rerank failed at q003 ({code}, {type(error).__name__})" in result.output
    assert hint in result.output
    assert "Nothing was scored or written" in result.output
    assert "the and or" not in result.output  # the id is named, never the question
    assert reranker.calls == 2  # stopped at the failure: no retry, no further question
    assert not out.exists()  # no results file for a half run
    assert not baseline.exists()  # and no baseline


@pytest.mark.usefixtures("settings")
def test_hybrid_rerank_with_the_provider_off_fails_before_any_work(tmp_path: Path) -> None:
    # No index exists and the embedder would be the first thing to run: the message proves neither
    # was reached. It is not a hybrid run under another name.
    result = invoke("--config", "hybrid_rerank", "--golden", str(golden_file(tmp_path)))

    assert result.exit_code == 1
    assert "hybrid_rerank needs RERANK_PROVIDER=cohere" in result.output
    assert "never falls back to hybrid" in result.output
    assert "no active index" not in result.output
    assert sum(len(texts) for stub in StubGemini.instances for texts, _ in stub.calls) == 0


def test_hybrid_rerank_without_a_key_or_a_model_fails_before_any_work(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    golden = golden_file(tmp_path)
    for overrides, variable in (
        ({"rerank_provider": "cohere", "rerank_model": RERANK_MODEL}, "COHERE_API_KEY"),
        (
            {"rerank_provider": "cohere", "cohere_api_key": SecretStr("test-key")},
            "RERANK_MODEL",
        ),
    ):
        broken = settings.model_copy(update=overrides)
        monkeypatch.setattr(grounded.cli, "get_settings", lambda broken=broken: broken)

        result = invoke("--config", "hybrid_rerank", "--golden", str(golden))

        assert result.exit_code == 1
        assert f"Eval failed: {variable} must be set" in result.output
        assert "no active index" not in result.output
    assert sum(len(texts) for stub in StubGemini.instances for texts, _ in stub.calls) == 0


def test_plain_modes_build_no_reranker_and_carry_no_rerank_fields(
    db: str,
    tmp_path: Path,
    baseline: Path,
    rerank_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_index(db, tmp_path)
    paced = use_reranker(monkeypatch, ReverseReranker())
    out = tmp_path / "results" / "run.json"

    result = invoke("--config", "hybrid", "--golden", str(golden_file(tmp_path)), "--out", str(out))

    assert result.exit_code == 0, result.output
    assert paced == []  # CI's configs (dense, fts, hybrid) never build or call a reranker
    hybrid = read_run(out).configs["hybrid"]
    assert hybrid.retrieval_config.rerank_provider == "none"
    assert hybrid.retrieval_config.rerank_model is None
    assert hybrid.rerank is None
