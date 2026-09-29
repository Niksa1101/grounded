"""`grounded eval retrieval` end to end: corpus_mini indexed with a fake embedder, real pgvector."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.retrieval_runner import golden_set_digest, read_run
from grounded.infra.kvcache import KVCache
from grounded.ingest.embed import FakeEmbedder, GeminiEmbedder, TokenCounter
from grounded.ingest.pipeline import index_spec, ingest, prepare_corpus
from grounded.ingest.types import ChunkingConfig, CorpusCheckout
from grounded.retrieval.config import RetrievalConfig
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
        (["--config", "hybrid"], "Unknown retrieval config 'hybrid'; available: dense."),
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
    assert result.exit_code == 1
    assert "GEMINI_API_KEY must be set to embed: 1 questions are not cached." in result.output
    assert isinstance(result.exception, SystemExit)


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
