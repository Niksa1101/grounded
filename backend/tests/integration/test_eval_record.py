"""``eval_runs`` rows and ``grounded eval record`` against the real schema (ticket 4.10c).

Everything runs on the local pgvector test database (the network guard of ``conftest`` refuses any
other host), never on Neon. The command's two connections are both this database here: the source
of the index hash (``DATABASE_URL``) and the owner target (``DATABASE_URL_DIRECT``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from psycopg.rows import dict_row
from typer.testing import CliRunner

import grounded.cli
from grounded.cli import app
from grounded.evals.eval_record import (
    ActiveIndex,
    EvalRecordError,
    EvalRunRow,
    insert_rows,
    read_active_index,
)
from grounded.evals.generation_baseline import update_generation_baseline
from grounded.evals.generation_results import parse_results
from grounded.infra.kvcache import KVCache
from grounded.ingest.embed import FakeEmbedder
from grounded.ingest.pipeline import index_spec, ingest, prepare_corpus
from grounded.ingest.types import ChunkingConfig, CorpusCheckout
from tests.support import make_settings

pytestmark = pytest.mark.integration

runner = CliRunner()
HEAD = "c" * 40
URL = "https://github.com/o/r/actions/runs/1"
SAMPLE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "promptfoo" / "results_sample_judge.json"
)
CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"
# The index the recorded sample was made against (its `index_version` label).
SAMPLE_INDEX = ActiveIndex(git_ref="0.141.1", config_hash="4949e8a3" + "0" * 56)


@pytest.fixture
def db(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE eval_runs, index_versions RESTART IDENTITY CASCADE")

    wipe()
    yield test_database_url
    wipe()


def stored(db: str) -> list[dict[str, Any]]:
    with psycopg.connect(db) as conn, conn.cursor(row_factory=dict_row) as cur:
        return cur.execute("SELECT * FROM eval_runs ORDER BY config_name").fetchall()


def make_row(config: str = "hybrid", **fields: Any) -> EvalRunRow:
    row: dict[str, Any] = {
        "suite": "generation",
        "config_name": config,
        "git_sha": HEAD,
        "branch": "main",
        "golden_set_version": "v1",
        "prompt_version": "answer_v1@08cc49e5",
        "index_config_hash": "4949e8a3" + "0" * 56,
        "generator_model": "gemini-3.5-flash-lite",
        "judge_model": "openai/gpt-oss-120b",
        "status": "pass",
        "metrics": {"correctness": 0.9, "n_correctness": 30, "n": 30, "cost_per_1k_usd": 1.6},
        "case_count": 30,
        "errored_case_count": 2,
        "report_url": URL,
    }
    return EvalRunRow.model_validate(row | fields)


# --- insert_rows --------------------------------------------------------------------------------


def test_the_row_model_has_exactly_the_columns_the_database_does_not_fill(db: str) -> None:
    with psycopg.connect(db) as conn:
        columns = {
            name
            for (name,) in conn.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'eval_runs'"
            )
        }

    assert set(EvalRunRow.model_fields) == columns - {"id", "created_at"}


def test_insert_writes_every_field_and_returns_the_ids(db: str) -> None:
    ids = insert_rows(db, [make_row("hybrid"), make_row("no_rag", index_config_hash="none")])

    rows = stored(db)
    assert [str(i) for i in ids] != []
    assert {r["id"] for r in rows} == set(ids)
    hybrid, no_rag = rows  # ORDER BY config_name
    assert hybrid["config_name"] == "hybrid"
    assert (hybrid["suite"], hybrid["branch"], hybrid["git_sha"]) == ("generation", "main", HEAD)
    assert hybrid["golden_set_version"] == "v1"
    assert hybrid["prompt_version"] == "answer_v1@08cc49e5"
    assert hybrid["index_config_hash"] == "4949e8a3" + "0" * 56
    assert (hybrid["generator_model"], hybrid["judge_model"]) == (
        "gemini-3.5-flash-lite",
        "openai/gpt-oss-120b",
    )
    assert hybrid["status"] == "pass"
    assert hybrid["metrics"] == {
        "correctness": 0.9,
        "n_correctness": 30,
        "n": 30,
        "cost_per_1k_usd": 1.6,
    }
    assert (hybrid["case_count"], hybrid["errored_case_count"]) == (30, 2)
    assert hybrid["report_url"] == URL
    assert hybrid["created_at"] is not None
    assert no_rag["index_config_hash"] == "none"


def test_nullable_columns_may_be_null(db: str) -> None:
    insert_rows(
        db,
        [make_row(prompt_version=None, generator_model=None, judge_model=None, report_url=None)],
    )

    (row,) = stored(db)
    assert (row["prompt_version"], row["generator_model"], row["judge_model"]) == (None,) * 3
    assert row["report_url"] is None


def test_a_failing_row_inserts_none_of_the_rows_of_the_call(db: str) -> None:
    # model_copy does not validate: this is how a value the database would refuse gets through.
    too_long = make_row("no_rag").model_copy(update={"git_sha": "d" * 41})

    with pytest.raises(psycopg.Error):
        insert_rows(db, [make_row("hybrid"), too_long])

    assert stored(db) == []  # the first row was rolled back with the second


def test_two_calls_make_two_sets_of_rows(db: str) -> None:
    # No natural key (DB.md §4): a re-run of a CI job is a new run and a new row.
    insert_rows(db, [make_row()])
    insert_rows(db, [make_row()])

    assert len(stored(db)) == 2


# --- read_active_index --------------------------------------------------------------------------


def build_index(db: str, tmp_path: Path, *, activate: bool) -> None:
    def words(text: str) -> int:
        return len(text.split())

    checkout = CorpusCheckout(path=CORPUS_MINI, ref="0.0.1", sha="a" * 40)
    cfg = ChunkingConfig(max_tokens=60, overlap_tokens=10, min_tokens=5, tokenizer="words")
    corpus = prepare_corpus(checkout, cfg, words)
    spec = index_spec(checkout, cfg, embedding_model="fake-embedding", embedding_dim=768)
    with KVCache(tmp_path / "ingest-cache.sqlite") as cache:
        ingest(
            db,
            corpus,
            spec,
            embedder=FakeEmbedder(model="fake-embedding", dim=768),
            cache=cache,
            write_every=100,
            count_tokens=words,
            max_input_tokens=2048,
            activate=activate,
        )


def test_the_active_index_is_read_with_its_full_hash(db: str, tmp_path: Path) -> None:
    build_index(db, tmp_path, activate=True)

    active = read_active_index(db)

    assert active.git_ref == "0.0.1"
    assert len(active.config_hash) == 64
    assert active.label == f"0.0.1@{active.config_hash[:8]}"


def test_no_active_index_is_an_error(db: str, tmp_path: Path) -> None:
    build_index(db, tmp_path, activate=False)

    with pytest.raises(EvalRecordError, match="no active index"):
        read_active_index(db)


# --- The command --------------------------------------------------------------------------------


def clean_sample() -> bytes:
    """The 4.06 promptfoo recording cut to q003, q007 and q045: the questions of both configs with
    no provider or judge error."""
    document = json.loads(SAMPLE.read_bytes())
    rows = document["results"]["results"]
    document["results"]["results"] = [
        row for row in rows if row["metadata"]["golden"]["id"] in {"q003", "q007", "q045"}
    ]
    return json.dumps(document).encode()


@pytest.fixture
def results(tmp_path: Path) -> Path:
    path = tmp_path / "promptfoo.json"
    path.write_bytes(clean_sample())
    return path


@pytest.fixture
def baseline(tmp_path: Path, results: Path) -> Path:
    path = tmp_path / "generation.json"
    update_generation_baseline(
        path, parse_results(results.read_bytes(), git_sha=HEAD, git_dirty=False)
    )
    return path


@pytest.fixture
def cli(db: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean checkout at HEAD, the test database as both connections, the sample's index live."""
    settings = make_settings(database_url=db, database_url_direct=db)
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.cli, "repo_state", lambda: (HEAD, False))
    monkeypatch.setattr(grounded.cli, "read_active_index", lambda _url: SAMPLE_INDEX)


def record(results: Path, baseline: Path, *extra: str) -> Any:
    args = [
        "eval",
        "record",
        "--suite",
        "generation",
        "--results",
        str(results),
        "--baseline",
        str(baseline),
        "--branch",
        "main",
        "--report-url",
        URL,
    ]
    return runner.invoke(app, [*args, *extra])


@pytest.mark.usefixtures("cli")
def test_without_write_the_command_prints_the_rows_and_writes_nothing(
    db: str, results: Path, baseline: Path
) -> None:
    result = record(results, baseline)

    assert result.exit_code == 0, result.output
    assert "Dry run: nothing was written" in result.output
    assert "hybrid:" in result.output
    assert "no_rag:" in result.output
    assert stored(db) == []


@pytest.mark.usefixtures("cli")
def test_with_write_it_inserts_one_row_per_config(db: str, results: Path, baseline: Path) -> None:
    result = record(results, baseline, "--write")

    assert result.exit_code == 0, result.output
    assert "Inserted 2 rows into eval_runs" in result.output
    hybrid, no_rag = stored(db)
    assert (hybrid["config_name"], no_rag["config_name"]) == ("hybrid", "no_rag")
    assert hybrid["git_sha"] == HEAD
    assert hybrid["branch"] == "main"
    assert hybrid["report_url"] == URL
    assert hybrid["index_config_hash"] == SAMPLE_INDEX.config_hash
    assert no_rag["index_config_hash"] == "none"
    assert hybrid["status"] == no_rag["status"] == "fail"  # the sample is far under the thresholds
    assert hybrid["case_count"] == no_rag["case_count"] == 3


@pytest.mark.usefixtures("cli")
def test_a_row_and_the_output_hold_no_question_text_or_id(
    db: str, results: Path, baseline: Path
) -> None:
    result = record(results, baseline, "--write")

    assert result.exit_code == 0, result.output
    golden = json.loads(results.read_bytes())["results"]["results"]
    questions = {row["metadata"]["golden"]["id"] for row in golden}
    dumped = json.dumps(stored(db), default=str) + result.output
    for question_id in questions:
        assert question_id not in dumped
    assert "How do I" not in dumped


def test_it_refuses_to_write_without_the_owner_connection(
    db: str, results: Path, baseline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(database_url=db)  # no DATABASE_URL_DIRECT
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.cli, "repo_state", lambda: (HEAD, False))
    monkeypatch.setattr(grounded.cli, "read_active_index", lambda _url: SAMPLE_INDEX)

    result = record(results, baseline, "--write")

    assert result.exit_code == 1
    assert "DATABASE_URL_DIRECT is not set" in result.output
    assert stored(db) == []


@pytest.mark.parametrize(
    ("state", "message"),
    [((None, None), "not a git checkout"), ((HEAD, True), "tracked files are modified")],
)
@pytest.mark.usefixtures("cli")
def test_it_refuses_a_checkout_it_cannot_name(
    db: str,
    results: Path,
    baseline: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: tuple[str | None, bool | None],
    message: str,
) -> None:
    monkeypatch.setattr(grounded.cli, "repo_state", lambda: state)

    result = record(results, baseline, "--write")

    assert result.exit_code == 1
    assert message in result.output
    assert stored(db) == []


@pytest.mark.usefixtures("cli")
def test_it_refuses_results_made_on_another_index_than_the_active_one(
    db: str, results: Path, baseline: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = ActiveIndex(git_ref="0.0.1", config_hash="beef" + "0" * 60)
    monkeypatch.setattr(grounded.cli, "read_active_index", lambda _url: other)

    result = record(results, baseline, "--write")

    assert result.exit_code == 1
    assert "0.141.1@4949e8a3" in result.output
    assert "0.0.1@beef0000" in result.output
    assert stored(db) == []


@pytest.mark.usefixtures("cli")
def test_it_names_the_missing_files_and_writes_nothing(
    db: str, results: Path, baseline: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "nope.json"

    assert "Cannot read the results file" in record(missing, baseline, "--write").output
    assert "Cannot read the baseline" in record(results, missing, "--write").output
    assert stored(db) == []


@pytest.mark.usefixtures("cli")
def test_results_the_gate_cannot_judge_exit_2_and_write_nothing(
    db: str, results: Path, baseline: Path, tmp_path: Path
) -> None:
    document = json.loads(results.read_bytes())
    for row in document["results"]["results"]:
        if row["provider"]["label"] == "hybrid":
            row["metadata"]["error_kind"] = "ProviderRequestRejected"
            row["response"]["metadata"]["error_kind"] = "ProviderRequestRejected"
            row["error"] = "[ProviderRequestRejected quota=false] bad key"
            row["failureReason"] = 2
            row["success"] = False
            row["gradingResult"] = None
    rejected = tmp_path / "rejected.json"
    rejected.write_text(json.dumps(document), encoding="utf-8")

    result = record(rejected, baseline, "--write")

    assert result.exit_code == 2, result.output
    assert "The gate could not run" in result.output
    assert stored(db) == []
