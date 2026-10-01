"""Retrieval eval runner without a database: fake embedder, scripted retrieval (Tech.md §15.2)."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from grounded.evals.metrics import ChunkRef
from grounded.evals.retrieval_runner import (
    BaselineMismatchError,
    RetrievalEvalError,
    evaluate,
    golden_set_digest,
    golden_set_version,
    metric_names,
    read_run,
    repo_state,
    results_path,
    score_question,
    update_baseline,
    write_run,
)
from grounded.ingest.embed import FakeEmbedder, Vector
from grounded.retrieval.config import RetrievalConfig, RetrievalMode
from grounded.schemas.eval import (
    GoldenItem,
    RetrievalConfigResult,
    RetrievalRun,
    RetrievalRunInfo,
)

PAGE = "docs/en/docs/tutorial/p.md"
OTHER = "docs/en/docs/tutorial/q.md"
NOW = datetime(2026, 9, 28, 12, 30, 5, tzinfo=UTC)


@dataclass(frozen=True)
class Ref:
    section_id: str
    anchor_path: tuple[str, ...]


def chunk(page: str, *anchors: str) -> Ref:
    return Ref(f"{page}#{anchors[-1] if anchors else ''}", anchors)


NOISE = chunk("docs/en/docs/elsewhere.md", "noise")


def retrieval_config(mode: RetrievalMode = "dense", **overrides: Any) -> RetrievalConfig:
    fields: dict[str, Any] = {
        "mode": mode,
        "k_dense": 20,
        "k_fts": 20,
        "k_fused": 40,
        "k_context": 5,
        "rrf_k": 60,
    }
    return RetrievalConfig.model_validate(fields | overrides)


def item(
    id: str = "q001", *, relevant: dict[str, int] | None = None, **overrides: Any
) -> GoldenItem:
    labels = relevant if relevant is not None else {f"{PAGE}#a": 2}
    row: dict[str, Any] = {
        "id": id,
        "question": f"question {id}?",
        "type": "how_to",
        "answerable": True,
        "reference_answer": "…",
        "relevant_sections": [{"section": s, "grade": g} for s, g in labels.items()],
    }
    return GoldenItem.model_validate(row | overrides)


def unanswerable(id: str) -> GoldenItem:
    return item(id, relevant={}, type="unanswerable", answerable=False)


def test_metric_names_in_report_order() -> None:
    assert metric_names() == ["recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10"]


def test_score_question_by_hand() -> None:
    # Labels: a (2) at rank 2, the whole page q.md (1) at rank 3, b (2) at rank 7, d (1) missing.
    question = item(
        relevant={f"{PAGE}#a": 2, f"{PAGE}#b": 2, OTHER: 1, f"{PAGE}#d": 1},
    )
    retrieved = [
        NOISE,
        chunk(PAGE, "a"),
        chunk(OTHER, "x"),
        NOISE,
        NOISE,
        NOISE,
        chunk(PAGE, "b"),
        chunk(PAGE, "a"),  # a second part of a: takes a position, doesn't re-rank a
    ]
    result = score_question(question, retrieved)

    assert result.ranks == {f"{PAGE}#a": 2, f"{PAGE}#b": 7, OTHER: 3, f"{PAGE}#d": None}
    dcg5 = 3 / math.log2(3) + 1 / math.log2(4)
    dcg10 = dcg5 + 3 / math.log2(8)
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4) + 1 / math.log2(5)
    assert result.metrics == pytest.approx(
        {
            "recall@5": 0.5,
            "recall@10": 1.0,
            "mrr": 0.5,
            "ndcg@5": dcg5 / ideal,
            "ndcg@10": dcg10 / ideal,
        }
    )
    assert list(result.metrics) == metric_names()
    assert result.retrieved[:3] == [NOISE.section_id, f"{PAGE}#a", f"{OTHER}#x"]


def test_score_question_nothing_found() -> None:
    result = score_question(item(), [NOISE, NOISE])
    assert result.ranks == {f"{PAGE}#a": None}
    assert set(result.metrics.values()) == {0.0}


class ScriptedSearch:
    """Returns a fixed list per question and records the vectors it was given."""

    def __init__(self, lists: dict[str, list[Ref]]) -> None:
        self.lists = lists
        self.seen: list[tuple[str, Vector]] = []

    async def __call__(self, question: str, vector: Vector) -> Sequence[ChunkRef]:
        self.seen.append((question, vector))
        return self.lists[question]


async def test_evaluate_scores_answerable_questions_and_averages() -> None:
    items = [item("q001"), unanswerable("q002"), item("q003")]
    search = ScriptedSearch(
        {
            "question q001?": [chunk(PAGE, "a")],  # rank 1: every metric 1.0
            "question q003?": [NOISE, NOISE, NOISE, NOISE, NOISE, NOISE, chunk(PAGE, "a")],
        }
    )
    embedder = FakeEmbedder(dim=8)

    config = retrieval_config()
    result = await evaluate(items, config=config, k=20, embedder=embedder, search=search)

    assert (result.config, result.k, result.n, result.skipped_unanswerable) == ("dense", 20, 2, 1)
    # The result says which retrieval produced it: the config itself and its hash.
    assert result.retrieval_config == config
    assert result.retrieval_config_hash == config.config_hash
    assert [q.id for q in result.questions] == ["q001", "q003"]
    ndcg = 3 / math.log2(8) / 3  # q003: rank 7, one grade-2 label
    assert result.metrics == pytest.approx(
        {
            "recall@5": (1.0 + 0.0) / 2,
            "recall@10": (1.0 + 1.0) / 2,
            "mrr": (1.0 + 1 / 7) / 2,
            "ndcg@5": (1.0 + 0.0) / 2,
            "ndcg@10": (1.0 + ndcg) / 2,
        }
    )
    # One embedding call, query task, answerable questions only, each vector to its question.
    assert embedder.calls == [(["question q001?", "question q003?"], "RETRIEVAL_QUERY")]
    vectors = await FakeEmbedder(dim=8).embed(
        ["question q001?", "question q003?"], "RETRIEVAL_QUERY"
    )
    assert search.seen == list(zip(["question q001?", "question q003?"], vectors, strict=True))


async def test_evaluate_needs_an_answerable_question() -> None:
    with pytest.raises(RetrievalEvalError, match="no answerable"):
        await evaluate(
            [unanswerable("q001")],
            config=retrieval_config(),
            k=20,
            embedder=FakeEmbedder(),
            search=ScriptedSearch({}),
        )


@pytest.mark.parametrize(
    ("name", "version"), [("golden_set.v1.jsonl", "v1"), ("golden_set.v12.jsonl", "v12")]
)
def test_golden_set_version(name: str, version: str) -> None:
    assert golden_set_version(Path("eval/golden") / name) == version


@pytest.mark.parametrize("name", ["golden.jsonl", "candidates.v1.jsonl", "golden_set.v1.json"])
def test_unversioned_golden_set_is_refused(name: str) -> None:
    with pytest.raises(RetrievalEvalError, match=r"golden_set\.v<N>\.jsonl"):
        golden_set_version(Path(name))


def test_results_path_is_a_utc_timestamp() -> None:
    assert results_path(NOW).name == "20260928T123005Z-retrieval.json"
    assert results_path(NOW).parent.name == "results"


def make_run(
    *configs: tuple[RetrievalMode, float], sha: str = "c" * 40, **info_overrides: Any
) -> RetrievalRun:
    info = RetrievalRunInfo(
        **{
            "date": NOW,
            "git_sha": sha,
            "git_dirty": False,
            "golden_set_version": "v1",
            "golden_set_sha256": "a" * 64,
            "index_version_id": 3,
            "index_config_hash": "f" * 64,
            "fastapi_ref": "0.141.1",
            "fastapi_sha": "9" * 40,
            "embedding_model": "gemini-embedding-001",
            "embedding_dim": 768,
        }
        | info_overrides
    )
    results = {
        name: RetrievalConfigResult(
            config=name,
            retrieval_config=retrieval_config(name),
            retrieval_config_hash=retrieval_config(name).config_hash,
            k=20,
            n=25,
            skipped_unanswerable=5,
            metrics=dict.fromkeys(metric_names(), value),
            questions=[],
        )
        for name, value in configs
    }
    return RetrievalRun(info=info, configs=results)


def result_fields(config: RetrievalConfig, **overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "config": "dense",
        "retrieval_config": config,
        "retrieval_config_hash": config.config_hash,
        "k": 20,
        "n": 1,
        "skipped_unanswerable": 0,
        "metrics": {},
        "questions": [],
    }
    return fields | overrides


def test_result_rejects_a_hash_that_does_not_match_its_config() -> None:
    edited = result_fields(
        retrieval_config("dense"),
        retrieval_config_hash=retrieval_config("dense", k_dense=10).config_hash,
    )
    with pytest.raises(ValueError, match="retrieval_config_hash does not match"):
        RetrievalConfigResult.model_validate(edited)


def test_result_rejects_a_config_for_another_mode() -> None:
    with pytest.raises(ValueError, match="does not match config"):
        RetrievalConfigResult.model_validate(result_fields(retrieval_config("hybrid")))


def test_run_file_round_trip(tmp_path: Path) -> None:
    run = make_run(("dense", 0.5))
    path = tmp_path / "nested" / "run.json"
    write_run(path, run)
    assert read_run(path) == run
    assert b"\r\n" not in path.read_bytes()


def test_baseline_rows_keep_their_own_config_hash(tmp_path: Path) -> None:
    # Configs of different modes hash differently by design, so the hash is never part of the
    # "same setup" check: a joint file holds one hash per row.
    rows = update_baseline(tmp_path / "retrieval.json", make_run(("dense", 0.5), ("fts", 0.25)))
    assert rows["dense"].retrieval_config_hash == retrieval_config("dense").config_hash
    assert rows["fts"].retrieval_config_hash == retrieval_config("fts").config_hash
    assert rows["dense"].retrieval_config_hash != rows["fts"].retrieval_config_hash


def test_update_baseline_creates_replaces_and_keeps_rows(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    update_baseline(path, make_run(("dense", 0.5)))
    first = json.loads(path.read_bytes())
    assert list(first) == ["dense"]
    assert first["dense"]["metrics"]["mrr"] == 0.5
    assert first["dense"]["n"] == 25
    assert first["dense"]["index_config_hash"] == "f" * 64
    assert first["dense"]["git_sha"] == "c" * 40
    assert first["dense"]["golden_set_sha256"] == "a" * 64
    assert first["dense"]["git_dirty"] is False
    assert first["dense"]["retrieval_config_hash"] == retrieval_config("dense").config_hash

    # A later run of another config adds its row and leaves dense alone ...
    update_baseline(path, make_run(("fts", 0.25), sha="d" * 40))
    second = json.loads(path.read_bytes())
    assert list(second) == ["dense", "fts"]
    assert second["dense"] == first["dense"]
    # ... and a re-run of dense replaces only dense.
    rows = update_baseline(path, make_run(("dense", 0.75), sha="e" * 40))
    assert rows["dense"].metrics["mrr"] == 0.75
    assert rows["fts"].git_sha == "d" * 40
    assert path.read_bytes().endswith(b"}\n")


def test_update_baseline_rejects_a_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    path.write_text('{"dense": {"metrics": {}}}\n', encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(BaselineMismatchError, match="no longer validate"):
        update_baseline(path, make_run(("fts", 0.25)))
    assert path.read_bytes() == before


def test_repo_state_in_and_outside_a_checkout(tmp_path: Path) -> None:
    sha, dirty = repo_state()
    assert sha is not None
    assert len(sha) == 40
    assert isinstance(dirty, bool)
    assert repo_state(tmp_path) == (None, None)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_repo_state_ignores_untracked_files_but_not_edits(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "commit.gpgsign", "false")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-q", "-m", "init")

    sha, dirty = repo_state(tmp_path)
    assert sha is not None
    assert dirty is False

    # A scratch file or a new ticket directory doesn't change the code under test ...
    (tmp_path / "scratch.py").write_text("print()\n", encoding="utf-8")
    assert repo_state(tmp_path) == (sha, False)

    # ... but an edit to a tracked file does.
    tracked.write_text("two\n", encoding="utf-8")
    assert repo_state(tmp_path) == (sha, True)


# --- Baseline integrity: rows from different setups are never mixed -----------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("golden_set_version", "v2"),
        ("golden_set_sha256", "b" * 64),
        ("index_config_hash", "e" * 64),
    ],
)
def test_update_baseline_refuses_a_run_from_another_setup(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "retrieval.json"
    update_baseline(path, make_run(("dense", 0.5)))
    before = path.read_bytes()

    with pytest.raises(BaselineMismatchError) as excinfo:
        update_baseline(path, make_run(("fts", 0.25), **{field: value}))

    message = str(excinfo.value)
    assert f"row 'dense' differs in {field}" in message
    # The fix is named: one run of all configs, in a stable order.
    assert "--config dense --config fts --write-baseline" in message
    assert path.read_bytes() == before  # byte-identical: nothing was written


def test_update_baseline_names_every_differing_field_and_the_model(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    update_baseline(path, make_run(("dense", 0.5)))
    other = make_run(
        ("fts", 0.25),
        golden_set_version="v2",
        golden_set_sha256="b" * 64,
        index_config_hash="e" * 64,
        embedding_model="other-model",
    )
    with pytest.raises(BaselineMismatchError) as excinfo:
        update_baseline(path, other)
    message = str(excinfo.value)
    for field in ("golden_set_version", "golden_set_sha256", "index_config_hash"):
        assert field in message
    assert "embedding_model: row gemini-embedding-001, run other-model" in message


def test_update_baseline_refuses_a_legacy_row_without_the_required_fields(tmp_path: Path) -> None:
    # A row written before golden_set_sha256/git_dirty existed can't be read any more (2.09): it
    # can't be shown to come from the same golden-set bytes. The message names the way out.
    path = tmp_path / "retrieval.json"
    update_baseline(path, make_run(("dense", 0.5)))
    rows = json.loads(path.read_bytes())
    del rows["dense"]["golden_set_sha256"], rows["dense"]["git_dirty"]
    del rows["dense"]["retrieval_config_hash"]
    path.write_text(json.dumps(rows), encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(BaselineMismatchError, match=r"no longer validate.*run all configs"):
        update_baseline(path, make_run(("fts", 0.25)))
    assert path.read_bytes() == before

    # Starting the file again from a joint run works.
    path.unlink()
    rows_after = update_baseline(path, make_run(("dense", 0.6)))
    assert rows_after["dense"].golden_set_sha256 == "a" * 64
    assert rows_after["dense"].retrieval_config_hash == retrieval_config("dense").config_hash


def test_update_baseline_accepts_a_joint_run_of_every_config(tmp_path: Path) -> None:
    path = tmp_path / "retrieval.json"
    update_baseline(path, make_run(("dense", 0.5), ("fts", 0.25)))
    joint = make_run(
        ("dense", 0.6),
        ("fts", 0.3),
        golden_set_version="v2",
        golden_set_sha256="b" * 64,
        index_config_hash="e" * 64,
    )
    rows = update_baseline(path, joint)  # nothing is kept, so nothing can disagree
    assert {row.golden_set_sha256 for row in rows.values()} == {"b" * 64}
    assert {row.index_config_hash for row in rows.values()} == {"e" * 64}


def test_update_baseline_copies_the_dirty_flag(tmp_path: Path) -> None:
    rows = update_baseline(tmp_path / "retrieval.json", make_run(("dense", 0.5), git_dirty=True))
    assert rows["dense"].git_dirty is True


def test_golden_set_digest_ignores_line_endings_but_not_content(tmp_path: Path) -> None:
    lf, crlf, edited = tmp_path / "lf.jsonl", tmp_path / "crlf.jsonl", tmp_path / "edited.jsonl"
    lf.write_bytes(b'{"id": "q001"}\n{"id": "q002"}\n')
    crlf.write_bytes(b'{"id": "q001"}\r\n{"id": "q002"}\r\n')
    edited.write_bytes(b'{"id": "q001"}\n{"id": "q003"}\n')

    assert golden_set_digest(lf) == golden_set_digest(crlf)
    assert golden_set_digest(lf) != golden_set_digest(edited)
    assert len(golden_set_digest(lf)) == 64
