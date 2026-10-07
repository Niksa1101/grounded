"""`grounded ask` end to end: real pgvector index, fake embedder, stub provider."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner

import grounded.cli
import grounded.retrieval.query_embedding
import grounded.runtime
from grounded.cli import app
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import ProviderRateLimited
from grounded.ingest.embed import FakeEmbedder, TokenCounter
from grounded.schemas.api import AskResponse
from grounded.schemas.llm import LLMAnswer, LLMClaim
from grounded.settings import Settings
from tests.hybrid_corpus import CORPUS, insert_page
from tests.support import (
    EMBEDDING_DIM,
    insert_index_version,
    make_settings,
    pricing_for_fake_embedder,
)

pytestmark = pytest.mark.integration

runner = CliRunner()


class StubGemini(FakeEmbedder):
    """Stands in for the request-path GeminiEmbedder, so no key and no network are needed."""

    @classmethod
    def for_request_path(cls, settings: Settings, count_tokens: TokenCounter) -> StubGemini:
        return cls(model=settings.embedding_model, dim=settings.embedding_dim)


@pytest.fixture
def settings(
    test_database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Settings]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    settings = make_settings(
        database_url=test_database_url,
        cache_dir=tmp_path / "cache",
        embedding_model="fake-embedding",
        embedding_dim=EMBEDDING_DIM,
    )
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: settings)
    monkeypatch.setattr(grounded.retrieval.query_embedding, "GeminiEmbedder", StubGemini)
    monkeypatch.setattr(grounded.runtime, "load_pricing", pricing_for_fake_embedder)
    yield settings
    wipe()


def test_ask_prints_a_valid_response_with_the_fake_provider(settings: Settings) -> None:
    result = runner.invoke(app, ["ask", "Where does the quokka sleep?", "--fake"])
    assert result.exit_code == 0, result.output
    body = AskResponse.model_validate(json.loads(result.stdout))
    assert body.status == "answered"
    assert body.meta.provider == "fake"
    assert [c.n for c in body.citations] == [1]
    assert body.answer_markdown.endswith("[1]")


def test_ask_without_a_gemini_key_fails_cleanly(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Whatever the environment (a CI secret, a developer .env) says, this run has no key.
    unset = settings.model_copy(update={"gemini_api_key": None, "gemini_model": None})
    monkeypatch.setattr(grounded.cli, "get_settings", lambda: unset)
    result = runner.invoke(app, ["ask", "Where does the quokka sleep?"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY must be set" in result.output


def test_ask_rejects_a_question_that_is_too_short(settings: Settings) -> None:
    result = runner.invoke(app, ["ask", "hi", "--fake"])
    assert result.exit_code == 1
    assert "Invalid question" in result.output


def test_ask_no_rag_needs_no_embedding_and_cites_nothing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("no_rag must not build an embedder")

    monkeypatch.setattr(grounded.retrieval.query_embedding, "GeminiEmbedder", fail)
    result = runner.invoke(
        app, ["ask", "Where does the quokka sleep?", "--fake", "--mode", "no_rag"]
    )
    assert result.exit_code == 0, result.output
    body = AskResponse.model_validate(json.loads(result.stdout))
    assert body.status == "answered"
    assert body.citations == []
    assert all(claim.citations == [] for claim in body.claims)
    assert body.meta.prompt_version.startswith("answer_no_rag_v1@")
    assert body.meta.index_version == "none"
    assert "[c1]" not in body.answer_markdown


def test_ask_rejects_an_unknown_mode(settings: Settings) -> None:
    result = runner.invoke(
        app, ["ask", "Where does the quokka sleep?", "--fake", "--mode", "dense"]
    )
    assert result.exit_code != 0


# --- ask --golden (3.14) -----------------------------------------------------------------------

BATCH_QUESTIONS = [
    "Where does the quokka sleep, exactly?",
    "What does the wombat dig, usually?",
]


def write_golden(tmp_path: Path, questions: list[str] = BATCH_QUESTIONS) -> Path:
    rows = [
        {
            "id": f"q{n:03d}",
            "question": question,
            "type": "factual",
            "answerable": True,
            "reference_answer": "An answer.",
            "relevant_sections": [{"section": "docs/en/docs/hybrid.md", "grade": 2}],
        }
        for n, question in enumerate(questions, start=1)
    ]
    path = tmp_path / "golden_set.v1.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def cited_answer() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[claim])


def test_golden_batch_answers_every_question_and_summarizes(
    settings: Settings, tmp_path: Path
) -> None:
    golden = write_golden(tmp_path)
    before = golden.read_bytes()
    out = tmp_path / "ask.json"
    result = runner.invoke(app, ["ask", "--golden", str(golden), "--out", str(out), "--fake"])
    assert result.exit_code == 0, result.output
    assert "Schema-valid AskResponse: 2/2" in result.stdout
    assert "Status: answered 2" in result.stdout
    assert "q001 answered" in result.stdout
    assert "q002 answered" in result.stdout
    # Ids are printed, never the question text (AGENTS.md §6.13).
    assert not any(question in result.output for question in BATCH_QUESTIONS)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["fake_provider"] is True
    assert written["golden_set_version"] == "v1"
    assert [r["id"] for r in written["results"]] == ["q001", "q002"]
    for row in written["results"]:
        AskResponse.model_validate(row["response"])
    assert not any(question in out.read_text(encoding="utf-8") for question in BATCH_QUESTIONS)
    assert golden.read_bytes() == before  # the golden file is only read


def test_golden_batch_second_run_is_served_from_the_answer_cache(
    settings: Settings, tmp_path: Path
) -> None:
    golden = write_golden(tmp_path)
    args = ["ask", "--golden", str(golden), "--out", str(tmp_path / "ask.json"), "--fake"]
    assert runner.invoke(app, args).exit_code == 0
    second = runner.invoke(app, args)
    assert second.exit_code == 0, second.output
    assert "Cache hits: 2;" in second.stdout


def test_golden_batch_limit_asks_only_the_first_questions(
    settings: Settings, tmp_path: Path
) -> None:
    golden = write_golden(tmp_path)
    result = runner.invoke(
        app,
        [
            "ask",
            "--golden",
            str(golden),
            "--out",
            str(tmp_path / "a.json"),
            "--limit",
            "1",
            "--fake",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "q002" not in result.stdout
    assert "Schema-valid AskResponse: 1/1" in result.stdout


def test_golden_batch_reports_failures_and_exits_non_zero(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # q001: invalid JSON twice (the pipeline's one retry fails too); q002 is fine.
    script = ["not json", "not json", cited_answer()]
    monkeypatch.setattr(grounded.cli, "StubLLMProvider", lambda: FakeLLMProvider(script))
    out = tmp_path / "ask.json"
    result = runner.invoke(
        app, ["ask", "--golden", str(write_golden(tmp_path)), "--out", str(out), "--fake"]
    )
    assert result.exit_code == 1
    assert "Schema-valid AskResponse: 1/2" in result.stdout
    assert "Validation retries: 1" in result.stdout
    assert "Failed q001: ProviderBadOutput" in result.stdout
    rows = json.loads(out.read_text(encoding="utf-8"))["results"]
    assert [(r["id"], r["outcome"]) for r in rows] == [("q001", "failed"), ("q002", "ok")]


def test_golden_batch_waits_out_a_per_minute_429_and_asks_again(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    limited = ProviderRateLimited("slow down", retry_after_s=0.0, is_quota=False)
    provider = FakeLLMProvider([limited, cited_answer(), cited_answer()])
    monkeypatch.setattr(grounded.cli, "StubLLMProvider", lambda: provider)
    result = runner.invoke(
        app,
        [
            "ask",
            "--golden",
            str(write_golden(tmp_path)),
            "--out",
            str(tmp_path / "a.json"),
            "--fake",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "rate-limit waits: 1" in result.stdout
    assert len(provider.calls) == 3
    assert provider.remaining == 0


def test_golden_batch_stops_on_a_daily_quota_and_keeps_the_partial_results(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quota = ProviderRateLimited("daily quota", retry_after_s=None, is_quota=True)
    provider = FakeLLMProvider([cited_answer(), quota])
    monkeypatch.setattr(grounded.cli, "StubLLMProvider", lambda: provider)
    out = tmp_path / "ask.json"
    questions = [*BATCH_QUESTIONS, "Is there a third question?"]
    result = runner.invoke(
        app,
        ["ask", "--golden", str(write_golden(tmp_path, questions)), "--out", str(out), "--fake"],
    )
    assert result.exit_code == 1
    assert "STOPPED EARLY: daily quota exhausted at q002" in result.stdout
    assert "q003" not in result.stdout  # never asked
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["summary"]["not_run"] == 1
    assert len(provider.calls) == 2


def test_golden_batch_refuses_a_question_and_a_golden_file_together(
    settings: Settings, tmp_path: Path
) -> None:
    result = runner.invoke(
        app, ["ask", "Where does the quokka sleep?", "--golden", str(write_golden(tmp_path))]
    )
    assert result.exit_code == 1
    assert "either QUESTION or --golden" in result.output


def test_golden_batch_never_overwrites_the_golden_file(settings: Settings, tmp_path: Path) -> None:
    golden = write_golden(tmp_path)
    before = golden.read_bytes()
    result = runner.invoke(app, ["ask", "--golden", str(golden), "--out", str(golden), "--fake"])
    assert result.exit_code == 1
    assert "--out must not be the golden-set file" in result.output
    assert golden.read_bytes() == before


def test_ask_without_a_question_or_a_golden_file_fails(settings: Settings) -> None:
    result = runner.invoke(app, ["ask", "--fake"])
    assert result.exit_code == 1
    assert "Give a QUESTION" in result.output
