"""The promptfoo provider over the real ``AskPipeline`` and a real pgvector index (4.05), and its
result fed to the real assertions: the contract between provider, promptfoo's context and
``promptfoo_asserts``, without promptfoo itself. The unit tests of each side are in
tests/unit/test_promptfoo_*.py."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from grounded.evals import promptfoo_asserts as asserts
from grounded.evals.promptfoo_provider import answer
from grounded.evals.promptfoo_tests import golden_case
from grounded.generation.pipeline import AskMode
from grounded.generation.providers.base import Usage
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.provider_errors import ProviderRateLimited
from grounded.ingest.embed import FakeEmbedder
from grounded.observability.cost import load_pricing
from grounded.runtime import open_runtime
from grounded.schemas.eval import GoldenItem, RelevantSection
from grounded.schemas.llm import LLMAnswer, LLMClaim
from tests.hybrid_corpus import CORPUS, PAGE, insert_page
from tests.support import (
    EMBEDDING_DIM,
    EVAL_SETTINGS,
    insert_index_version,
    make_settings,
)

pytestmark = pytest.mark.integration

QUESTION = "Where does the quokka sleep?"
GEMINI = "gemini-3.5-flash-lite"
PAGE_PATH = PAGE[0]


@pytest.fixture
def index(test_database_url: str) -> Iterator[str]:
    def wipe() -> None:
        with psycopg.connect(test_database_url, autocommit=True) as conn:
            conn.execute("TRUNCATE index_versions, request_logs RESTART IDENTITY CASCADE")

    wipe()
    with psycopg.connect(test_database_url) as conn:
        version = insert_index_version(conn, active=True, config_hash="1" * 64)
        insert_page(conn, version, CORPUS)
    yield test_database_url
    wipe()


def good() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=["c1"], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep. [c1]", claims=[claim])


def uncited() -> LLMAnswer:
    claim = LLMClaim(text="Quokkas sleep.", citation_ids=[], self_confidence=0.9)
    return LLMAnswer(status="answered", answer_markdown="Quokkas sleep.", claims=[claim])


def item(*, section: str, answerable: bool = True) -> GoldenItem:
    return GoldenItem(
        id="q001",
        question=QUESTION,
        type="factual" if answerable else "unanswerable",
        answerable=answerable,
        reference_answer="Somewhere.",
        relevant_sections=[RelevantSection(section=section, grade=2)] if answerable else [],
    )


def promptfoo_context(payload: dict[str, Any], golden: GoldenItem) -> dict[str, Any]:
    """What promptfoo hands an assertion: the provider's metadata and the test's golden payload."""
    golden_payload = golden_case(golden, version="v1", sha256="x")
    return {
        "vars": {"question": golden.question},
        "test": {"metadata": {"golden": golden_payload}},
        "providerResponse": {"output": payload["output"], "metadata": payload["metadata"]},
        "metadata": payload["metadata"],
    }


async def ask(
    url: str, tmp_path: Path, provider: FakeLLMProvider, mode: AskMode
) -> tuple[dict[str, Any], bool]:
    settings = make_settings(database_url=url, cache_dir=tmp_path / "cache", **EVAL_SETTINGS)
    async with open_runtime(
        settings, embedder=FakeEmbedder(dim=EMBEDDING_DIM), provider=provider
    ) as runtime:
        result = await answer(runtime.pipeline, QUESTION, mode, pricing=load_pricing())
    return result.payload, result.stop_run


def fake_gemini(*script: Any) -> FakeLLMProvider:
    usage = Usage(input_tokens=1_000, output_tokens=500, thinking_tokens=50)
    return FakeLLMProvider(script, name="gemini", model=GEMINI, usage=usage)


async def test_hybrid_records_the_retrieval_and_the_assertions_read_it(
    index: str, tmp_path: Path
) -> None:
    payload, stop = await ask(index, tmp_path, fake_gemini(good()), AskMode.HYBRID)

    assert stop is False
    metadata = payload["metadata"]
    assert metadata["mode"] == "hybrid"
    assert metadata["status"] == "answered"
    assert metadata["validation_retries"] == 0
    assert metadata["invalid_citation_count"] == 0
    assert metadata["llm_cache_hits"] == 0
    # The model could cite the chunks that were in its prompt, with the text the DB holds.
    context = metadata["context"]
    assert [c["label"] for c in context] == [f"c{n}" for n in range(1, len(context) + 1)]
    assert all(c["section_id"].startswith(PAGE_PATH + "#") for c in context)
    assert all(c["content"] for c in context)
    assert set(c["section_id"] for c in context) <= set(metadata["retrieved_section_ids"])
    cited = payload["output"]["citations"][0]["chunk_id"]
    assert cited == context[0]["chunk_id"]  # c1
    assert metadata["claims"][0]["chunk_ids"] == [cited]
    assert payload["tokenUsage"]["prompt"] == 1_000

    # The same payload through the four assertions, with a golden item for this page.
    golden = item(section=PAGE_PATH)  # a whole-page label: every chunk of the page matches it
    results = {
        check.__name__: check(payload["output"], promptfoo_context(payload, golden))
        for check in (
            asserts.schema_first_try,
            asserts.citation_validity,
            asserts.refusal_correctness,
            asserts.citation_precision,
        )
    }
    assert {name: r["score"] for name, r in results.items()} == {
        "schema_first_try": 1.0,
        "citation_validity": 1.0,
        "refusal_correctness": 1.0,
        "citation_precision": 1.0,
    }
    assert not any(r["not_applicable"] for r in results.values())


async def test_citation_precision_is_zero_for_a_page_the_item_does_not_label(
    index: str, tmp_path: Path
) -> None:
    payload, _ = await ask(index, tmp_path, fake_gemini(good()), AskMode.HYBRID)

    other = item(section="docs/en/docs/elsewhere.md")
    result = asserts.citation_precision(payload["output"], promptfoo_context(payload, other))

    assert (result["score"], result["pass"], result["not_applicable"]) == (0.0, False, False)


async def test_no_rag_needs_no_index_and_two_metrics_are_not_applicable(
    index: str, tmp_path: Path
) -> None:
    with psycopg.connect(index, autocommit=True) as conn:
        conn.execute("TRUNCATE index_versions RESTART IDENTITY CASCADE")  # nothing to retrieve from

    payload, stop = await ask(index, tmp_path, fake_gemini(uncited()), AskMode.NO_RAG)

    assert stop is False
    metadata = payload["metadata"]
    assert metadata["mode"] == "no_rag"
    assert metadata["context"] == []
    assert metadata["retrieved_section_ids"] == []
    assert metadata["index_version"] == "none"
    context = promptfoo_context(payload, item(section=PAGE_PATH))
    assert asserts.schema_first_try(payload["output"], context)["score"] == 1.0
    assert asserts.refusal_correctness(payload["output"], context)["score"] == 1.0
    assert asserts.citation_validity(payload["output"], context)["not_applicable"] is True
    assert asserts.citation_precision(payload["output"], context)["not_applicable"] is True


async def test_a_daily_quota_becomes_a_tagged_error_that_stops_the_run(
    index: str, tmp_path: Path
) -> None:
    quota = ProviderRateLimited("daily quota exhausted", retry_after_s=None, is_quota=True)

    payload, stop = await ask(index, tmp_path, fake_gemini(quota), AskMode.HYBRID)

    assert stop is True
    assert payload["error"] == "[ProviderRateLimited quota=true] daily quota exhausted"
    assert payload["metadata"]["error_kind"] == "ProviderRateLimited"
    assert payload["metadata"]["is_quota"] is True
    assert "output" not in payload
    # The embedding was done before the generator failed, and it was billed on paper.
    assert payload["cost"] > 0
