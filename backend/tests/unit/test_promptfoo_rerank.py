"""The promptfoo side of ``hybrid_rerank`` (6.04): the provider mode, its refusal to be mixed up
with ``hybrid``, the metadata that lets a baseline writer see a degraded case, and the second
config file that only the manual command uses (CI keeps running the first one, unchanged)."""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from grounded.evals import promptfoo_provider as provider
from grounded.evals.promptfoo_provider import PipelineRunner, call_api, mode_of
from grounded.generation.pipeline import AskMode
from grounded.infra.provider_errors import ProviderUnavailable
from grounded.observability.cost import load_pricing
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.rerank import RerankErrorCode
from grounded.runtime import ProviderConfigError

REPO = Path(__file__).resolve().parents[3]
PROMPTFOO_DIR = REPO / "eval" / "promptfoo"
RERANK_CONFIG = (PROMPTFOO_DIR / "promptfooconfig.rerank.yaml").read_text(encoding="utf-8")
MAIN_CONFIG = (PROMPTFOO_DIR / "promptfooconfig.yaml").read_text(encoding="utf-8")
PRICING = load_pricing()


# --- The provider mode ---------------------------------------------------------------------------


def test_hybrid_rerank_is_the_hybrid_pipeline_mode() -> None:
    options = {"id": "file://provider.py", "config": {"mode": "hybrid_rerank"}}

    assert mode_of(options) is AskMode.HYBRID


def test_an_unknown_mode_names_the_three_that_exist() -> None:
    with pytest.raises(ValueError, match=r"must be one of hybrid, no_rag, hybrid_rerank, got 'x'"):
        mode_of({"config": {"mode": "x"}})


class Recording:
    """An asker that degrades or fails the way the pipeline would, and says what it was asked."""

    def __init__(self, *, used: bool, error: RerankErrorCode | None) -> None:
        self.used = used
        self.error: RerankErrorCode | None = error
        self.questions: list[str] = []

    async def ask(
        self,
        question: str,
        request_id: UUID,
        *,
        mode: AskMode = AskMode.HYBRID,
        trace: RequestTrace | None = None,
    ) -> Any:
        assert trace is not None
        self.questions.append(question)
        trace.rerank_used = self.used
        trace.rerank_error = self.error
        raise ProviderUnavailable("503")


def runner_over(asker: Recording, *, rerank_on: bool) -> tuple[PipelineRunner, list[int]]:
    opened: list[int] = []

    @asynccontextmanager
    async def open_it() -> AsyncGenerator[Recording]:
        opened.append(1)
        yield asker

    return PipelineRunner(open_it, pricing=PRICING, rerank_on=rerank_on), opened


def test_a_rerank_entry_needs_the_reranker_and_never_becomes_a_hybrid_run() -> None:
    asker = Recording(used=False, error=None)
    runner, opened = runner_over(asker, rerank_on=False)

    with pytest.raises(ProviderConfigError, match=r"hybrid_rerank needs RERANK_PROVIDER=cohere"):
        runner.call("Q?", AskMode.HYBRID, rerank=True)

    assert opened == []  # refused before anything was opened or asked
    assert asker.questions == []


def test_a_hybrid_entry_refuses_to_run_with_the_reranker_on() -> None:
    asker = Recording(used=False, error=None)
    runner, opened = runner_over(asker, rerank_on=True)

    with pytest.raises(ProviderConfigError, match="answered with the reranker"):
        runner.call("Q?", AskMode.HYBRID)

    assert (opened, asker.questions) == ([], [])


def test_no_rag_does_not_care_and_the_matching_setups_run() -> None:
    for on, rerank, mode in (
        (True, False, AskMode.NO_RAG),
        (False, False, AskMode.NO_RAG),
        (True, True, AskMode.HYBRID),
        (False, False, AskMode.HYBRID),
    ):
        asker = Recording(used=False, error=None)
        runner, opened = runner_over(asker, rerank_on=on)
        try:
            payload = runner.call("Q?", mode, rerank=rerank)
        finally:
            runner.close()

        assert opened == [1]
        assert asker.questions == ["Q?"]
        assert payload["metadata"]["error_kind"] == "ProviderUnavailable"


@pytest.mark.parametrize(
    ("used", "error"), [(True, None), (False, "timeout"), (False, "rate_limited"), (False, None)]
)
def test_every_result_carries_what_the_rerank_stage_did(
    used: bool, error: RerankErrorCode | None
) -> None:
    runner, _ = runner_over(Recording(used=used, error=error), rerank_on=True)
    try:
        payload = runner.call("Q?", AskMode.HYBRID, rerank=True)
    finally:
        runner.close()

    metadata = payload["metadata"]
    assert (metadata["rerank_used"], metadata["rerank_error"]) == (used, error)
    assert metadata["mode"] == "hybrid"  # the pipeline mode; the label is the config's


def test_call_api_tells_the_runner_which_entry_it_is_for(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, AskMode, bool]] = []

    class Stub:
        def call(self, question: str, mode: AskMode, *, rerank: bool = False) -> dict[str, Any]:
            seen.append((question, mode, rerank))
            return {}

    monkeypatch.setattr(provider, "default_runner", lambda: Stub())
    for mode in ("no_rag", "hybrid", "hybrid_rerank"):
        call_api("Q?", {"config": {"mode": mode}}, {})

    assert seen == [
        ("Q?", AskMode.NO_RAG, False),
        ("Q?", AskMode.HYBRID, False),
        ("Q?", AskMode.HYBRID, True),
    ]


# --- The second config file ----------------------------------------------------------------------


def test_the_rerank_config_has_the_one_provider_the_manual_run_needs() -> None:
    assert re.findall(r"label: (\w+)\s+config:\s+mode: (\w+)", RERANK_CONFIG) == [
        ("hybrid_rerank", "hybrid_rerank")
    ]
    assert re.findall(r"id: file://(\w+\.py)", RERANK_CONFIG) == ["provider.py"]
    assert (PROMPTFOO_DIR / "provider.py").is_file()


def test_the_rerank_config_is_graded_exactly_like_the_main_one() -> None:
    marker = "# Test cases come from the golden set"

    # Same tests, same flags (concurrency 1, no promptfoo cache) and the same six assertions: the
    # whole tail of the file is shared, so a change to one cannot be forgotten in the other.
    assert RERANK_CONFIG[RERANK_CONFIG.index(marker) :] == MAIN_CONFIG[MAIN_CONFIG.index(marker) :]


def test_the_config_that_ci_runs_has_no_rerank_provider() -> None:
    assert "rerank" not in MAIN_CONFIG.lower()
    assert re.findall(r"label: (\w+)", MAIN_CONFIG) == ["no_rag", "hybrid"]


def test_no_workflow_or_action_reads_the_rerank_config_or_holds_a_cohere_key() -> None:
    # PRD D55: hybrid_rerank is run by hand. Quota, key and cache stay out of CI.
    for path in (REPO / ".github").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "promptfooconfig.rerank" not in text, path
            assert "COHERE" not in text.upper(), path
            assert "hybrid_rerank" not in text, path
