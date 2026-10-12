"""``hybrid_rerank`` in the retrieval eval (6.04) without a database: the config, the stop on a
reranker error and the numbers that go into the results file. The pipeline from the database to the
metrics is exercised end to end in tests/integration/test_cli_eval.py."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from grounded.evals.metrics import ChunkRef
from grounded.evals.retrieval_runner import (
    RerankRunError,
    RetrievalEvalError,
    _percentile,
    evaluate,
    retrieval_configs,
    run_retrieval_eval,
)
from grounded.infra.provider_errors import (
    ProviderBadOutput,
    ProviderError,
    ProviderRateLimited,
    ProviderRequestRejected,
    ProviderTimeout,
    ProviderUnavailable,
)
from grounded.ingest.embed import FakeEmbedder, Vector
from grounded.retrieval.config import RetrievalConfig, RetrievalMode
from grounded.schemas.eval import GoldenItem, RerankRunStats, RetrievalConfigResult
from tests.support import make_settings

PAGE = "docs/en/docs/tutorial/p.md"
MODEL = "rerank-v4.0-pro"


@dataclass(frozen=True)
class Ref:
    section_id: str
    anchor_path: tuple[str, ...]


def item(id: str) -> GoldenItem:
    return GoldenItem.model_validate(
        {
            "id": id,
            "question": f"question {id}?",
            "type": "how_to",
            "answerable": True,
            "reference_answer": "…",
            "relevant_sections": [{"section": f"{PAGE}#a", "grade": 2}],
        }
    )


def config(mode: RetrievalMode = "hybrid_rerank", **overrides: Any) -> RetrievalConfig:
    fields: dict[str, Any] = {
        "mode": mode,
        "k_dense": 20,
        "k_fts": 20,
        "k_fused": 40,
        "k_context": 5,
        "rrf_k": 60,
    }
    if mode == "hybrid_rerank":
        fields |= {"rerank_provider": "cohere", "rerank_model": MODEL}
    return RetrievalConfig.model_validate(fields | overrides)


# --- The config ----------------------------------------------------------------------------------


def test_hybrid_rerank_needs_a_provider_and_a_model() -> None:
    assert config().mode == "hybrid_rerank"
    overrides_list: list[dict[str, Any]] = [
        {"rerank_provider": "none", "rerank_model": None},
        {"rerank_model": None},
        {"rerank_model": ""},
    ]
    for overrides in overrides_list:
        with pytest.raises(ValidationError, match="hybrid_rerank needs"):
            config(**overrides)


def test_hybrid_rerank_has_a_hash_of_its_own() -> None:
    plain = config("hybrid")
    reranked = config("hybrid_rerank")
    on_hybrid = config("hybrid", rerank_provider="cohere", rerank_model=MODEL)  # the request path

    assert len({plain.config_hash, reranked.config_hash, on_hybrid.config_hash}) == 3


def test_a_plain_mode_is_scored_without_the_rerank_fields_even_when_rerank_is_on() -> None:
    settings = make_settings(rerank_provider="cohere", rerank_model=MODEL, cohere_api_key="k")

    [dense, hybrid, rerank] = retrieval_configs(settings, ["dense", "hybrid", "hybrid_rerank"])

    # The committed `hybrid` row was made with no reranker; a "hybrid" row of a run that has one in
    # its `.env` must not claim one (and must hash like the committed row).
    assert hybrid == config("hybrid") == RetrievalConfig.from_settings(make_settings(), "hybrid")
    assert dense.rerank_provider == "none"
    assert dense.rerank_model is None
    assert (rerank.mode, rerank.rerank_provider, rerank.rerank_model) == (
        "hybrid_rerank",
        "cohere",
        MODEL,
    )


def test_hybrid_rerank_without_a_provider_in_the_settings_is_refused_not_downgraded() -> None:
    with pytest.raises(ValidationError, match="hybrid_rerank needs"):
        retrieval_configs(make_settings(), ["hybrid_rerank"])


# --- A reranker error is not a result ------------------------------------------------------------


class Scripted:
    """A search that answers per question, or raises the error scripted for it."""

    def __init__(self, outcomes: dict[str, list[Ref] | ProviderError]) -> None:
        self.outcomes = outcomes
        self.asked: list[str] = []

    async def __call__(self, question: str, vector: Vector) -> Sequence[ChunkRef]:
        self.asked.append(question)
        outcome = self.outcomes[question]
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


RANKED = [Ref(f"{PAGE}#a", ("a",))]


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ProviderRateLimited("slow down", retry_after_s=30.0, is_quota=False), "rate_limited"),
        (ProviderTimeout("3s"), "timeout"),
        (ProviderUnavailable("503"), "unavailable"),
        (ProviderRequestRejected("no such model", status_code=404), "rejected"),
        (ProviderBadOutput("not the documented shape"), "bad_output"),
    ],
)
async def test_the_first_reranker_error_ends_the_run_at_that_question(
    error: ProviderError, code: str
) -> None:
    search = Scripted({"question q001?": RANKED, "question q002?": error, "question q003?": RANKED})

    with pytest.raises(RerankRunError) as stopped:
        await evaluate(
            [item("q001"), item("q002"), item("q003")],
            config=config(),
            k=40,
            embedder=FakeEmbedder(dim=8),
            search=search,
        )

    assert stopped.value.question_id == "q002"
    assert stopped.value.code == code
    assert stopped.value.error is error
    assert f"q002 ({code}, {type(error).__name__})" in str(stopped.value)
    assert "question q002" not in str(stopped.value)  # the id, never the text
    assert search.asked == ["question q001?", "question q002?"]  # q003 was never tried


async def test_an_error_stops_the_run_and_nothing_is_skipped_or_replaced() -> None:
    # No result for a half run exists: evaluate raises, it does not return the other questions.
    search = Scripted({"question q001?": ProviderTimeout("3s")})

    with pytest.raises(RetrievalEvalError):  # RerankRunError is one
        await evaluate(
            [item("q001")], config=config(), k=40, embedder=FakeEmbedder(dim=8), search=search
        )


async def test_a_provider_error_of_another_mode_is_not_turned_into_a_rerank_stop() -> None:
    boom = ProviderUnavailable("not the reranker")
    search = Scripted({"question q001?": boom})

    with pytest.raises(ProviderUnavailable):
        await evaluate(
            [item("q001")],
            config=config("hybrid"),
            k=40,
            embedder=FakeEmbedder(dim=8),
            search=search,
        )


async def test_a_run_that_names_hybrid_rerank_without_a_reranker_fails_before_any_work() -> None:
    embedder = FakeEmbedder(dim=8)

    with pytest.raises(RetrievalEvalError, match="RERANK_PROVIDER=cohere"):
        await run_retrieval_eval(
            "postgresql://nobody@unreachable.invalid/none",  # never connected to
            [item("q001")],
            configs=[config()],
            embedder=embedder,
            golden_set_version="v1",
            golden_set_sha256="a" * 64,
            now=datetime(2026, 10, 12, tzinfo=UTC),
        )

    assert embedder.calls == []


# --- The numbers ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "p50", "p95"),
    [
        ([], None, None),
        ([7.0], 7.0, 7.0),
        ([40.0, 10.0, 30.0, 20.0], 20.0, 40.0),  # 2nd and 4th (ceil of 2 and 3.8)
        (list(map(float, range(1, 26))), 13.0, 24.0),  # 13th and 24th (ceil of 12.5, 23.75)
    ],
)
def test_percentiles_are_nearest_rank(
    values: list[float], p50: float | None, p95: float | None
) -> None:
    assert (_percentile(values, 50), _percentile(values, 95)) == (p50, p95)


def test_a_results_row_without_rerank_stats_is_valid_and_has_none() -> None:
    plain = config("hybrid")
    result = RetrievalConfigResult(
        config="hybrid",
        retrieval_config=plain,
        retrieval_config_hash=plain.config_hash,
        k=40,
        n=1,
        skipped_unanswerable=0,
        metrics={},
        questions=[],
    )
    assert result.rerank is None

    stats = RerankRunStats(model=MODEL, live_calls=25, cache_hits=0)
    assert (stats.latency_p50_ms, stats.latency_p95_ms) == (None, None)
