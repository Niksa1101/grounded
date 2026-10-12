"""What the pipeline refuses at construction, and what a trace charges, when rerank is on (6.03).

No database: the pool is built and never opened. The request path itself (order, degradation, cap,
log rows) is in tests/integration/test_rerank_path.py.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from uuid import uuid4

import pytest

from grounded.generation.pipeline import AskPipeline
from grounded.generation.prompts import load_answer_prompt, load_no_rag_prompt
from grounded.generation.providers.fake import FakeLLMProvider
from grounded.infra.db import create_pool
from grounded.infra.timing import StageTimer
from grounded.ingest.embed import FakeEmbedder
from grounded.observability.cost import Pricing, PricingError, load_pricing
from grounded.observability.request_log import RequestTrace
from grounded.retrieval.index import ActiveIndexCache
from grounded.retrieval.rerank import FakeReranker
from grounded.settings import Settings
from tests.support import FakeClock, make_settings

PRO = "rerank-v4.0-pro"


def cohere_settings(**overrides: Any) -> Settings:
    return make_settings(**{"rerank_provider": "cohere", "rerank_model": PRO, **overrides})


def pipeline(
    settings: Settings, *, reranker: FakeReranker | None, pricing: Pricing | None = None
) -> AskPipeline:
    pool = create_pool(settings)  # built, never opened: construction does not touch the database
    return AskPipeline(
        settings=settings,
        pool=pool,
        index_cache=ActiveIndexCache(pool, ttl_s=settings.active_index_ttl_s),
        embedder=FakeEmbedder(),
        provider=FakeLLMProvider(),
        prompt=load_answer_prompt(),
        no_rag_prompt=load_no_rag_prompt(),
        pricing=pricing or load_pricing(),
        clock=FakeClock(),
        reranker=reranker,
    )


def test_a_reranker_with_a_priced_model_is_accepted() -> None:
    pipeline(cohere_settings(), reranker=FakeReranker())


def test_no_reranker_is_the_default_and_needs_no_rerank_price() -> None:
    # RERANK_PROVIDER=none: nothing is required of pricing.toml's rerank rows.
    bare = load_pricing().model_copy(update={"rerank": {}})
    pipeline(make_settings(), reranker=None, pricing=bare)


def test_an_unpriced_rerank_model_is_a_startup_error() -> None:
    settings = cohere_settings(rerank_model="rerank-not-in-the-file")
    with pytest.raises(PricingError, match="rerank model 'rerank-not-in-the-file'"):
        pipeline(settings, reranker=FakeReranker())


def test_an_unverified_rerank_price_is_a_startup_error() -> None:
    real = load_pricing()
    unverified = real.model_copy(
        update={
            "rerank": {
                **real.rerank,
                PRO: real.rerank[PRO].model_copy(update={"status": "unverified"}),
            }
        }
    )
    with pytest.raises(PricingError, match="unverified"):
        pipeline(cohere_settings(), reranker=FakeReranker(), pricing=unverified)


@pytest.mark.parametrize("model", [None, "", "   "])
def test_a_reranker_without_a_model_is_refused(model: str | None) -> None:
    with pytest.raises(ValueError, match="RERANK_MODEL"):
        pipeline(
            make_settings(rerank_provider="cohere", rerank_model=model), reranker=FakeReranker()
        )


def test_the_reranker_and_the_provider_setting_must_agree() -> None:
    # The config hash (and so the answer-cache key) says whether the order was reranked: a
    # pipeline that disagrees with it would file answers under the wrong key.
    with pytest.raises(ValueError, match="RERANK_PROVIDER=none, reranker passed"):
        pipeline(make_settings(), reranker=FakeReranker())
    with pytest.raises(ValueError, match="RERANK_PROVIDER=cohere, reranker missing"):
        pipeline(cohere_settings(), reranker=None)


# --- what a trace charges ----------------------------------------------------------------------


def trace() -> RequestTrace:
    return RequestTrace(request_id=uuid4(), timer=StageTimer(FakeClock()))


def test_a_trace_charges_every_attempted_rerank_call_at_the_list_price() -> None:
    t = trace()
    t.rerank_model, t.rerank_calls = PRO, 1
    assert t.shadow_cost(load_pricing()) == Decimal("0.00250000")  # 1 / 1000 * 2.50


def test_a_trace_with_no_attempted_call_charges_nothing_for_rerank() -> None:
    # The cap was reached, the reserve failed, or rerank is off: the model may be set, no call ran.
    t = trace()
    t.rerank_model, t.rerank_calls = PRO, 0
    assert t.shadow_cost(load_pricing()) == Decimal("0.00000000")


def test_the_rerank_fields_default_to_a_request_that_did_not_rerank() -> None:
    t = trace()
    assert (t.rerank_used, t.rerank_error, t.rerank_calls, t.rerank_model) == (
        False,
        None,
        0,
        None,
    )
