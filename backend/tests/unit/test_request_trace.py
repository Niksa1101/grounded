"""``RequestTrace``: tokens per provider and model, the shadow cost that follows them, and the time
left before the request deadline (7.04). The row and the response read these numbers, so they are
checked here without a database."""

from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import pytest

from grounded.generation.providers.base import Usage
from grounded.infra.timing import StageTimer
from grounded.observability.cost import PricingError, load_pricing
from grounded.observability.request_log import RequestTrace
from tests.support import FakeClock

GEMINI = "gemini-3.5-flash-lite"  # $0.30 in, $2.50 out per 1M tokens
GROQ = "openai/gpt-oss-120b"  # $0.15 in, $0.60 out per 1M tokens


def make_trace(clock: FakeClock | None = None, deadline_s: float | None = None) -> RequestTrace:
    trace = RequestTrace(request_id=uuid4(), timer=StageTimer(clock or FakeClock()))
    trace.deadline_s = deadline_s
    return trace


def test_usage_of_two_providers_is_priced_each_at_its_own_model() -> None:
    trace = make_trace()
    trace.add_usage(Usage(input_tokens=1_000, output_tokens=500), provider="gemini", model=GEMINI)
    trace.add_usage(Usage(input_tokens=1_000, output_tokens=500), provider="gemini", model=GEMINI)
    trace.add_usage(Usage(input_tokens=2_000, output_tokens=300), provider="groq", model=GROQ)

    assert (trace.input_tokens, trace.output_tokens) == (4_000, 1_300)  # the row's totals
    # 2 x (1,000 x 0.30 + 500 x 2.50) / 1e6 for Gemini, (2,000 x 0.15 + 300 x 0.60) / 1e6 for Groq
    assert trace.shadow_cost(load_pricing()) == Decimal("0.00358000")


def test_the_cost_of_one_provider_is_what_it_was_before_the_split() -> None:
    trace = make_trace()
    trace.provider, trace.model = "gemini", GEMINI
    trace.add_usage(Usage(input_tokens=1_000, output_tokens=500))  # no names: the trace's own
    trace.embedding_model, trace.embedding_tokens = "gemini-embedding-001", 28

    assert trace.generation_usage == {("gemini", GEMINI): (1_000, 500)}
    assert trace.shadow_cost(load_pricing()) == Decimal("0.00155420")  # as test_request_log


def test_nothing_billed_costs_only_what_else_ran() -> None:
    trace = make_trace()
    trace.provider, trace.model = "gemini", GEMINI  # named, but no attempt reported usage
    trace.embedding_model, trace.embedding_tokens = "gemini-embedding-001", 28

    assert trace.input_tokens is None
    assert trace.shadow_cost(load_pricing()) == Decimal("0.00000420")


def test_the_stub_provider_is_free() -> None:
    trace = make_trace()
    trace.add_usage(Usage(input_tokens=5, output_tokens=5), provider="fake", model="stub")
    assert trace.shadow_cost(load_pricing()) == Decimal("0.00000000")


def test_an_unpriced_model_that_billed_tokens_is_an_error() -> None:
    trace = make_trace()
    trace.add_usage(Usage(input_tokens=1, output_tokens=1), provider="gemini", model="gemini-x")
    with pytest.raises(PricingError):
        trace.shadow_cost(load_pricing())


def test_time_left_is_the_deadline_minus_the_time_the_request_has_used() -> None:
    clock = FakeClock()
    trace = make_trace(clock, deadline_s=25.0)
    assert trace.time_left_s() == 25.0
    clock.advance(9.5)
    assert trace.time_left_s() == pytest.approx(15.5)
    clock.advance(20.0)
    assert trace.time_left_s() == pytest.approx(-4.5)  # past it: negative, never a reason to raise


def test_without_a_deadline_there_is_no_time_left_to_report() -> None:
    clock = FakeClock()
    trace = make_trace(clock)
    clock.advance(1_000.0)
    assert trace.time_left_s() is None
