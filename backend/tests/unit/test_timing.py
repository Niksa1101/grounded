"""``StageTimer`` on a hand-driven clock (AGENTS.md §8: no wall-clock timing)."""

from __future__ import annotations

import pytest

from grounded.infra.timing import StageTimer
from tests.support import FakeClock


def test_a_stage_measures_the_time_spent_inside_its_block() -> None:
    clock = FakeClock()
    timer = StageTimer(clock)
    clock.advance(0.010)  # outside any stage
    with timer.stage("embed"):
        clock.advance(0.250)
    with timer.stage("llm"):
        clock.advance(1.5)
    assert timer.stage_ms("embed") == 250
    assert timer.stage_ms("llm") == 1500
    assert timer.total_ms() == 1760  # the total also counts the time between stages


def test_a_stage_that_never_ran_is_none_not_zero() -> None:
    timer = StageTimer(FakeClock())
    assert timer.stage_ms("rerank") is None


def test_entering_a_stage_twice_accumulates() -> None:
    clock = FakeClock()
    timer = StageTimer(clock)
    for seconds in (0.020, 0.100):  # the index lookup, then the search
        with timer.stage("retrieval"):
            clock.advance(seconds)
    assert timer.stage_ms("retrieval") == 120


def test_a_stage_that_raises_still_records_its_time() -> None:
    clock = FakeClock()
    timer = StageTimer(clock)

    def fail_after_300_ms() -> None:
        with timer.stage("llm"):
            clock.advance(0.300)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        fail_after_300_ms()
    assert timer.stage_ms("llm") == 300


def test_the_total_keeps_growing_until_it_is_read() -> None:
    clock = FakeClock()
    timer = StageTimer(clock)
    assert timer.total_ms() == 0
    clock.advance(2.0004)
    assert timer.total_ms() == 2000
