"""``request_deadline`` (5.08): only its own expiry is the deadline."""

from __future__ import annotations

import asyncio

import pytest

from grounded.api.deadline import DeadlineExceededError, request_deadline

# Real (few-millisecond) time: the helper is a thin layer over asyncio.timeout, and a fake loop
# clock would test the fake. The waits below are the only sleeps and end by cancellation.
SHORT_S = 0.02


async def test_work_that_outlives_the_deadline_is_cancelled_and_reported() -> None:
    cancelled = False

    async def work() -> None:
        async with request_deadline(SHORT_S):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                nonlocal cancelled
                cancelled = True
                raise

    with pytest.raises(DeadlineExceededError) as caught:
        await work()
    assert cancelled  # the work was stopped where it stood, not left running
    assert caught.value.deadline_s == SHORT_S


async def test_work_that_finishes_in_time_is_untouched() -> None:
    async with request_deadline(5.0):
        await asyncio.sleep(0)


async def test_no_deadline_means_no_limit() -> None:
    async with request_deadline(None):
        await asyncio.sleep(SHORT_S * 2)  # longer than the deadline the other tests use


async def test_a_timeout_raised_by_the_work_is_not_the_deadline() -> None:
    # A stage's own timeout is a TimeoutError too; it must reach the error handlers unchanged.
    with pytest.raises(TimeoutError) as caught:
        async with request_deadline(5.0):
            raise TimeoutError("a stage gave up")
    assert not isinstance(caught.value, DeadlineExceededError)


async def test_other_errors_pass_through() -> None:
    with pytest.raises(ValueError, match="boom"):
        async with request_deadline(5.0):
            raise ValueError("boom")
