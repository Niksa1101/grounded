"""The one place that picks an event loop async psycopg can run on (infra/event_loop.py)."""

from __future__ import annotations

import asyncio
import sys

import pytest

from grounded.infra.event_loop import loop_factory, new_event_loop


def test_windows_gets_the_selector_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")

    assert loop_factory() is asyncio.SelectorEventLoop


def test_other_platforms_keep_the_default_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    assert loop_factory() is None


def test_new_event_loop_is_a_selector_loop_that_runs_a_coroutine() -> None:
    async def value() -> int:
        return 7

    loop = new_event_loop()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert loop.run_until_complete(value()) == 7
    finally:
        loop.close()
