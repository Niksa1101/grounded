"""Per-request stage timing on a monotonic clock (Tech.md §6, §14).

``StageTimer`` starts counting when it is created, so ``total_ms`` covers everything the request did
since then (validation, cache lookup, the stages, building the response). ``stage(name)`` adds the
time spent inside its ``with`` block to that stage, **also when the block raises**: a failed LLM
call still shows how long it took, which is the row you want when debugging a timeout. A stage
entered twice (retrieval has an index lookup and the search) accumulates. A stage that never ran
reports ``None``, not 0, so skipped work (no rerank, ``no_rag``) stays distinguishable from fast
work.

The clock is injected so tests drive time by hand instead of relying on the wall clock
(AGENTS.md §8). Totals are rounded per value, so the stages need not sum to the total.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Literal

type Clock = Callable[[], float]  # seconds, monotonic

Stage = Literal["embed", "retrieval", "rerank", "llm"]


class StageTimer:
    def __init__(self, clock: Clock = time.perf_counter) -> None:
        self._clock = clock
        self._started = clock()
        self._spent: dict[Stage, float] = {}

    @contextmanager
    def stage(self, name: Stage) -> Generator[None]:
        entered = self._clock()
        try:
            yield
        finally:
            self._spent[name] = self._spent.get(name, 0.0) + (self._clock() - entered)

    def total_ms(self) -> int:
        return _ms(self._clock() - self._started)

    def stage_ms(self, name: Stage) -> int | None:
        spent = self._spent.get(name)
        return None if spent is None else _ms(spent)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)
