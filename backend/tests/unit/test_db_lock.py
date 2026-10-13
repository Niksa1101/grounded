"""The wait loop of the test-database lock (R.30), driven by a clock the test moves by hand.

Nothing here sleeps or reads the wall clock (AGENTS.md §8). What the lock does against a real
server is in ``tests/integration/test_test_database_lock.py``.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from typing import Any

import pytest

from tests.db_lock import LOCK_POLL_S, LOCK_TIMEOUT_S, lock_test_database, wait_for_lock
from tests.support import FakeClock


class Rig:
    """A ``try_lock`` that answers from a script, a clock that ``sleep`` moves, and the calls."""

    def __init__(self, answers: Iterator[bool]) -> None:
        self.answers = answers
        self.clock = FakeClock()
        self.tries = 0
        self.sleeps: list[float] = []
        self.announced = 0

    def try_lock(self) -> bool:
        self.tries += 1
        return next(self.answers)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.clock.advance(seconds)

    def announce(self) -> None:
        self.announced += 1

    def wait(self, timeout_s: float = 10.0) -> bool:
        return wait_for_lock(
            self.try_lock,
            announce=self.announce,
            sleep=self.sleep,
            now=self.clock,
            timeout_s=timeout_s,
            poll_s=1.0,
        )


def test_a_free_lock_is_taken_at_once_without_a_word() -> None:
    rig = Rig(iter([True]))

    assert rig.wait() is True
    assert (rig.tries, rig.sleeps, rig.announced) == (1, [], 0)


def test_a_held_lock_is_retried_every_poll_and_announced_once() -> None:
    rig = Rig(iter([False, False, False, True]))

    assert rig.wait() is True
    assert rig.tries == 4
    assert rig.sleeps == [1.0, 1.0, 1.0]
    assert rig.announced == 1  # one line, however long the wait


def test_a_lock_that_never_frees_gives_up_at_the_timeout() -> None:
    rig = Rig(iter(lambda: False, True))  # always held

    assert rig.wait(timeout_s=10.0) is False
    assert rig.clock.now == 1000.0 + 10.0  # it waited the whole period, not a second more
    assert rig.tries == 11  # the try at t=0, then one after each of the ten waits
    assert rig.announced == 1


def test_the_real_timeout_is_15_minutes_polled_every_second() -> None:
    assert LOCK_TIMEOUT_S == 15 * 60
    assert LOCK_POLL_S == 1.0


class FakeAdmin:
    """Stands in for the autocommit admin connection: ``execute(sql, params).fetchone()``."""

    def __init__(self, answers: list[bool]) -> None:
        self.answers = answers
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, statement: str, params: tuple[Any, ...] = ()) -> FakeAdmin:
        self.statements.append((statement, params))
        return self

    def fetchone(self) -> tuple[bool]:
        return (self.answers.pop(0),)


def test_the_connection_is_asked_for_a_lock_named_after_the_database() -> None:
    admin = FakeAdmin([True])

    assert lock_test_database(admin, "grounded_test") is True

    [(statement, params)] = admin.statements
    assert "pg_try_advisory_lock(hashtext(%s))" in statement
    assert params == ("grounded_test",)


def test_a_wait_says_which_database_the_other_session_is_using() -> None:
    admin = FakeAdmin([False, True])
    out = io.StringIO()
    clock = FakeClock()

    taken = lock_test_database(admin, "grounded_test", sleep=clock.advance, now=clock, out=out)

    assert taken is True
    assert out.getvalue() == "another pytest session is using grounded_test, waiting...\n"
    assert clock.now == 1000.0 + LOCK_POLL_S


def test_a_wait_that_gives_up_returns_false() -> None:
    admin = FakeAdmin([False] * 2000)
    clock = FakeClock()

    taken = lock_test_database(
        admin, "grounded_test", sleep=clock.advance, now=clock, out=io.StringIO()
    )

    assert taken is False
    assert clock.now == pytest.approx(1000.0 + LOCK_TIMEOUT_S)
