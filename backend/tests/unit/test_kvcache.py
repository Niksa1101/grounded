from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import pytest

from grounded.infra.kvcache import KVCache


@pytest.fixture
def cache(tmp_path: Path) -> KVCache:
    return KVCache(tmp_path / "cache.sqlite")


def test_round_trip_and_missing_keys_are_absent(cache: KVCache) -> None:
    cache.put_many({"a": b"1", "b": b"\x00\xff"})
    assert cache.get_many(["a", "b", "nope"]) == {"a": b"1", "b": b"\x00\xff"}
    assert cache.get_many([]) == {}
    assert len(cache) == 2


def test_first_value_wins(cache: KVCache) -> None:
    # Keys are content-addressed, so a rewrite of the same key is a no-op, not an update.
    cache.put_many({"a": b"first"})
    cache.put_many({"a": b"second"})
    assert cache.get_many(["a"]) == {"a": b"first"}


def test_lookup_spans_several_queries(cache: KVCache) -> None:
    # More keys than one SELECT binds (500), including across the batch boundaries.
    items = {f"k{i}": str(i).encode() for i in range(1201)}
    cache.put_many(items)
    assert cache.get_many([*items, "missing"]) == items


def test_persists_across_reopen_and_creates_parent_dir(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "cache.sqlite"
    with KVCache(path) as first:
        first.put_many({"a": b"1"})
    with KVCache(path) as second:
        assert second.get_many(["a"]) == {"a": b"1"}


def test_failed_put_writes_nothing(cache: KVCache) -> None:
    # The second row can't be bound, after the first one was already inserted in the transaction.
    # (A None value wouldn't do: INSERT OR IGNORE silently skips NOT NULL violations too.)
    bad = cast(Any, {"good": b"1", "bad": object()})
    with pytest.raises(sqlite3.Error):
        cache.put_many(bad)
    assert len(cache) == 0
    cache.put_many({"after": b"ok"})  # the connection is still usable (no open transaction)
    assert cache.get_many(["after"]) == {"after": b"ok"}


def test_usable_from_worker_threads(cache: KVCache) -> None:
    # Async callers go through asyncio.to_thread, so calls arrive on arbitrary threads.
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: cache.put_many({f"k{i}": b"v"}), range(40)))
    assert len(cache) == 40


def test_closed_cache_rejects_use(tmp_path: Path) -> None:
    with KVCache(tmp_path / "cache.sqlite") as cache:
        pass
    with pytest.raises(sqlite3.ProgrammingError):
        cache.get_many(["a"])
