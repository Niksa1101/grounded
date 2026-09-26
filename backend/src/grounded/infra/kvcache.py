"""Persistent key-value cache in a local SQLite file (Tech.md §11).

One file per cache under ``CACHE_DIR`` (``embeddings.sqlite`` now; rerank and eval LLM responses
later). Offline tooling only: ingest, evals, dev. The request path in prod uses in-memory caches.

The API is synchronous (stdlib ``sqlite3``); async callers wrap calls in ``asyncio.to_thread`` so
disk I/O never blocks the event loop (AGENTS.md §6.12). A lock serializes access because those
calls may land on different worker threads.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Self

# SQLite caps bound parameters per statement (32766 since 3.32); stay well below it.
_KEYS_PER_QUERY = 500


class KVCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        # WAL lets a second process read while ingest writes (e.g. an eval run on the same cache).
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS kv ("
            " key TEXT PRIMARY KEY,"
            " value BLOB NOT NULL,"
            " created_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ")"
        )

    def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        """Values for the keys that are present; missing keys are simply absent."""
        found: dict[str, bytes] = {}
        with self._lock:
            for start in range(0, len(keys), _KEYS_PER_QUERY):
                batch = keys[start : start + _KEYS_PER_QUERY]
                # Only "?" placeholders are formatted in; the keys themselves are bound.
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT key, value FROM kv WHERE key IN ({placeholders})", tuple(batch)
                )
                found.update((key, bytes(value)) for key, value in rows)
        return found

    def put_many(self, items: Mapping[str, bytes]) -> None:
        """Store all items in one transaction. An existing key keeps its first value: entries
        are content-addressed, so a second write for the same key carries the same value."""
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(
                    "INSERT OR IGNORE INTO kv (key, value) VALUES (?, ?)", items.items()
                )
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def __len__(self) -> int:
        with self._lock:
            (count,) = self._conn.execute("SELECT count(*) FROM kv").fetchone()
        return int(count)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
