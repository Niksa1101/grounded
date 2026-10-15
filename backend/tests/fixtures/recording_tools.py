"""The parts the two ``record_generate_fixture.py`` scripts share (``gemini/``, ``groq/``).

A dev tool for recording REAL provider 429 responses by hand. Never run by pytest and never imported
by the app; ``tests/unit/test_record_fixtures.py`` imports it to pin the pure parts (the scrubbing,
the ``--burst`` argument, the counting). No network, no key, no ``grounded`` import here.

What it holds:

- ``split_burst``: the ``--burst N`` option (default 1, hard cap ``BURST_CAP``). The cap is not an
  option: a per-minute 429 needs no more, and a daily 429 is **never provoked** (the scripts record
  one only right after it happened for real).
- ``run_burst``: fires N requests at once, counts the status codes and keeps the **first** 429 in
  the order the replies arrived. The summary line has counts only, never a key or a body.
- ``scrub`` and ``keep_headers``: what may be written to the repo. A real daily 429 names the Groq
  organization (``org_...``); none of that, no request id and no key shape may reach a fixture.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

BURST_CAP = 40

BURST_ONLY_FOR_PER_MINUTE = (
    "--burst only goes with the per-minute 429 (gemini: `error 429_per_minute`, groq: "
    "`429_per_minute`). A daily 429 is never provoked, because that would burn a daily quota on "
    "purpose: run the daily mode (`error 429_daily_quota` / `429_daily_quota`) right after a real "
    "daily 429 happened."
)


def parse_burst(value: str) -> int:
    """``value`` as a request count in ``1..BURST_CAP``, else the script ends with a message."""
    try:
        count = int(value)
    except ValueError:
        raise SystemExit(f"--burst needs a whole number of requests, got {value!r}") from None
    if count < 1:
        raise SystemExit(f"--burst must be at least 1, got {count}")
    if count > BURST_CAP:
        raise SystemExit(
            f"--burst {count} is above the hard cap of {BURST_CAP} requests. The cap is not an "
            "option: a per-minute 429 does not need more, and a daily quota is never burned on purpose."
        )
    return count


def split_burst(args: list[str]) -> tuple[list[str], int]:
    """``args`` without ``--burst N`` / ``--burst=N``, and N (1 when absent). A bad N, a missing
    value or a repeated option ends the script with a message, before anything is sent."""
    rest: list[str] = []
    values: list[str] = []
    items = iter(args)
    for arg in items:
        if arg == "--burst":
            values.append(next(items, ""))
        elif arg.startswith("--burst="):
            values.append(arg.partition("=")[2])
        else:
            rest.append(arg)
    if len(values) > 1:
        raise SystemExit("--burst was given more than once")
    return rest, parse_burst(values[0]) if values else 1


def burst_notice(count: int, target: str) -> str:
    """Printed before anything is sent: how many real requests, and that they use quota."""
    noun = "request" if count == 1 else "requests"
    return (
        f"About to send {count} real {noun} to {target}. This uses quota: {count} {noun} of this "
        "key's per-minute and daily limits (shared with CI and the demo)."
    )


@dataclass(frozen=True)
class Reply:
    """One HTTP reply, reduced to what a fixture keeps. ``read`` is what the adapter made of it."""

    status_code: int
    headers: dict[str, str]
    body: dict[str, Any]
    read: str = ""


@dataclass(frozen=True)
class BurstResult:
    """``counts`` maps a label (a status code, or the class name of a failure that had no reply) to
    how many requests ended that way. ``first_429`` is the first 429 that arrived, if any."""

    counts: dict[str, int]
    first_429: Reply | None

    def summary(self) -> str:
        """One line: counts only, never a key or a body."""
        total = sum(self.counts.values())
        parts = ", ".join(f"{label} x{self.counts[label]}" for label in sorted(self.counts))
        return f"sent {total}: {parts}"


async def run_burst(count: int, send: Callable[[], Awaitable[Reply]]) -> BurstResult:
    """Run ``count`` calls of ``send`` at once and wait for all of them. The first 429 in the
    order the replies **arrived** is kept and the others are only counted. A call that raises
    (a connection failure) is counted under the class name of its error, never its message."""
    tasks = [asyncio.ensure_future(send()) for _ in range(count)]
    counts: Counter[str] = Counter()
    first_429: Reply | None = None
    for finished in asyncio.as_completed(tasks):
        try:
            reply = await finished
        except Exception as exc:  # counted and reported by class name, not swallowed
            counts[type(exc).__name__] += 1
            continue
        counts[str(reply.status_code)] += 1
        if reply.status_code == 429 and first_429 is None:
            first_429 = reply
    return BurstResult(dict(counts), first_429)


# --- What may be written to the repo --------------------------------------------------------------

# Each pattern with its fixed placeholder. The organization id of Groq's daily 429, request ids, the
# key shapes of the two providers, a bearer token and any UUID.
_SCRUBBED = (
    (re.compile(r"\borg_[0-9A-Za-z]+"), "org_REDACTED"),
    (re.compile(r"\breq_[0-9A-Za-z]+"), "req_REDACTED"),
    (re.compile(r"\bgsk_[0-9A-Za-z_-]{8,}"), "gsk_REDACTED"),
    (re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"), "AIza_REDACTED"),
    (re.compile(r"(?i)\bBearer\s+[0-9A-Za-z._~+/=-]+"), "Bearer REDACTED"),
    (
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
        "00000000-0000-0000-0000-000000000000",
    ),
)

_KEPT_HEADER_PREFIX = "x-ratelimit"
_KEPT_HEADERS = frozenset({"retry-after"})


def scrub(value: Any) -> Any:
    """``value`` (a body, nested dicts and lists of JSON) with every id and key shape replaced by a
    fixed placeholder. Pure: a new structure, the same text everywhere else. Idempotent."""
    if isinstance(value, str):
        for pattern, placeholder in _SCRUBBED:
            value = pattern.sub(placeholder, value)
        return value
    if isinstance(value, dict):
        return {scrub(key): scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    return value


def keep_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """The headers a 429 fixture keeps: the rate-limit ones and ``retry-after``, lower-cased,
    sorted, scrubbed. Everything else (cookies, ray and request ids, dates) is dropped."""
    kept = {
        key.lower(): scrub(value)
        for key, value in headers.items()
        if key.lower().startswith(_KEPT_HEADER_PREFIX) or key.lower() in _KEPT_HEADERS
    }
    return dict(sorted(kept.items()))
