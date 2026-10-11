"""Question normalization and its hash (Tech.md §11, DB.md §4), and the keyed hash of a client IP.

One definition, two users: ``request_logs.question_hash`` (3.12) and the answer cache key (3.13), so
"the same question" means the same thing in the logs and in the cache.

Normalization: Unicode NFKC, lowercase, collapse runs of whitespace to one space, strip trailing
sentence punctuation. NFKC also folds full-width punctuation to ASCII before the strip.

**Only sentence punctuation is stripped** (``. , ; : ! ?`` and ``…`` ``。``), not every punctuation
or symbol character: "What is C#?" and "What is C?" are different questions, and so are "foo()" and
"foo(". A cache key that merged them would serve the wrong answer. Leading punctuation and anything
inside the text are kept.
"""

from __future__ import annotations

import hashlib
import hmac
import unicodedata

_TRAILING = frozenset(".,;:!?…。")


def normalize_question(text: str) -> str:
    folded = " ".join(unicodedata.normalize("NFKC", text).lower().split())
    # Punctuation can hide behind spaces ("why ?"), so strip both until neither is left.
    end = len(folded)
    while end and (folded[end - 1] in _TRAILING or folded[end - 1] == " "):
        end -= 1
    return folded[:end]


def question_hash(text: str) -> str:
    """sha256 (hex) of the normalized question."""
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def hash_ip(ip: str, key: bytes) -> str:
    """HMAC-SHA256 (hex) of a client IP, keyed with ``IP_HASH_SECRET`` (Tech.md §12).

    Keyed on purpose: the IPv4 space is small enough to reverse a plain sha256 by brute force, so an
    unkeyed hash would not be pseudonymous. Callers pass the canonical spelling of the address
    (``api/security.py``) so one client always maps to one hash.
    """
    return hmac.new(key, ip.encode("utf-8"), hashlib.sha256).hexdigest()
