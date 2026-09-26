"""Approximate token counting for chunk sizing (Tech.md §2, §5.5).

The count is only a size budget for the chunker: Gemini's tokenizer differs from tiktoken's, so
``max_tokens`` is a target, not a provider limit. The chunker takes the counter as a parameter,
which keeps it a pure function and lets tests use a trivial "one word = one token" counter.

tiktoken downloads the encoding file on first use and caches it on disk (``TIKTOKEN_CACHE_DIR``,
read by tiktoken itself). pytest blocks the network, so CI warms that cache in a step before the
tests (``ci.yml``). Locally, any use outside pytest (e.g. the first ingest) fills it once.
"""

from __future__ import annotations

from collections.abc import Callable

import tiktoken

type TokenCounter = Callable[[str], int]


def make_token_counter(encoding: str) -> TokenCounter:
    """Return a counter for tiktoken ``encoding`` (e.g. ``"o200k_base"``).

    Special-token strings such as ``<|endoftext|>`` are counted as ordinary text: the docs are
    data, and tiktoken's default is to raise on them.
    """
    enc = tiktoken.get_encoding(encoding)

    def count_tokens(text: str) -> int:
        return len(enc.encode(text, disallowed_special=()))

    return count_tokens
