"""ChunkingConfig, Chunk and the tiktoken counter: the pieces around the Author-owned chunker."""

from __future__ import annotations

import hashlib
import json

import pytest

from grounded.ingest.tokens import make_token_counter
from grounded.ingest.types import Chunk, ChunkingConfig
from tests.support import make_settings


def test_config_from_settings() -> None:
    cfg = ChunkingConfig.from_settings(make_settings())
    assert cfg == ChunkingConfig(
        max_tokens=450, overlap_tokens=50, min_tokens=40, tokenizer="o200k_base"
    )
    assert cfg.strategy == "headers"


def test_canonical_json_is_key_sorted_and_compact() -> None:
    cfg = ChunkingConfig(max_tokens=450, overlap_tokens=50, min_tokens=40, tokenizer="o200k_base")
    assert cfg.canonical_json() == (
        '{"max_tokens":450,"min_tokens":40,"overlap_tokens":50,'
        '"strategy":"headers","tokenizer":"o200k_base"}'
    )
    assert json.loads(cfg.canonical_json())["max_tokens"] == 450


@pytest.mark.parametrize(
    ("max_tokens", "overlap_tokens", "min_tokens"),
    [(20, 6, 20), (20, 6, 0), (20, 20, 5), (20, -1, 5)],
)
def test_config_rejects_impossible_sizes(
    max_tokens: int, overlap_tokens: int, min_tokens: int
) -> None:
    with pytest.raises(ValueError, match="tokens"):
        ChunkingConfig(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            min_tokens=min_tokens,
            tokenizer="words",
        )


def test_chunk_embedding_text_and_hash() -> None:
    chunk = Chunk(
        ordinal=0,
        section_id="docs/en/docs/tutorial/background-tasks.md#create-a-task-function",
        anchor_path=("create-a-task-function",),
        breadcrumb=("Tutorial - User Guide", "Background Tasks", "Create a task function"),
        heading_level=2,
        url="https://fastapi.tiangolo.com/tutorial/background-tasks/#create-a-task-function",
        content="## Create a task function\n\nFirst, create a function.",
        token_count=10,
    )
    expected_text = (
        "Tutorial - User Guide > Background Tasks > Create a task function\n\n"
        "## Create a task function\n\nFirst, create a function."
    )
    assert chunk.embedding_text == expected_text
    # DB.md §4: sha256(breadcrumb_text || '\n\n' || content), the embedding cache key.
    assert chunk.content_hash == hashlib.sha256(expected_text.encode()).hexdigest()


def test_tiktoken_counter() -> None:
    # Needs the o200k_base file in tiktoken's cache: CI warms it before pytest (network is blocked
    # here). Locally, run once:
    #   uv run python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
    count = make_token_counter("o200k_base")
    assert count("") == 0
    assert count("hello world") == 2
    # Docs text is data: a special-token string is counted as text instead of raising.
    assert count("<|endoftext|>") > 1
