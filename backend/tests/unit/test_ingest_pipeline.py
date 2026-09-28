"""The database-free parts of the ingest pipeline: corpus preparation, index identity, token stats,
input-length and cache checks. Storing and activation are in tests/integration/test_ingest.py."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

import grounded.ingest.pipeline as pipeline
from grounded.infra.kvcache import KVCache
from grounded.ingest.corpus import EXCLUDED_PAGES
from grounded.ingest.embed import embedding_cache_key
from grounded.ingest.markdown import PARSER_VERSION
from grounded.ingest.pipeline import (
    IngestError,
    check_input_lengths,
    count_uncached,
    index_chunking_config,
    index_spec,
    prepare_corpus,
    token_stats,
)
from grounded.ingest.types import ChunkingConfig, CorpusCheckout

CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"
CHECKOUT = CorpusCheckout(path=CORPUS_MINI, ref="0.0.1", sha="a" * 40)
CFG = ChunkingConfig(max_tokens=60, overlap_tokens=10, min_tokens=5, tokenizer="words")


def words(text: str) -> int:
    return len(text.split())


# --- prepare_corpus --------------------------------------------------------------------------


def test_prepare_corpus_parses_and_chunks_every_discovered_page() -> None:
    corpus = prepare_corpus(CHECKOUT, CFG, words)
    paths = [prepared.doc.source_path for prepared in corpus.documents]
    assert paths == sorted(paths)
    assert "docs/en/docs/tutorial/background-tasks.md" in paths
    assert not any("release-notes" in path for path in paths)  # excluded pages stay out
    assert corpus.chunks == [c for p in corpus.documents for c in p.chunks]
    assert all(
        chunk.section_id.startswith(p.doc.source_path)
        for p in corpus.documents
        for chunk in p.chunks
    )


def test_prepare_corpus_is_deterministic() -> None:
    assert prepare_corpus(CHECKOUT, CFG, words) == prepare_corpus(CHECKOUT, CFG, words)


def test_prepare_corpus_without_chunks_is_an_error(tmp_path: Path) -> None:
    docs = tmp_path / "docs" / "en" / "docs"
    docs.mkdir(parents=True)
    (docs / "index.md").write_text("# Only a title\n", encoding="utf-8")
    with pytest.raises(IngestError, match="no chunks"):
        prepare_corpus(dataclasses.replace(CHECKOUT, path=tmp_path), CFG, words)


# --- token_stats -----------------------------------------------------------------------------


def test_token_stats_nearest_rank() -> None:
    # 20 sizes 1..20: p50 = 10th value, p95 = ceil(19) = 19th value.
    stats = token_stats(list(range(20, 0, -1)), max_tokens=18)
    assert stats == {"min": 1, "p50": 10, "p95": 19, "max": 20, "total": 210, "over_max": 2}


def test_token_stats_single_chunk() -> None:
    assert token_stats([7], max_tokens=5) == {
        "min": 7, "p50": 7, "p95": 7, "max": 7, "total": 7, "over_max": 1
    }  # fmt: skip


def test_token_stats_needs_counts() -> None:
    with pytest.raises(ValueError, match="no token counts"):
        token_stats([], max_tokens=5)


# --- index identity --------------------------------------------------------------------------


def test_chunking_config_records_exclusions_and_parser_version() -> None:
    stored = index_chunking_config(CFG)
    assert stored == {
        **dataclasses.asdict(CFG),
        "excluded_pages": sorted(EXCLUDED_PAGES),
        "parser_version": PARSER_VERSION,
    }
    json.dumps(stored)  # goes into a jsonb column


def test_config_hash_is_sha256_of_sha_model_dim_and_canonical_config() -> None:
    spec = index_spec(CHECKOUT, CFG, embedding_model="m", embedding_dim=768)
    canonical = json.dumps(index_chunking_config(CFG), sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(f"{'a' * 40}|m|768|{canonical}".encode()).hexdigest()
    assert spec.config_hash == expected
    assert spec.label == f"0.0.1@{expected[:8]}"
    assert (spec.git_ref, spec.git_sha, spec.embedding_model, spec.embedding_dim) == (
        "0.0.1", "a" * 40, "m", 768
    )  # fmt: skip


def test_config_hash_ignores_the_ref_name_but_not_the_sha() -> None:
    base = index_spec(CHECKOUT, CFG, embedding_model="m", embedding_dim=768)
    renamed = dataclasses.replace(CHECKOUT, ref="other-tag")
    moved = dataclasses.replace(CHECKOUT, sha="b" * 40)
    assert index_spec(renamed, CFG, embedding_model="m", embedding_dim=768).config_hash == (
        base.config_hash
    )
    assert index_spec(moved, CFG, embedding_model="m", embedding_dim=768).config_hash != (
        base.config_hash
    )


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "dim",
        "chunking",
        "parser_version",
        "excluded_pages",
    ],
)
def test_config_hash_changes_with_anything_that_changes_the_index(
    change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = index_spec(CHECKOUT, CFG, embedding_model="m", embedding_dim=768).config_hash
    model, dim, cfg = "m", 768, CFG
    if change == "model":
        model = "m2"
    elif change == "dim":
        dim = 1536
    elif change == "chunking":
        cfg = dataclasses.replace(CFG, max_tokens=61)
    elif change == "parser_version":
        monkeypatch.setattr(pipeline, "PARSER_VERSION", PARSER_VERSION + 1)
    else:
        monkeypatch.setattr(pipeline, "EXCLUDED_PAGES", {**EXCLUDED_PAGES, "new.md": "why"})
    assert index_spec(CHECKOUT, cfg, embedding_model=model, embedding_dim=dim).config_hash != base


# --- pre-embedding checks --------------------------------------------------------------------


def test_input_length_check_names_the_sections() -> None:
    chunks = prepare_corpus(CHECKOUT, CFG, words).chunks
    longest = max(chunks, key=lambda c: words(c.embedding_text))
    limit = words(longest.embedding_text) - 1
    with pytest.raises(IngestError, match="embedding input limit") as excinfo:
        check_input_lengths(chunks, words, limit)
    assert longest.section_id in str(excinfo.value)
    check_input_lengths(chunks, words, limit + 1)  # at the limit is fine


def test_input_length_check_lists_at_most_ten_sections() -> None:
    chunks = prepare_corpus(CHECKOUT, CFG, words).chunks * 3
    assert len(chunks) > 10
    with pytest.raises(IngestError, match=f"{len(chunks)} chunk\\(s\\)") as excinfo:
        check_input_lengths(chunks, words, 0)
    assert f"and {len(chunks) - 10} more" in str(excinfo.value)


def test_count_uncached_counts_distinct_missing_texts(tmp_path: Path) -> None:
    chunks = prepare_corpus(CHECKOUT, CFG, words).chunks
    spec = index_spec(CHECKOUT, CFG, embedding_model="m", embedding_dim=8)
    with KVCache(tmp_path / "e.sqlite") as cache:
        assert count_uncached(cache, spec, chunks + chunks) == len(chunks)
        first = chunks[0].embedding_text
        cache.put_many({embedding_cache_key("m", 8, "RETRIEVAL_DOCUMENT", first): b"x"})
        assert count_uncached(cache, spec, chunks) == len(chunks) - 1
        # Another model's or task's vectors don't count.
        other = index_spec(CHECKOUT, CFG, embedding_model="m2", embedding_dim=8)
        assert count_uncached(cache, other, chunks) == len(chunks)
