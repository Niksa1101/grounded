"""Value objects passed between ingestion stages.

Frozen and slotted: every stage is a pure-ish function of the previous stage's output, and ingest
determinism (same tag + config → same index) depends on nobody mutating them along the way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from grounded.settings import Settings


@dataclass(frozen=True, slots=True)
class CorpusCheckout:
    """A local checkout of the docs repository at one pinned tag."""

    path: Path
    ref: str  # the tag, e.g. "0.141.1"
    sha: str  # resolved commit SHA (40 hex), recorded per index version


@dataclass(frozen=True, slots=True)
class PageRef:
    """One docs page selected for indexing, before it is read or parsed."""

    source_path: str  # repo-relative POSIX path: "docs/en/docs/tutorial/background-tasks.md"
    url: str  # canonical page URL: "https://fastapi.tiangolo.com/tutorial/background-tasks/"
    # Section names of the enclosing directories, outermost first: ("Tutorial - User Guide",).
    # Empty for top-level pages. The first breadcrumb levels of every chunk on the page.
    nav_path: tuple[str, ...]


# --- Parsed page ------------------------------------------------------------------------------
# A page is a flat sequence of blocks in document order. Sections (heading + what follows until the
# next heading of the same or higher level) are *not* built here: grouping blocks into sections and
# chunks is the chunker's job.


@dataclass(frozen=True, slots=True)
class HeadingBlock:
    level: int  # 1..6
    text: str  # plain text for breadcrumbs: inline markup and the "{ #anchor }" suffix removed
    anchor: str  # explicit "{ #anchor }" if the source has one, else a toc-style slug
    markdown: str  # the heading as Markdown without the anchor suffix: "## Use `Depends`"


# What a text block is, so the chunker can decide how (or whether) to split it without re-parsing
# Markdown: paragraphs split into sentences, lists into top-level items, everything else is kept
# whole. "container" is a prose-only admonition/tab merged into one block (see markdown.py).
type TextKind = Literal["paragraph", "list", "table", "blockquote", "html", "container"]


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One top-level Markdown element other than a heading or code: a paragraph, list, table,
    blockquote, raw HTML, or an admonition/tab marker line (``/// note``). Verbatim source lines."""

    markdown: str
    kind: TextKind
    # Lists only: each top-level item verbatim with its marker (nested lists stay inside their
    # item), in order. Empty for every other kind.
    items: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """A fenced or indented code block, fences included. The chunker must never split one."""

    markdown: str
    lang: str  # first word of the fence info, lowercased ("python", "console"); "" if none


type Block = HeadingBlock | TextBlock | CodeBlock


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    source_path: str  # "docs/en/docs/tutorial/background-tasks.md"
    url: str  # page URL without an anchor
    title: str  # plain text of the page's H1
    nav_path: tuple[str, ...]  # see PageRef.nav_path
    blocks: tuple[Block, ...]  # document order, H1 included
    content_hash: str  # sha256 hex of the resolved Markdown the blocks were parsed from


# --- Chunking ---------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Everything that changes chunk boundaries. Stored as ``index_versions.chunking_config`` and
    part of ``config_hash`` (DB.md §4), so a change here always means a new index version."""

    max_tokens: int
    overlap_tokens: int
    min_tokens: int
    tokenizer: str  # tiktoken encoding the counts come from: counts differ between encodings
    strategy: str = "headers"

    def __post_init__(self) -> None:
        if not 0 < self.min_tokens < self.max_tokens:
            raise ValueError("need 0 < min_tokens < max_tokens")
        if not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("need 0 <= overlap_tokens < max_tokens")

    @classmethod
    def from_settings(cls, settings: Settings) -> ChunkingConfig:
        return cls(
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            min_tokens=settings.chunk_min_tokens,
            tokenizer=settings.tokenizer_encoding,
        )

    def canonical_json(self) -> str:
        """Key-sorted, whitespace-free JSON: the same config always hashes the same."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of a page (Tech.md §5.5), before it is embedded and stored.

    Several chunks may share a ``section_id``: the parts of a section that was too long.
    """

    ordinal: int  # 0-based position within the document
    section_id: str  # "<source_path>#<deepest anchor>"; the page intro has an empty anchor
    anchor_path: tuple[str, ...]  # ancestor anchors, outermost first; () for the page intro
    breadcrumb: tuple[str, ...]  # nav names + H1 + H2 (+ H3) texts
    heading_level: int  # level of the deepest heading the chunk belongs to (1 for the intro)
    url: str  # page URL + "#" + deepest anchor; the bare page URL for the intro
    content: str  # Markdown as stored, embedded and shown to the LLM
    token_count: int  # count_tokens(content); breadcrumb not included

    @property
    def breadcrumb_text(self) -> str:
        return " > ".join(self.breadcrumb)

    @property
    def embedding_text(self) -> str:
        """What gets embedded: heading context measurably helps retrieval (Tech.md §5.6)."""
        return f"{self.breadcrumb_text}\n\n{self.content}"

    @property
    def content_hash(self) -> str:
        """sha256 of ``embedding_text``: the embedding cache key and ``chunks.content_hash``."""
        return hashlib.sha256(self.embedding_text.encode("utf-8")).hexdigest()
