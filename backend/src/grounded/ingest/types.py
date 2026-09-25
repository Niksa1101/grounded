"""Value objects passed between ingestion stages.

Frozen and slotted: every stage is a pure-ish function of the previous stage's output, and ingest
determinism (same tag + config → same index) depends on nobody mutating them along the way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True, slots=True)
class TextBlock:
    """One top-level Markdown element other than a heading or code: a paragraph, list, table,
    blockquote, raw HTML, or an admonition/tab marker line (``/// note``). Verbatim source lines."""

    markdown: str


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
