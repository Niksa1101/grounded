"""Header-aware chunker (Tech.md §5.5). **Author-owned module** (AGENTS.md §3), written by the
Agent at the Author's explicit request (2026-09-26).

Spec tests: ``tests/unit/test_chunker.py``.

The work is a pipeline over an intermediate ``_Section`` (metadata + the blocks it holds), one
step per contract rule, so each rule stays readable on its own:

    blocks → _split_sections (1) → _absorb_thin_parents (2) → _merge_small_siblings (3)
           → _section_parts (4: long sections only) → Chunk (5)

Sections keep *blocks*, not strings, until the very end: rules 2-4 move, join and split along
block boundaries, which a joined string would have lost.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from grounded.ingest.tokens import TokenCounter
from grounded.ingest.types import (
    Block,
    Chunk,
    ChunkingConfig,
    HeadingBlock,
    ParsedDocument,
    TextBlock,
)

_BLOCK_SEP = "\n\n"


def chunk_document(
    doc: ParsedDocument, cfg: ChunkingConfig, count_tokens: TokenCounter
) -> list[Chunk]:
    """Split one parsed page into chunks.

    Pure and deterministic: same ``doc`` + ``cfg`` + counter → identical output. Sizes are always
    ``count_tokens(content)``; the breadcrumb is not counted.

    **1. Sections.** H2 and H3 headings split ``doc.blocks`` into sections. The blocks between the
    H1 and the first H2/H3 are the page intro (anchor ``""``, heading = the H1). H4-H6 headings are
    ordinary content of their section. A section's content starts with its own heading line.

    - ``anchor_path``: ``(h2,)`` or ``(h2, h3)``; ``()`` for the intro; ``(h3,)`` for an H3 that
      comes before any H2 (it happens on 7 pages at 0.141.1).
    - ``breadcrumb``: ``doc.nav_path + (doc.title,)`` + the texts of the headings in anchor_path.
    - ``section_id``: ``f"{doc.source_path}#{deepest anchor}"`` (``"...md#"`` for the intro).
    - ``url``: ``f"{doc.url}#{deepest anchor}"``; ``doc.url`` for the intro.
    - ``heading_level``: level of the deepest heading (1 for the intro).
    - A page with nothing but its H1 yields no chunks.

    **2. Empty and small parents.** A section directly followed by one of its children (an intro
    followed by an H2/H3, an H2 followed by an H3) is not a chunk of its own when its content is
    too thin to stand alone. Its content (heading line + blocks) is prepended to the child's:

    - always, if it has no blocks of its own (just the heading line);
    - if its own content is < ``min_tokens``, only when the child's content stays ≤ ``max_tokens``;
      otherwise it stays a (small) chunk of its own.
    - "Own content" is the section's heading line + its own blocks, before anything was prepended
      to it. Prepending chains: a small intro, then a small H2, then an H3 all end up in the H3.
    - The child keeps its own metadata; a label on the parent still matches through the child's
      ``anchor_path``. This step runs before small-section merging (3).
    - A section left with nothing but heading lines (no child took it) yields no chunk: a heading
      alone carries nothing to retrieve. The H1-only page above is the common case.

    **3. Small sections** (content < ``min_tokens``) merge with an adjacent *sibling* section
    (same parent: same ``anchor_path[:-1]``, and adjacent in document order), only if the merged
    content is ≤ ``max_tokens``:

    - prefer the next section; merging repeats while the merged chunk is still < ``min_tokens``;
    - otherwise the previous section (e.g. a small last section merges backward);
    - otherwise the section stays alone. The intro has no siblings, so it never merges (it can
      only be prepended to its first child, see 2).
    - A merged chunk takes ``anchor_path``, ``section_id``, ``url`` and ``heading_level`` from its
      *first* section, and the *common parent's* breadcrumb (without the first section's heading).
    - Merged content keeps every section's heading line, in document order.

    **4. Long sections** (content > ``max_tokens``) are split into parts that share the section's
    metadata (only ``ordinal`` differs). Blocks are packed greedily in document order:

    - The heading line is only in the first part and is never a part on its own: it stays with
      the first block after it. If that block doesn't fit next to the heading it is broken up (see
      below), so its first pieces go with the heading; an atomic first block stays with the
      heading even if the two together exceed ``max_tokens``.
    - Any later block that doesn't fit into the current part starts the next part, if it fits
      there (after the overlap). Otherwise it is broken up and its pieces are packed, starting in
      the current part: a paragraph (``TextBlock.kind == "paragraph"``) into sentences, a list into
      its top-level ``items``.
    - Everything else is atomic and never broken: code blocks, tables, list items, sentences,
      containers, HTML, blockquotes, H4+ heading lines. An atomic block that doesn't fit after the
      overlap starts a part without overlap; one larger than ``max_tokens`` becomes a part of its
      own. Only these parts, and a heading glued to an atomic first block, may exceed the limit.
    - **Overlap:** a part after the first starts with the longest run of whole trailing sentences
      of the previous part that totals ≤ ``overlap_tokens``, taken only from a paragraph (or a
      paragraph piece) that ends the previous part. No overlap after code, tables, lists or any
      other block, and none if the last sentence alone is larger than ``overlap_tokens``. Code is
      never copied.
    - A sentence ends at ``.``, ``!`` or ``?`` followed by whitespace or the end of the block. The
      tests only use such plain sentences; abbreviations ("e.g."), versions ("3.10") and inline
      code are yours to handle.

    **5. Output.** Chunks in document order, ``ordinal`` = 0, 1, 2, …; ``content`` has no leading
    or trailing whitespace; ``token_count == count_tokens(content)``. Whitespace between joined
    pieces is the implementer's choice (tests compare words).
    """
    sections = _split_sections(doc)
    sections = _absorb_thin_parents(sections, cfg, count_tokens)
    sections = _merge_small_siblings(sections, cfg, count_tokens)
    chunks: list[Chunk] = []
    for section in sections:
        for content in _section_parts(section, cfg, count_tokens):
            chunks.append(_make_chunk(doc, section, content, len(chunks), count_tokens))
    return chunks


# --- Sections (rules 1-3) -----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Section:
    """A future chunk (or run of chunks, if it's long): its metadata and the blocks it holds.

    ``blocks`` starts with the section's own heading line (the H1 for the intro), preceded by the
    content of any thin parents prepended to it (rule 2), followed by sibling sections merged into
    it (rule 3).
    """

    anchor_path: tuple[str, ...]
    breadcrumb: tuple[str, ...]  # of the first section; a merged chunk drops its last heading
    heading_level: int
    blocks: tuple[Block, ...]
    merged: bool = False  # siblings were merged in (rule 3): the breadcrumb stops at the parent

    @property
    def is_intro(self) -> bool:
        return not self.anchor_path


def _split_sections(doc: ParsedDocument) -> list[_Section]:
    """Rule 1: cut the page at every H2 and H3. Everything before the first cut is the intro."""
    top = (*doc.nav_path, doc.title)
    sections: list[_Section] = []
    # The section being filled; it starts as the intro, whose heading line is the H1 block itself.
    path: tuple[HeadingBlock, ...] = ()
    level = 1
    blocks: list[Block] = []
    last_h2: HeadingBlock | None = None
    for block in doc.blocks:
        if isinstance(block, HeadingBlock) and block.level in (2, 3):
            sections.append(_new_section(top, path, level, blocks))
            if block.level == 2:
                last_h2 = block
            # An H3 hangs under the last H2, or directly under the page if no H2 came yet.
            path = (last_h2, block) if block.level == 3 and last_h2 is not None else (block,)
            level = block.level
            blocks = [block]
        else:
            blocks.append(block)
    sections.append(_new_section(top, path, level, blocks))
    return sections


def _new_section(
    top: tuple[str, ...], path: tuple[HeadingBlock, ...], level: int, blocks: list[Block]
) -> _Section:
    return _Section(
        anchor_path=tuple(h.anchor for h in path),
        breadcrumb=(*top, *(h.text for h in path)),
        heading_level=level,
        blocks=tuple(blocks),
    )


def _absorb_thin_parents(
    sections: list[_Section], cfg: ChunkingConfig, count_tokens: TokenCounter
) -> list[_Section]:
    """Rule 2: hand an empty or small parent's content down to its first child."""
    out: list[_Section] = []
    carry: tuple[Block, ...] = ()  # content of thin parents waiting for their child
    for i, section in enumerate(sections):
        blocks = (*carry, *section.blocks)
        child = sections[i + 1] if i + 1 < len(sections) else None
        if child is not None and _is_child(child, section):
            heading_only = len(section.blocks) == 1
            # "Small" is judged on the section's own content, not on what was carried into it.
            small = _size(section.blocks, count_tokens) < cfg.min_tokens
            if heading_only or (
                small and _size((*blocks, *child.blocks), count_tokens) <= cfg.max_tokens
            ):
                carry = blocks
                continue
        carry = ()
        if any(not isinstance(b, HeadingBlock) for b in blocks):
            out.append(
                _Section(section.anchor_path, section.breadcrumb, section.heading_level, blocks)
            )
    return out


def _merge_small_siblings(
    sections: list[_Section], cfg: ChunkingConfig, count_tokens: TokenCounter
) -> list[_Section]:
    """Rule 3: merge a small section into its next sibling (repeatedly), else its previous one."""
    out: list[_Section] = []
    i = 0
    while i < len(sections):
        current = sections[i]
        i += 1
        if current.is_intro:
            out.append(current)
            continue
        merged_forward = False
        while (
            _size(current.blocks, count_tokens) < cfg.min_tokens
            and i < len(sections)
            and _is_sibling(current, sections[i])
            and _size((*current.blocks, *sections[i].blocks), count_tokens) <= cfg.max_tokens
        ):
            current = _merge(current, sections[i])
            i += 1
            merged_forward = True
        if (
            not merged_forward
            and _size(current.blocks, count_tokens) < cfg.min_tokens
            and out
            and _is_sibling(out[-1], current)
            and _size((*out[-1].blocks, *current.blocks), count_tokens) <= cfg.max_tokens
        ):
            out[-1] = _merge(out[-1], current)
        else:
            out.append(current)
    return out


def _is_child(child: _Section, parent: _Section) -> bool:
    depth = len(parent.anchor_path)
    return len(child.anchor_path) > depth and child.anchor_path[:depth] == parent.anchor_path


def _is_sibling(a: _Section, b: _Section) -> bool:
    return not a.is_intro and not b.is_intro and a.anchor_path[:-1] == b.anchor_path[:-1]


def _merge(first: _Section, second: _Section) -> _Section:
    return _Section(
        anchor_path=first.anchor_path,
        breadcrumb=first.breadcrumb,
        heading_level=first.heading_level,
        blocks=(*first.blocks, *second.blocks),
        merged=True,
    )


def _join(blocks: tuple[Block, ...]) -> str:
    return _BLOCK_SEP.join(b.markdown for b in blocks).strip()


def _size(blocks: tuple[Block, ...], count_tokens: TokenCounter) -> int:
    return count_tokens(_join(blocks))


# --- Long sections (rule 4) ---------------------------------------------------------------------

type _PieceKind = Literal["heading", "block", "sentence", "item"]

# Pieces cut from one block are rejoined the way they were written; different blocks get a blank
# line between them.
_SAME_BLOCK_SEP: dict[_PieceKind, str] = {"sentence": " ", "item": "\n"}


@dataclass(frozen=True, slots=True)
class _Piece:
    """A unit the packer places: a heading line, a whole block, a sentence or a list item."""

    text: str
    source: int  # index of the block in the section it comes from
    kind: _PieceKind
    sentences: tuple[str, ...] = ()  # paragraph text only: the one thing overlap may copy


def _section_parts(section: _Section, cfg: ChunkingConfig, count_tokens: TokenCounter) -> list[str]:
    """The section's content, split into parts only if it is over ``max_tokens``."""
    content = _join(section.blocks)
    if count_tokens(content) <= cfg.max_tokens:
        return [content]
    return _Packer(cfg, count_tokens).pack(section.blocks)


class _Packer:
    """Greedy packing of one long section into parts; ``_place`` is the decision table of rule 4."""

    def __init__(self, cfg: ChunkingConfig, count_tokens: TokenCounter) -> None:
        self._cfg = cfg
        self._count = count_tokens
        self._parts: list[list[_Piece]] = []
        self._current: list[_Piece] = []

    def pack(self, blocks: tuple[Block, ...]) -> list[str]:
        # The heading lines at the top: the section's own, plus those of empty parents (rule 2).
        head = 0
        while head < len(blocks) and _is_section_heading(blocks[head]):
            head += 1
        self._current = [_Piece(b.markdown, i, "heading") for i, b in enumerate(blocks[:head])]
        for index in range(head, len(blocks)):
            whole, pieces = _pieces(blocks[index], index)
            self._place(whole, pieces)
        self._parts.append(self._current)
        return [_render(part) for part in self._parts]

    def _place(self, piece: _Piece, pieces: tuple[_Piece, ...] = ()) -> None:
        """Put ``piece`` somewhere; ``pieces`` is what it breaks into (empty if it is atomic)."""
        if self._fits([*self._current, piece]):
            self._current.append(piece)
        elif all(p.kind == "heading" for p in self._current):
            # The heading never stands alone: break the first block up, or keep it whole over max.
            if pieces:
                self._place_all(pieces)
            else:
                self._current.append(piece)
        elif self._fits([*(overlap := self._overlap()), piece]):
            self._start([*overlap, piece])
        elif pieces:
            self._place_all(pieces)  # starting in the current part
        else:
            self._start([piece])  # atomic: a fresh part without overlap, alone if oversized

    def _place_all(self, pieces: tuple[_Piece, ...]) -> None:
        for piece in pieces:
            self._place(piece)

    def _start(self, pieces: list[_Piece]) -> None:
        self._parts.append(self._current)
        self._current = pieces

    def _fits(self, pieces: list[_Piece]) -> bool:
        return self._count(_render(pieces)) <= self._cfg.max_tokens

    def _overlap(self) -> list[_Piece]:
        """The longest run of whole trailing sentences ≤ ``overlap_tokens``, taken from the
        paragraph that ends the current part; nothing if the part ends with anything else."""
        last = self._current[-1]
        sentences: list[str] = []
        for piece in reversed(self._current):
            if piece.source != last.source or not piece.sentences:
                break
            sentences[:0] = piece.sentences
        taken: list[str] = []
        for sentence in reversed(sentences):
            candidate = [sentence, *taken]
            if self._count(" ".join(candidate)) > self._cfg.overlap_tokens:
                break
            taken = candidate
        return [_Piece(s, last.source, "sentence", (s,)) for s in taken]


def _is_section_heading(block: Block) -> bool:
    return isinstance(block, HeadingBlock) and block.level <= 3


def _pieces(block: Block, index: int) -> tuple[_Piece, tuple[_Piece, ...]]:
    """The block as one piece, and the smaller pieces it breaks into (none if atomic)."""
    if isinstance(block, TextBlock) and block.kind == "paragraph":
        sentences = _split_sentences(block.markdown)
        whole = _Piece(block.markdown, index, "block", sentences)
        if len(sentences) < 2:
            return whole, ()
        return whole, tuple(_Piece(s, index, "sentence", (s,)) for s in sentences)
    if isinstance(block, TextBlock) and block.kind == "list" and len(block.items) > 1:
        return _Piece(block.markdown, index, "block"), tuple(
            _Piece(item, index, "item") for item in block.items
        )
    return _Piece(block.markdown, index, "block"), ()


def _render(pieces: list[_Piece]) -> str:
    out: list[str] = []
    for i, piece in enumerate(pieces):
        if i:
            same_block = piece.source == pieces[i - 1].source
            out.append(_SAME_BLOCK_SEP.get(piece.kind, _BLOCK_SEP) if same_block else _BLOCK_SEP)
        out.append(piece.text)
    return "".join(out).strip()


# --- Sentences ----------------------------------------------------------------------------------

# ".", "!" or "?" (plus closing quotes, brackets or emphasis markers) before whitespace or the end.
_SENTENCE_END = re.compile(r"[.!?][\"')\]*_]*(?=\s|$)")
# Inline code, including double-backtick spans that contain single backticks.
_CODE_SPAN = re.compile(r"(`+).+?(?<!`)\1(?!`)", re.DOTALL)
# A period after these isn't a sentence end. "etc." is left out on purpose: it usually ends one.
_ABBREVIATIONS = ("e.g.", "i.e.", "vs.", "cf.")


def _split_sentences(text: str) -> tuple[str, ...]:
    """Split a paragraph into sentences; the pieces cover all of its text, in order.

    Not a sentence end: punctuation inside inline code (``app.get()``), inside a word or number
    (``3.10``, ``fastapi.tiangolo.com``: no whitespace after it), or after an abbreviation.
    """
    code = [m.span() for m in _CODE_SPAN.finditer(text)]
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if any(lo < match.start() < hi for lo, hi in code):
            continue
        if text[: match.start() + 1].lower().endswith(_ABBREVIATIONS):
            continue
        if sentence := text[start : match.end()].strip():
            sentences.append(sentence)
        start = match.end()
    if rest := text[start:].strip():
        sentences.append(rest)
    return tuple(sentences)


# --- Output (rule 5) ----------------------------------------------------------------------------


def _make_chunk(
    doc: ParsedDocument,
    section: _Section,
    content: str,
    ordinal: int,
    count_tokens: TokenCounter,
) -> Chunk:
    anchor = section.anchor_path[-1] if section.anchor_path else ""
    return Chunk(
        ordinal=ordinal,
        section_id=f"{doc.source_path}#{anchor}",
        anchor_path=section.anchor_path,
        breadcrumb=section.breadcrumb[:-1] if section.merged else section.breadcrumb,
        heading_level=section.heading_level,
        url=f"{doc.url}#{anchor}" if anchor else doc.url,
        content=content,
        token_count=count_tokens(content),
    )
