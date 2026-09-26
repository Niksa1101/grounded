"""Header-aware chunker (Tech.md §5.5). **Author-owned module** (AGENTS.md §3).

Spec tests: ``tests/unit/test_chunker.py``.
"""

from __future__ import annotations

from grounded.ingest.tokens import TokenCounter
from grounded.ingest.types import Chunk, ChunkingConfig, ParsedDocument


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
    raise NotImplementedError("chunk_document is Author-owned: see the spec tests")
