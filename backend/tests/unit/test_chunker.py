"""Spec for the header-aware chunker (Tech.md §5.5; contract in ``chunk_document``'s docstring).

These tests are written before the implementation (Author-owned). They use a "one word = one
token" counter, so every expected size can be checked by counting words: ``"## Alpha"`` is 2
tokens, ``"a1 a2 a3."`` is 3. Contents are compared as word lists (``content.split()``), so the
whitespace used to join pieces is up to the implementation.

Unless a test says otherwise: max_tokens=20, overlap_tokens=6, min_tokens=5, and every page starts
with ``# Title`` + a 6-word intro paragraph (an 8-token intro chunk, ordinal 0).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.ingest.chunker import _split_sentences, chunk_document
from grounded.ingest.corpus import discover_pages
from grounded.ingest.markdown import parse_page
from grounded.ingest.tokens import TokenCounter
from grounded.ingest.types import (
    Block,
    Chunk,
    ChunkingConfig,
    CodeBlock,
    HeadingBlock,
    ParsedDocument,
    TextBlock,
)

PATH = "docs/en/docs/tutorial/page.md"
URL = "https://fastapi.tiangolo.com/tutorial/page/"
NAV = ("Tutorial - User Guide",)
TOP = (*NAV, "Title")  # breadcrumb of the intro and of anything merged at page level
CFG = ChunkingConfig(max_tokens=20, overlap_tokens=6, min_tokens=5, tokenizer="words")
CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"


def count_words(text: str) -> int:
    return len(text.split())


# --- Builders ---------------------------------------------------------------------------------


def h(level: int, text: str) -> HeadingBlock:
    return HeadingBlock(
        level=level, text=text, anchor=text.lower(), markdown=f"{'#' * level} {text}"
    )


def sents(tag: str, *sizes: int) -> list[str]:
    """Sentences with unique words: sents("a", 2, 3) == ["a1 a2.", "a3 a4 a5."]."""
    out: list[str] = []
    n = 0
    for size in sizes:
        out.append(" ".join(f"{tag}{n + i}" for i in range(1, size + 1)) + ".")
        n += size
    return out


def para(*sentences: str) -> TextBlock:
    return TextBlock(markdown=" ".join(sentences), kind="paragraph")


def code(tag: str, n: int) -> CodeBlock:
    """A fenced block of n + 2 words (both fence lines count as one word each)."""
    body = " ".join(f"{tag}{i}" for i in range(1, n + 1))
    return CodeBlock(markdown=f"```python\n{body}\n```", lang="python")


def bullets(*items: str) -> TextBlock:
    return TextBlock(markdown="\n".join(items), kind="list", items=items)


def table(rows: int) -> TextBlock:
    """5 words per row ("| r1a | r1b |") + 1 for the separator line: 5 + 1 + 5 * rows words."""
    lines = ["| ha | hb |", "|---|---|", *(f"| r{i}a | r{i}b |" for i in range(1, rows + 1))]
    return TextBlock(markdown="\n".join(lines), kind="table")


INTRO = para(*sents("i", 6))


def doc(*blocks: Block, intro: bool = True) -> ParsedDocument:
    head: tuple[Block, ...] = (h(1, "Title"), INTRO) if intro else (h(1, "Title"),)
    return ParsedDocument(
        source_path=PATH,
        url=URL,
        title="Title",
        nav_path=NAV,
        blocks=(*head, *blocks),
        content_hash="0" * 64,
    )


def run(
    document: ParsedDocument, cfg: ChunkingConfig = CFG, counter: TokenCounter = count_words
) -> list[Chunk]:
    return chunk_document(document, cfg, counter)


def body(document: ParsedDocument, cfg: ChunkingConfig = CFG) -> list[Chunk]:
    """Chunks after the standard intro chunk."""
    chunks = run(document, cfg)
    assert chunks[0].anchor_path == ()
    return chunks[1:]


def words(*texts: str) -> list[str]:
    return " ".join(texts).split()


# --- 1. Sections and metadata -----------------------------------------------------------------


def test_intro_and_one_section() -> None:
    a = sents("a", 6)
    chunks = run(doc(h(2, "Alpha"), para(*a)))

    intro, alpha = chunks
    assert intro == Chunk(
        ordinal=0,
        section_id=f"{PATH}#",
        anchor_path=(),
        breadcrumb=TOP,
        heading_level=1,
        url=URL,
        content=intro.content,
        token_count=8,  # "# Title" (2) + 6
    )
    assert intro.content.split() == words("# Title", INTRO.markdown)
    assert alpha == Chunk(
        ordinal=1,
        section_id=f"{PATH}#alpha",
        anchor_path=("alpha",),
        breadcrumb=(*TOP, "Alpha"),
        heading_level=2,
        url=f"{URL}#alpha",
        content=alpha.content,
        token_count=8,
    )
    assert alpha.content.split() == words("## Alpha", *a)


def test_h2_and_h3_are_boundaries() -> None:
    chunks = run(
        doc(
            h(2, "Alpha"),
            para(*sents("a", 6)),
            h(3, "Beta"),
            para(*sents("b", 6)),
            h(3, "Gamma"),
            para(*sents("g", 6)),
            h(2, "Delta"),
            para(*sents("d", 6)),
        )
    )
    # Every section is 8 tokens (>= min 5): no merging.
    assert [c.anchor_path for c in chunks] == [
        (),
        ("alpha",),
        ("alpha", "beta"),
        ("alpha", "gamma"),
        ("delta",),
    ]
    assert [c.breadcrumb for c in chunks] == [
        TOP,
        (*TOP, "Alpha"),
        (*TOP, "Alpha", "Beta"),
        (*TOP, "Alpha", "Gamma"),
        (*TOP, "Delta"),
    ]
    assert [c.heading_level for c in chunks] == [1, 2, 3, 3, 2]
    assert [c.section_id for c in chunks] == [
        f"{PATH}#",
        f"{PATH}#alpha",
        f"{PATH}#beta",
        f"{PATH}#gamma",
        f"{PATH}#delta",
    ]
    assert [c.ordinal for c in chunks] == [0, 1, 2, 3, 4]


def test_h4_and_deeper_are_content_not_boundaries() -> None:
    a, d = sents("a", 6), sents("d", 6)
    [alpha] = body(doc(h(2, "Alpha"), para(*a), h(4, "Deep"), para(*d)))
    assert alpha.anchor_path == ("alpha",)
    assert alpha.content.split() == words("## Alpha", *a, "#### Deep", *d)  # 16 tokens


def test_h3_before_any_h2_hangs_off_the_page() -> None:
    [beta] = body(doc(h(3, "Beta"), para(*sents("b", 6))))
    assert beta.anchor_path == ("beta",)
    assert beta.breadcrumb == (*TOP, "Beta")
    assert beta.heading_level == 3
    assert beta.section_id == f"{PATH}#beta"


def test_explicit_anchor_is_used_for_ids_and_urls() -> None:
    heading = HeadingBlock(
        level=2, text="Use Depends", anchor="use-depends-custom", markdown="## Use `Depends`"
    )
    [chunk] = body(doc(heading, para(*sents("a", 6))))
    assert chunk.section_id == f"{PATH}#use-depends-custom"
    assert chunk.url == f"{URL}#use-depends-custom"
    assert chunk.breadcrumb == (*TOP, "Use Depends")  # plain text, not Markdown
    assert chunk.content.startswith("## Use `Depends`")


# --- 2. Empty and small parents ---------------------------------------------------------------


def test_empty_h2_heading_goes_to_its_first_h3() -> None:
    b, g = sents("b", 6), sents("g", 6)
    beta, gamma = body(doc(h(2, "Alpha"), h(3, "Beta"), para(*b), h(3, "Gamma"), para(*g)))
    assert beta.content.split() == words("## Alpha", "### Beta", *b)  # 10 tokens
    assert beta.token_count == 10
    # The child keeps its own metadata; a label on Alpha still matches through anchor_path.
    assert beta.anchor_path == ("alpha", "beta")
    assert beta.breadcrumb == (*TOP, "Alpha", "Beta")
    assert beta.heading_level == 3
    assert gamma.content.split() == words("### Gamma", *g)


def test_empty_intro_heading_goes_to_the_first_section() -> None:
    a = sents("a", 6)
    [alpha] = run(doc(h(2, "Alpha"), para(*a), intro=False))
    assert alpha.content.split() == words("# Title", "## Alpha", *a)
    assert alpha.anchor_path == ("alpha",)
    assert alpha.ordinal == 0


def test_small_h2_text_goes_to_its_first_h3() -> None:
    b, g = sents("b", 6), sents("g", 6)
    # Alpha = "## Alpha a1 a2." = 4 < 5; Beta = 8, so 4 + 8 = 12 <= 20.
    beta, gamma = body(
        doc(h(2, "Alpha"), para("a1 a2."), h(3, "Beta"), para(*b), h(3, "Gamma"), para(*g))
    )
    assert beta.content.split() == words("## Alpha", "a1 a2.", "### Beta", *b)
    assert beta.token_count == 12
    assert beta.anchor_path == ("alpha", "beta")  # the child keeps its own metadata
    assert beta.breadcrumb == (*TOP, "Alpha", "Beta")
    assert beta.heading_level == 3
    assert beta.section_id == f"{PATH}#beta"
    assert gamma.content.split() == words("### Gamma", *g)


def test_small_parent_stays_alone_when_the_child_would_exceed_max() -> None:
    # Alpha (4) + Beta (2 + 16 = 18) = 22 > 20. Alpha has no adjacent sibling to merge with either
    # (the intro before it and Beta after it are not siblings), so it stays a small chunk.
    alpha, beta = body(doc(h(2, "Alpha"), para("a1 a2."), h(3, "Beta"), para(*sents("b", 16))))
    assert (alpha.anchor_path, alpha.token_count) == (("alpha",), 4)
    assert alpha.breadcrumb == (*TOP, "Alpha")
    assert (beta.anchor_path, beta.token_count) == (("alpha", "beta"), 18)


def test_small_intro_goes_to_the_first_section() -> None:
    a = sents("a", 6)
    # Intro = "# Title i1." = 3 < 5; Alpha = 8, so 3 + 8 = 11 <= 20.
    [alpha] = run(doc(para("i1."), h(2, "Alpha"), para(*a), intro=False))
    assert alpha.content.split() == words("# Title", "i1.", "## Alpha", *a)
    assert alpha.token_count == 11
    assert alpha.anchor_path == ("alpha",)
    assert alpha.section_id == f"{PATH}#alpha"
    assert alpha.breadcrumb == (*TOP, "Alpha")
    assert alpha.ordinal == 0


def test_small_intro_stays_alone_without_room() -> None:
    # Intro (3) + Alpha (2 + 18 = 20) = 23 > 20. The intro has no sibling, so it can't merge either.
    intro, alpha = run(doc(para("i1."), h(2, "Alpha"), para(*sents("a", 18)), intro=False))
    assert (intro.anchor_path, intro.token_count) == ((), 3)
    assert intro.breadcrumb == TOP
    assert (alpha.anchor_path, alpha.token_count) == (("alpha",), 20)


def test_prepending_chains_through_small_parents() -> None:
    b = sents("b", 6)
    # Own content: intro "# Title i1." = 3 and Alpha "## Alpha a1." = 3, both < 5 (Alpha is judged
    # without the intro prepended to it). Beta = 8, so 3 + 3 + 8 = 14 <= 20.
    [beta] = run(doc(para("i1."), h(2, "Alpha"), para("a1."), h(3, "Beta"), para(*b), intro=False))
    assert beta.content.split() == words("# Title", "i1.", "## Alpha", "a1.", "### Beta", *b)
    assert beta.token_count == 14
    assert beta.anchor_path == ("alpha", "beta")
    assert beta.ordinal == 0


def test_page_with_only_a_title_has_no_chunks() -> None:
    assert run(doc(intro=False)) == []


# --- 3. Small sections ------------------------------------------------------------------------


def test_small_section_merges_with_the_next_sibling() -> None:
    b = sents("b", 6)
    # Alpha = "## Alpha a1." = 3 < 5; Alpha + Beta = 3 + 8 = 11 <= 20.
    [merged] = body(doc(h(2, "Alpha"), para("a1."), h(2, "Beta"), para(*b)))
    assert merged.content.split() == words("## Alpha", "a1.", "## Beta", *b)
    assert merged.token_count == 11
    # Metadata from the first section, breadcrumb from the common parent (the page).
    assert merged.anchor_path == ("alpha",)
    assert merged.section_id == f"{PATH}#alpha"
    assert merged.url == f"{URL}#alpha"
    assert merged.heading_level == 2
    assert merged.breadcrumb == TOP


def test_merging_continues_while_still_small() -> None:
    cfg = ChunkingConfig(max_tokens=20, overlap_tokens=6, min_tokens=8, tokenizer="words")
    chunks = body(
        doc(
            h(2, "Alpha"),
            para("a1."),  # 3
            h(2, "Beta"),
            para("b1."),  # 3 -> Alpha+Beta = 6, still < 8
            h(2, "Gamma"),
            para("g1."),  # 3 -> 9 >= 8: stop
            h(2, "Delta"),
            para(*sents("d", 6)),  # 8: not small
        ),
        cfg,
    )
    assert [(c.anchor_path, c.token_count) for c in chunks] == [(("alpha",), 9), (("delta",), 8)]
    assert chunks[0].content.split() == words(
        "## Alpha", "a1.", "## Beta", "b1.", "## Gamma", "g1."
    )


def test_small_section_stays_alone_when_the_merge_would_exceed_max() -> None:
    # Alpha (3) + Beta (2 + 17 = 19) = 22 > 20; the intro before Alpha is not a sibling.
    alpha, beta = body(doc(h(2, "Alpha"), para("a1."), h(2, "Beta"), para(*sents("b", 17))))
    assert (alpha.anchor_path, alpha.token_count) == (("alpha",), 3)
    assert alpha.breadcrumb == (*TOP, "Alpha")  # not merged: its own breadcrumb
    assert (beta.anchor_path, beta.token_count) == (("beta",), 19)


def test_small_last_section_merges_backward() -> None:
    a = sents("a", 6)
    # Beta = 3 < 5 and has no next sibling; Alpha (8) + Beta (3) = 11 <= 20.
    [merged] = body(doc(h(2, "Alpha"), para(*a), h(2, "Beta"), para("b1.")))
    assert merged.content.split() == words("## Alpha", *a, "## Beta", "b1.")
    assert merged.anchor_path == ("alpha",)  # the first (previous) section's anchor
    assert merged.breadcrumb == TOP


def test_small_last_section_stays_alone_without_room() -> None:
    # Alpha (2 + 16 = 18) + Beta (3) = 21 > 20.
    alpha, beta = body(doc(h(2, "Alpha"), para(*sents("a", 16)), h(2, "Beta"), para("b1.")))
    assert (alpha.anchor_path, alpha.token_count) == (("alpha",), 18)
    assert (beta.anchor_path, beta.token_count) == (("beta",), 3)
    assert beta.breadcrumb == (*TOP, "Beta")


def test_merging_never_crosses_a_parent() -> None:
    o = sents("o", 6)
    chunks = body(
        doc(
            h(2, "Alpha"),
            para(*sents("a", 6)),  # 8
            h(3, "One"),
            para(*o),  # 8
            h(3, "Two"),
            para("t1."),  # 3: small; next section Beta has another parent
            h(2, "Beta"),
            para(*sents("b", 6)),  # 8
        )
    )
    assert [c.anchor_path for c in chunks] == [("alpha",), ("alpha", "one"), ("beta",)]
    merged = chunks[1]
    assert merged.content.split() == words("### One", *o, "### Two", "t1.")
    assert merged.breadcrumb == (*TOP, "Alpha")  # common parent: Alpha
    assert merged.heading_level == 3
    assert merged.section_id == f"{PATH}#one"


# --- 4. Long sections -------------------------------------------------------------------------


def test_long_section_splits_between_blocks_with_overlap() -> None:
    a, b, c = sents("a", 3, 3, 3), sents("b", 3, 3, 3), sents("c", 3, 3, 3)
    # Section = 2 + 27 = 29 > 20. Part 1: heading + A + B = 20. C doesn't fit.
    # Overlap: B's trailing sentences up to 6 tokens = b4-b6 + b7-b9 (3 + 3). Part 2: 6 + 9 = 15.
    first, second = body(doc(h(2, "Alpha"), para(*a), para(*b), para(*c)))
    assert first.content.split() == words("## Alpha", *a, *b)
    assert second.content.split() == words(b[1], b[2], *c)
    assert (first.token_count, second.token_count) == (20, 15)
    # Parts share the section's metadata; only the ordinal differs.
    for part in (first, second):
        assert part.anchor_path == ("alpha",)
        assert part.section_id == f"{PATH}#alpha"
        assert part.url == f"{URL}#alpha"
        assert part.breadcrumb == (*TOP, "Alpha")
        assert part.heading_level == 2
    assert (first.ordinal, second.ordinal) == (1, 2)


def test_a_block_that_fits_the_next_part_moves_whole() -> None:
    a, b = sents("a", 5, 5, 5), sents("b", 3, 3)
    # heading + A = 17; B (6) would make 23. B fits a fresh part after the overlap (a11-a15 = 5),
    # so it moves whole instead of putting b1-b3 into part 1.
    first, second = body(doc(h(2, "Alpha"), para(*a), para(*b)))
    assert first.content.split() == words("## Alpha", *a)
    assert second.content.split() == words(a[2], *b)
    assert (first.token_count, second.token_count) == (17, 11)


def test_long_paragraph_splits_into_sentences() -> None:
    s = sents("s", 5, 5, 5, 5, 5, 5)
    # Section = 2 + 30. The paragraph can't fit any part whole, so it's split into sentences and
    # packed from the heading on: 2 + 5 + 5 + 5 = 17 (+5 would be 22).
    # Overlap: s[2] (5 <= 6; s[1] + s[2] = 10 > 6). Part 2: 5 + 5 + 5 + 5 = 20.
    first, second = body(doc(h(2, "Alpha"), para(*s)))
    assert first.content.split() == words("## Alpha", s[0], s[1], s[2])
    assert second.content.split() == words(s[2], s[3], s[4], s[5])
    assert (first.token_count, second.token_count) == (17, 20)


def test_no_overlap_when_the_last_sentence_exceeds_the_budget() -> None:
    s = sents("s", 7, 7, 7, 7)
    # Part 1: 2 + 7 + 7 = 16. The last sentence (7) is bigger than overlap_tokens (6).
    first, second = body(doc(h(2, "Alpha"), para(*s)))
    assert first.content.split() == words("## Alpha", s[0], s[1])
    assert second.content.split() == words(s[2], s[3])


def test_no_overlap_after_a_code_block() -> None:
    block, p = code("k", 12), sents("p", 3, 3, 3)
    # heading + code = 2 + 14 = 16; + 9 = 25 > 20. Code is never copied as overlap.
    first, second = body(doc(h(2, "Alpha"), block, para(*p)))
    assert first.content.split() == words("## Alpha", block.markdown)
    assert second.content.split() == words(*p)


def test_code_block_is_never_split_and_may_exceed_max() -> None:
    block = code("k", 30)  # 32 tokens
    # heading + x = 5. The code doesn't fit, not even in a fresh part, and is atomic: it gets a
    # part of its own, without overlap. After code there is no overlap either.
    first, second, third = body(doc(h(2, "Alpha"), para("x1 x2 x3."), block, para("y1 y2 y3.")))
    assert first.content.split() == words("## Alpha", "x1 x2 x3.")
    assert second.content == block.markdown
    assert second.token_count == 32
    assert third.content.split() == words("y1 y2 y3.")


def test_oversized_first_block_keeps_the_heading() -> None:
    block = code("k", 30)
    first, second = body(doc(h(2, "Alpha"), block, para("y1 y2 y3.")))
    assert first.content.startswith("## Alpha")
    assert block.markdown in first.content
    assert first.token_count == 34
    assert second.content.split() == words("y1 y2 y3.")


def test_heading_is_never_left_alone_in_a_part() -> None:
    a = sents("a", 5, 5, 5, 4)
    # The paragraph (19) would fit a fresh part, but not next to the heading (21). Moving it would
    # leave a heading-only part, so it's split instead: 2 + 5 + 5 + 5 = 17; then a[2] (5) + 4 = 9.
    first, second = body(doc(h(2, "Alpha"), para(*a)))
    assert first.content.split() == words("## Alpha", a[0], a[1], a[2])
    assert second.content.split() == words(a[2], a[3])


def test_heading_stays_with_an_atomic_first_block_even_over_max() -> None:
    block = code("k", 17)  # 19 tokens: fits a part alone, but 2 + 19 = 21 with the heading
    first, second = body(doc(h(2, "Alpha"), block, para("y1 y2 y3.")))
    assert first.content.split() == words("## Alpha", block.markdown)
    assert first.token_count == 21
    assert second.content.split() == words("y1 y2 y3.")


def test_table_is_atomic() -> None:
    grid = table(5)  # 5 + 1 + 25 = 31 tokens
    first, second, third = body(doc(h(2, "Alpha"), para("x1 x2 x3."), grid, para("y1 y2 y3.")))
    assert first.content.split() == words("## Alpha", "x1 x2 x3.")
    assert second.content == grid.markdown
    assert third.content.split() == words("y1 y2 y3.")


def test_long_list_splits_between_top_level_items() -> None:
    items = tuple(f"* l{i}1 l{i}2 l{i}3 l{i}4 l{i}5" for i in range(1, 6))  # 6 tokens each
    # Section = 2 + 30. The list can't fit any part whole, so it's split into items:
    # 2 + 6 + 6 + 6 = 20. No overlap after a list item. Part 2: 6 + 6 = 12.
    first, second = body(doc(h(2, "Alpha"), bullets(*items)))
    assert first.content.split() == words("## Alpha", *items[:3])
    assert second.content.split() == words(*items[3:])
    for item in items:  # items stay whole
        assert sum(item in c.content for c in (first, second)) == 1


# --- 5. Output invariants ---------------------------------------------------------------------


MIXED = doc(
    h(2, "Alpha"), para(*sents("a", 4, 4)),
    h(3, "One"), para(*sents("o", 5, 5, 5, 5, 5)), code("k", 10), para(*sents("p", 3)),
    h(3, "Two"), para("t1."),
    h(2, "Beta"), bullets("* b1 b2 b3", "* b4 b5 b6"), table(4), code("z", 25),
    h(4, "Deep"), para(*sents("d", 6, 6, 6)),
    h(2, "Gamma"),
    h(3, "Three"), para(*sents("r", 8)),
)  # fmt: skip


def test_output_invariants() -> None:
    chunks = run(MIXED)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    atomic = [
        b.markdown
        for b in MIXED.blocks
        if isinstance(b, CodeBlock) or (isinstance(b, TextBlock) and b.kind == "table")
    ]
    for chunk in chunks:
        assert chunk.content
        assert chunk.content == chunk.content.strip()
        assert chunk.token_count == count_words(chunk.content)
        assert chunk.section_id.startswith(f"{PATH}#")
        if chunk.token_count > CFG.max_tokens:  # only an oversized atomic block may exceed
            assert any(count_words(m) > CFG.max_tokens and m in chunk.content for m in atomic)
    for markdown in atomic:  # never split, never copied as overlap
        assert sum(markdown in c.content for c in chunks) == 1


def test_deterministic() -> None:
    assert run(MIXED) == run(MIXED)


def test_sizes_come_from_the_injected_counter() -> None:
    def double(text: str) -> int:
        return 2 * len(text.split())

    a = sents("a", 3, 3, 3)
    # 11 words = 22 tokens > 20. Heading (4) + a[0] (6) + a[1] (6) = 16; + a[2] would be 22.
    # Overlap: a[1] (6 <= 6). Part 2: 6 + 6 = 12.
    first, second = run(doc(h(2, "Alpha"), para(*a)), CFG, double)[1:]
    assert first.content.split() == words("## Alpha", a[0], a[1])
    assert second.content.split() == words(a[1], a[2])
    assert (first.token_count, second.token_count) == (16, 12)


def test_corpus_mini_pages_chunk_cleanly() -> None:
    cfg = ChunkingConfig(max_tokens=450, overlap_tokens=50, min_tokens=40, tokenizer="words")
    for page in discover_pages(CORPUS_MINI):
        parsed = parse_page(page, CORPUS_MINI)
        chunks = chunk_document(parsed, cfg, count_words)
        has_content = any(not isinstance(b, HeadingBlock) for b in parsed.blocks)
        assert bool(chunks) == has_content, page.source_path
        for block in parsed.blocks:
            if isinstance(block, CodeBlock):
                assert any(block.markdown in c.content for c in chunks), page.source_path


# --- Implementation choices the contract leaves open ------------------------------------------


def test_a_heading_with_nothing_under_it_yields_no_chunk() -> None:
    # Alpha has no blocks and its next section (Beta) is a sibling, not a child to take it.
    b = sents("b", 6)
    [beta] = body(doc(h(2, "Alpha"), h(2, "Beta"), para(*b)))
    assert beta.anchor_path == ("beta",)
    assert beta.content.split() == words("## Beta", *b)


def test_a_page_of_headings_only_has_no_chunks() -> None:
    assert run(doc(h(2, "Alpha"), h(3, "Beta"), h(2, "Gamma"), intro=False)) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One two. Three four! Five? Six", ["One two.", "Three four!", "Five?", "Six"]),
        ("Use a tool, e.g. pip. Then run it.", ["Use a tool, e.g. pip.", "Then run it."]),
        ("It needs Python 3.10 or newer. Done.", ["It needs Python 3.10 or newer.", "Done."]),
        ("Call `app.get(). ` first. Then go.", ["Call `app.get(). ` first.", "Then go."]),
        ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
        ("A **bold claim.** Next one.", ["A **bold claim.**", "Next one."]),
        ("Line one\ncontinues here. Next.", ["Line one\ncontinues here.", "Next."]),
    ],
)
def test_sentence_boundaries(text: str, expected: list[str]) -> None:
    assert list(_split_sentences(text)) == expected
