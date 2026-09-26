"""Markdown → blocks (Tech.md §5.4): headings, anchors, code atomicity, containers, hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from grounded.ingest.corpus import discover_pages
from grounded.ingest.markdown import (
    ParseError,
    parse_markdown,
    parse_page,
    slugify,
    split_heading_anchor,
)
from grounded.ingest.types import CodeBlock, HeadingBlock, PageRef, ParsedDocument, TextBlock

CORPUS_MINI = Path(__file__).resolve().parents[1] / "fixtures" / "corpus_mini"
PAGE = PageRef(
    source_path="docs/en/docs/tutorial/page.md",
    url="https://fastapi.tiangolo.com/tutorial/page/",
    nav_path=("Tutorial - User Guide",),
)


def _parse(text: str) -> ParsedDocument:
    return parse_markdown(text, PAGE, CORPUS_MINI)


def _headings(text: str) -> list[HeadingBlock]:
    return [b for b in _parse(text).blocks if isinstance(b, HeadingBlock)]


# --- Slugs and anchors ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "slug"),
    [
        ("Create a task function", "create-a-task-function"),
        ("Using BackgroundTasks", "using-backgroundtasks"),
        ("Path Parameters - Numeric Validations", "path-parameters-numeric-validations"),
        ("Order the parameters as you need, tricks", "order-the-parameters-as-you-need-tricks"),
        ("Recap: what's next?", "recap-whats-next"),
        ("Café & Résumé", "cafe-resume"),
        ("snake_case stays", "snake_case-stays"),
        ("  lots   of   space  ", "lots-of-space"),
    ],
)
def test_slugify_matches_python_markdown_toc(text: str, slug: str) -> None:
    assert slugify(text) == slug


def test_split_heading_anchor() -> None:
    assert split_heading_anchor("Create a task { #create-a-task }") == (
        "Create a task",
        "create-a-task",
    )
    assert split_heading_anchor("Plain title") == ("Plain title", None)
    assert split_heading_anchor("With class {#x .cls }") == ("With class", "x")


def test_explicit_anchor_is_used_and_removed_from_the_text() -> None:
    [h1, h2] = _headings("# Title { #title }\n\n## Use `Depends` here { #use-depends }\n")
    assert h1 == HeadingBlock(level=1, text="Title", anchor="title", markdown="# Title")
    assert h2.anchor == "use-depends"
    assert h2.text == "Use Depends here"  # plain text for breadcrumbs
    assert h2.markdown == "## Use `Depends` here"  # Markdown keeps the code span


def test_missing_anchor_is_slugified_from_plain_text() -> None:
    [_, h2] = _headings("# T\n\n## Read **the** [docs](https://x.y) of `Query`\n")
    assert h2.text == "Read the docs of Query"
    assert h2.anchor == "read-the-docs-of-query"


def test_generated_anchors_are_unique_on_a_page() -> None:
    headings = _headings("# T\n\n## Recap\n\n## Recap\n\n## Recap { #recap_1 }\n\n### Recap\n")
    # Explicit ids are reserved first; generated ones take the next free _N suffix.
    assert [h.anchor for h in headings] == ["t", "recap", "recap_2", "recap_1", "recap_3"]


def test_setext_heading_is_normalized_to_atx() -> None:
    [h1, h2] = _headings("Title\n=====\n\nSection\n-------\n")
    assert (h1.level, h1.markdown) == (1, "# Title")
    assert (h2.level, h2.markdown) == (2, "## Section")


# --- Blocks -----------------------------------------------------------------------------------


def test_blocks_are_verbatim_in_document_order() -> None:
    doc = _parse(
        "# Title\n\nIntro *text*.\n\n* one\n* two\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        '```Python\nx = 1\n```\n\n    indented()\n\n---\n\n<div class="termy">\n\n$ run\n\n</div>\n'
    )
    assert doc.blocks == (
        HeadingBlock(level=1, text="Title", anchor="title", markdown="# Title"),
        TextBlock(markdown="Intro *text*.", kind="paragraph"),
        TextBlock(markdown="* one\n* two", kind="list", items=("* one", "* two")),
        TextBlock(markdown="| a | b |\n|---|---|\n| 1 | 2 |", kind="table"),
        CodeBlock(markdown="```Python\nx = 1\n```", lang="python"),
        CodeBlock(markdown="    indented()", lang=""),
        # the thematic break is dropped; raw HTML is kept as text
        TextBlock(markdown='<div class="termy">', kind="html"),
        TextBlock(markdown="$ run", kind="paragraph"),
        TextBlock(markdown="</div>", kind="html"),
    )


def test_list_items_are_top_level_and_verbatim() -> None:
    doc = _parse(
        "# T\n\n1. First item\n   * nested a\n   * nested b\n2. Second\n\n   continued.\n\n"
        "- loose one\n\n- loose two\n"
    )
    ordered, bullet = doc.blocks[1:]
    assert isinstance(ordered, TextBlock)
    assert ordered.kind == "list"
    # A nested list stays inside its item; a continuation paragraph belongs to its item.
    assert ordered.items == (
        "1. First item\n   * nested a\n   * nested b",
        "2. Second\n\n   continued.",
    )
    assert bullet == TextBlock(
        markdown="- loose one\n\n- loose two", kind="list", items=("- loose one", "- loose two")
    )


def test_non_list_text_blocks_have_no_items() -> None:
    doc = _parse("# T\n\nPara.\n\n> quote\n\n| a |\n|---|\n| 1 |\n")
    assert [(b.kind, b.items) for b in doc.blocks[1:] if isinstance(b, TextBlock)] == [
        ("paragraph", ()),
        ("blockquote", ()),
        ("table", ()),
    ]


def test_heading_inside_code_fence_is_code() -> None:
    doc = _parse("# Title\n\n```python\n# not a heading\n## nor this\n```\n")
    assert [type(b) for b in doc.blocks] == [HeadingBlock, CodeBlock]


def test_h4_and_deeper_are_heading_blocks_too() -> None:
    # Whether H4+ start a chunk is the chunker's decision; the parser reports every level.
    assert [h.level for h in _headings("# A\n\n#### B\n\n###### C\n")] == [1, 4, 6]


def test_prose_admonition_becomes_one_text_block() -> None:
    doc = _parse("# T\n\n/// tip | Pro tip\n\nFirst.\n\n* a list\n\n///\n\nAfter.\n")
    assert doc.blocks[1:] == (
        TextBlock(markdown="/// tip | Pro tip\n\nFirst.\n\n* a list\n\n///", kind="container"),
        TextBlock(markdown="After.", kind="paragraph"),
    )


def test_nested_prose_containers_merge_into_the_outermost() -> None:
    doc = _parse("# T\n\n//// tab | A\n\n/// note\n\nInner.\n\n///\n\n////\n")
    assert doc.blocks[1:] == (
        TextBlock(markdown="//// tab | A\n\n/// note\n\nInner.\n\n///\n\n////", kind="container"),
    )


def test_container_with_code_stays_flat_so_code_stays_atomic() -> None:
    doc = _parse("# T\n\n/// note\n\nSee:\n\n```python\nx = 1\n```\n\n///\n")
    assert doc.blocks[1:] == (
        TextBlock(markdown="/// note", kind="paragraph"),
        TextBlock(markdown="See:", kind="paragraph"),
        CodeBlock(markdown="```python\nx = 1\n```", lang="python"),
        TextBlock(markdown="///", kind="paragraph"),
    )


def test_front_matter_jinja_html_and_comments_are_dropped() -> None:
    doc = _parse(
        "---\ninclude_yaml:\n  sponsors: data/sponsors.yml\n---\n\n# Home\n\n<!-- sponsors -->\n\n"
        '<div>\n{% for s in sponsors %}<a href="{{ s.url }}"></a>{% endfor %}\n</div>\n\nText.\n'
    )
    assert doc.blocks == (
        HeadingBlock(level=1, text="Home", anchor="home", markdown="# Home"),
        TextBlock(markdown="Text.", kind="paragraph"),
    )


def test_crlf_input_parses_like_lf() -> None:
    text = "# Title\n\nSome text.\n\n```python\nx = 1\n```\n"
    assert _parse(text.replace("\n", "\r\n")) == _parse(text)


def test_page_without_h1_is_an_error() -> None:
    with pytest.raises(ParseError, match="no H1"):
        _parse("## Only a section\n\nText.\n")


# --- Whole pages from corpus_mini -------------------------------------------------------------


def _page(suffix: str) -> PageRef:
    return next(p for p in discover_pages(CORPUS_MINI) if p.source_path.endswith(suffix))


def test_parse_page_resolves_includes_into_code_blocks() -> None:
    doc = parse_page(_page("tutorial/background-tasks.md"), CORPUS_MINI)

    assert doc.title == "Background Tasks"
    assert doc.url == "https://fastapi.tiangolo.com/tutorial/background-tasks/"
    assert doc.nav_path == ("Tutorial - User Guide",)
    assert [(h.level, h.anchor) for h in doc.blocks if isinstance(h, HeadingBlock)] == [
        (1, "background-tasks"),
        (2, "using-backgroundtasks"),
        (2, "create-a-task-function"),
        (3, "technical-details"),
    ]
    code = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert len(code) == 2
    assert all(c.lang == "python" for c in code)
    assert "background_tasks.add_task" in code[0].markdown
    assert "# Code above omitted 👆" in code[1].markdown
    assert not any("{*" in b.markdown for b in doc.blocks)


def test_parse_page_legacy_include_and_jinja_raw_markers() -> None:
    doc = parse_page(_page("advanced/templates.md"), CORPUS_MINI)
    [code] = [b for b in doc.blocks if isinstance(b, CodeBlock)]
    assert code.lang == "jinja"
    assert code.markdown.split("\n")[0] == "```jinja"
    assert "<h1>Item ID: {{ id }}</h1>" in code.markdown
    assert "raw %}" not in code.markdown


def test_content_hash_is_sha256_of_the_resolved_markdown() -> None:
    page = _page("tutorial/background-tasks.md")
    doc = parse_page(page, CORPUS_MINI)
    assert len(doc.content_hash) == 64
    assert doc.content_hash == parse_page(page, CORPUS_MINI).content_hash

    source = (CORPUS_MINI / page.source_path).read_text(encoding="utf-8")
    unresolved = hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest()
    assert doc.content_hash != unresolved  # includes are part of the hashed content
