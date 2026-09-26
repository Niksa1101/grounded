"""Parse one docs page into a flat list of heading / text / code blocks (Tech.md §5.3, §5.4).

Pipeline for a page: normalize line endings → drop YAML front matter → resolve code includes →
drop Jinja ``{% raw %}`` markers → hash → markdown-it tokens → blocks.

Why markdown-it and not regexes: it knows CommonMark's block structure, so a ``# comment`` inside a
code fence is code, not a heading, and a fence inside a list stays inside that list. Block content
is sliced from the source lines (``token.map``) instead of re-rendered, so what gets chunked,
embedded and shown to the LLM is exactly the Markdown the docs contain.

Two kinds of code block are dropped after parsing, because they only add noise to retrieval
(Author decision, 2026-09-26): a repeat of an identical code block earlier on the same page (the
docs re-include one file per section with other lines highlighted, and highlights are stripped),
and a block carrying embedded binary data (a long base64 run, e.g. an image in a string literal:
nothing to retrieve, and a risk for the embedding model's input limit).

Things the site renders but this parser keeps as plain text: admonitions (``/// note`` … ``///``)
and tabs (``//// tab | …``) are not CommonMark containers, so their marker lines become text blocks
and their contents are parsed normally.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Final

from markdown_it import MarkdownIt
from markdown_it.token import Token

from grounded.ingest.includes import resolve_includes
from grounded.ingest.types import (
    Block,
    CodeBlock,
    HeadingBlock,
    PageRef,
    ParsedDocument,
    TextBlock,
    TextKind,
)

logger = logging.getLogger(__name__)

_MD: Final = MarkdownIt("commonmark").enable("table")

# Bump whenever the same page parses to different blocks: include resolution, what gets dropped,
# how blocks are cut. It is part of every index version's config hash (ingest/pipeline.py), so an
# index built by an older parser is never mistaken for the current one. Pure refactors keep it.
PARSER_VERSION: Final = 1

# attr_list id on a heading: "Create a task function { #create-a-task-function }".
_ANCHOR_RE = re.compile(r"\s*\{\s*#(?P<id>[^\s}]+)[^}]*\}\s*$")
# Jinja raw markers the docs' macros plugin strips before rendering.
_JINJA_RAW_RE = re.compile(r"^\s*\{%-?\s*(?:end)?raw\s*-?%\}\s*$")
# Opening line of a pymdownx block (admonition, details, tab): "/// tip", "//// tab | Python 3.10+".
_CONTAINER_OPEN_RE = re.compile(r"^(?P<fence>/{3,})\s*[A-Za-z]")
# Embedded binary data: base64 has no spaces, so real code never has a run this long.
_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{400,}={0,2}")
# Python-Markdown toc's de-duplication suffix: "title", "title_1", "title_2", ...
_ID_COUNT_RE = re.compile(r"^(?P<base>.*)_(?P<n>[0-9]+)$")
# Top-level token → TextBlock.kind (lists are handled separately: they also carry their items).
# Anything else that reaches a text block (a paragraph, a lone container marker) is "paragraph".
_TEXT_KINDS: Final[dict[str, TextKind]] = {
    "table_open": "table",
    "blockquote_open": "blockquote",
    "html_block": "html",
}


class ParseError(Exception):
    """A page can't be turned into a document (e.g. it has no H1 title)."""


def parse_page(page: PageRef, root: Path) -> ParsedDocument:
    """Read and parse ``page`` from the checkout at ``root``."""
    text = (root / page.source_path).read_text(encoding="utf-8")
    return parse_markdown(text, page, root)


def parse_markdown(text: str, page: PageRef, root: Path) -> ParsedDocument:
    """Parse page source ``text``; ``root`` is the checkout that include paths resolve against."""
    text = _strip_front_matter(text.replace("\r\n", "\n"))
    text = resolve_includes(text, root, source_path=page.source_path)
    text = "\n".join(line for line in text.split("\n") if not _JINJA_RAW_RE.match(line))
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    blocks = _drop_noise_code(_blocks(text, page.source_path), page.source_path)
    title = next((b.text for b in blocks if isinstance(b, HeadingBlock) and b.level == 1), None)
    if not title:
        raise ParseError(f"{page.source_path}: page has no H1 title")
    return ParsedDocument(
        source_path=page.source_path,
        url=page.url,
        title=title,
        nav_path=page.nav_path,
        blocks=tuple(blocks),
        content_hash=content_hash,
    )


def slugify(text: str, separator: str = "-") -> str:
    """Python-Markdown ``toc`` default slug: ASCII-fold, drop punctuation, lowercase, join words.

    Used only for headings without an explicit ``{ #anchor }``, so generated anchors match the
    ones the live site links to.
    """
    value = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value).strip().lower()
    return re.sub(rf"[{separator}\s]+", separator, value)


def split_heading_anchor(inline_markdown: str) -> tuple[str, str | None]:
    """``"Title { #title }"`` → ``("Title", "title")``; ``"Title"`` → ``("Title", None)``."""
    match = _ANCHOR_RE.search(inline_markdown)
    if match is None:
        return inline_markdown.strip(), None
    return inline_markdown[: match.start()].strip(), match["id"]


def _strip_front_matter(text: str) -> str:
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return text
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1 :])
    return text  # an unclosed "---" is a thematic break, not front matter


def _blocks(text: str, source_path: str) -> list[Block]:
    lines = text.split("\n")
    tokens = _MD.parse(text)
    headings: list[tuple[int, str, str | None]] = []  # (level, inline markdown, explicit anchor)
    # Blocks with their source line span; a heading is first stored as an index into `headings`,
    # because its anchor can only be assigned once every explicit anchor on the page is known.
    pending: list[tuple[Block | int, int, int]] = []
    dropped = 0

    for i, token in enumerate(tokens):
        # Only top-level block tokens; their map covers any nested content.
        if token.level != 0 or token.nesting < 0 or token.map is None:
            continue
        start, end = token.map
        source = _slice(lines, start, end)
        match token.type:
            case "heading_open":
                inline_md, anchor = split_heading_anchor(tokens[i + 1].content)
                headings.append((int(token.tag[1]), inline_md, anchor))
                pending.append((len(headings) - 1, start, end))
            case "fence":
                lang = token.info.split()[0].lower() if token.info.strip() else ""
                pending.append((CodeBlock(markdown=source, lang=lang), start, end))
            case "code_block":
                pending.append((CodeBlock(markdown=source, lang=""), start, end))
            case "hr":
                continue
            case "html_block" if _is_unrenderable_html(token):
                dropped += 1
            case "bullet_list_open" | "ordered_list_open":
                items = tuple(_slice(lines, *span) for span in _list_item_spans(tokens, i))
                pending.append((TextBlock(markdown=source, kind="list", items=items), start, end))
            case _:
                kind = _TEXT_KINDS.get(token.type, "paragraph")
                pending.append((TextBlock(markdown=source, kind=kind), start, end))

    if dropped:
        logger.debug("html blocks dropped", extra={"page": source_path, "count": dropped})
    pending = _merge_prose_containers(pending, lines)
    heading_blocks = _heading_blocks(headings)
    return [heading_blocks[item] if isinstance(item, int) else item for item, _, _ in pending]


def _drop_noise_code(blocks: list[Block], source_path: str) -> list[Block]:
    """Keep the first of identical code blocks on a page; drop those with embedded binary data."""
    seen: set[str] = set()
    kept: list[Block] = []
    repeats = blobs = 0
    for block in blocks:
        if isinstance(block, CodeBlock):
            if _BLOB_RE.search(block.markdown):
                blobs += 1
                continue
            if block.markdown in seen:
                repeats += 1
                continue
            seen.add(block.markdown)
        kept.append(block)
    if repeats or blobs:
        logger.debug(
            "code blocks dropped",
            extra={"page": source_path, "repeats": repeats, "blobs": blobs},
        )
    return kept


def _merge_prose_containers(
    pending: list[tuple[Block | int, int, int]], lines: list[str]
) -> list[tuple[Block | int, int, int]]:
    """Turn each ``/// type`` … ``///`` container holding only text into one text block.

    Otherwise the chunker sees the "/// tip" marker, the body and the closing "///" as separate
    paragraphs and may cut a tip away from its marker. A container with code or a heading inside
    stays flat, so the code block keeps its own (atomic) block.
    """
    groups: list[tuple[int, int]] = []  # (open index, close index) into `pending`
    stack: list[tuple[str, int]] = []
    for index, (item, _, _) in enumerate(pending):
        if not isinstance(item, TextBlock):
            continue
        first_line = item.markdown.split("\n", 1)[0].strip()
        if (opening := _CONTAINER_OPEN_RE.match(first_line)) is not None:
            stack.append((opening["fence"], index))
        elif stack and first_line == item.markdown.strip() == stack[-1][0]:
            _, open_index = stack.pop()
            inner = pending[open_index : index + 1]
            if all(isinstance(block, TextBlock) for block, _, _ in inner):
                groups.append((open_index, index))

    merged: list[tuple[Block | int, int, int]] = []
    index = 0
    for open_index, close_index in sorted(groups):
        if open_index < index:
            continue  # nested in a container that was already merged
        merged.extend(pending[index:open_index])
        start, end = pending[open_index][1], pending[close_index][2]
        container = TextBlock(markdown=_slice(lines, start, end), kind="container")
        merged.append((container, start, end))
        index = close_index + 1
    merged.extend(pending[index:])
    return merged


def _list_item_spans(tokens: list[Token], list_index: int) -> list[tuple[int, int]]:
    """Source line spans of the top-level items of the list opened at ``tokens[list_index]``."""
    opening = tokens[list_index]
    spans: list[tuple[int, int]] = []
    for token in tokens[list_index + 1 :]:
        if token.level == opening.level and token.nesting < 0:
            break  # the list's own closing token
        if token.type == "list_item_open" and token.level == opening.level + 1 and token.map:
            spans.append((token.map[0], token.map[1]))
    return spans


def _slice(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end]).strip("\n")


def _heading_blocks(headings: list[tuple[int, str, str | None]]) -> list[HeadingBlock]:
    """Assign anchors the way Python-Markdown does: explicit ids are reserved first, then generated
    slugs get ``_1``, ``_2``… suffixes when they collide with an id already on the page."""
    used = {anchor for _, _, anchor in headings if anchor}
    blocks: list[HeadingBlock] = []
    for level, inline_md, explicit in headings:
        text = _plain_text(inline_md)
        anchor = explicit or _unique(slugify(text), used)
        markdown = f"{'#' * level} {inline_md}"
        blocks.append(HeadingBlock(level=level, text=text, anchor=anchor, markdown=markdown))
    return blocks


def _unique(anchor: str, used: set[str]) -> str:
    while anchor in used or not anchor:
        match = _ID_COUNT_RE.match(anchor)
        anchor = f"{match['base']}_{int(match['n']) + 1}" if match else f"{anchor}_1"
    used.add(anchor)
    return anchor


def _plain_text(inline_markdown: str) -> str:
    """Visible text of inline Markdown: code spans and text kept, markup and HTML tags dropped."""
    parts: list[str] = []
    for token in _MD.parseInline(inline_markdown):
        for child in token.children or []:
            if child.type in {"text", "code_inline", "text_special"}:
                parts.append(child.content)
            elif child.type in {"softbreak", "hardbreak"}:
                parts.append(" ")
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def _is_unrenderable_html(token: Token) -> bool:
    """HTML comments aren't rendered, and HTML with Jinja expands data files we don't have
    (e.g. the sponsor grids on the home page)."""
    content = token.content.strip()
    is_comment = content.startswith("<!--") and content.endswith("-->")
    return is_comment or "{%" in content or "{{" in content
