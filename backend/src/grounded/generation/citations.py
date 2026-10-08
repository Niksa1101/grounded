"""Citation validation, mapping and marker rewriting (Tech.md §9.5 step 2, §9.6).

The model cites per-request labels (``c1..cK``); this module is where they become something a reader
can trust. A label is *valid* when it is in this request's label map. Everything a response shows
(URL, title, breadcrumb, snippet) comes from the DB row behind a valid label, never from the model
(AGENTS.md §6.3). Invalid labels are removed, **counted** and never silently repaired, because the
rate of invalid labels is a quality metric (Tech §9.6).

**Marker grammar.** ``[cN]``, one label per bracket pair, and no markers inside fenced code
(Tech §9.2). A marker is any ``[c<digits>]``, so ``[c0]`` and ``[c12]`` are markers with an invalid
label rather than text that slips through. The one malformed shape that is also handled is a
bracket pair holding several labels (``[c1, c2]``, ``[c1;c2]``): it is removed and counted once, as
a single invalid reference. Anything else is prose and is left alone.

**Fenced code is left untouched.** Inside a fenced block ``[c1]`` is code, not a citation: it is
neither rewritten nor counted. A fence opens on a line starting with at least three backticks or
tildes (any leading indentation, so fences nested in a list item work) and closes on a line of the
same character at least as long, as in CommonMark. An unclosed fence runs to the end of the text.
Inline code spans are not treated specially: the prompt forbids markers there, and a marker in a
span is a model mistake that is counted like any other.

**Display numbers.** ``n`` follows the first appearance of a valid label in ``answer_markdown``.
A label cited only by a claim comes after those, in claim order. Claims map to the same numbers,
and ``citations`` is ordered by ``n``, one entry per distinct cited chunk.

**``invalid_citation_count``** is the number of removed references: every invalid marker in the
text plus every invalid label in a claim's ``citation_ids`` (the same bad label in both places
counts twice). Labels of claims that are dropped as a whole (below) are not counted again.

**Semantic checks** (they report; the caller decides what to do, 3.07 retries on them):

- ``answered`` / ``partial`` with zero valid citations is bad output: ``MappedAnswer.bad_output``
  holds a reason written to be quoted back to the model.
- ``insufficient_context`` must have no claims: they are dropped and counted
  (``dropped_claim_count``).
- ``require_citations=False`` (the ``no_rag`` mode, 3.11) turns the zero-valid-citations check off:
  there are no sources, so ``labels`` is empty and nothing can be cited. Markers the model writes
  anyway are removed and counted as invalid, as for any label that is not in ``labels``.

**A refusal is citation-free** (Tech §9.7, 3.11). In an ``insufficient_context`` answer every marker
is removed from the text, whether its label is valid or not, and the answer has no ``citations``.
Only the markers with an invalid label are counted in ``invalid_citation_count``; a valid label
there is not an error of the model's, it just has no place in a refusal.

**No URLs from the model** (AGENTS.md §6.3, PRD D47). The prompt forbids them, and this module
enforces it on ``answer_markdown`` (``strip_urls``, before the markers are rewritten): a Markdown
link or image keeps only its text, a link reference definition is removed (otherwise ``[1]``, a
rewritten marker, could be a link to the host it defines), an autolink or a bare URL is removed,
and every removal is counted (``removed_url_count``). The URLs a reader can click are the
DB-sourced ``citations``. Loopback URLs (``http://127.0.0.1:8000/docs``) are kept but wrapped in
inline code: the FastAPI docs tell readers to open them, and in code they cannot become links.
Fenced code and inline code spans are left alone (unlike markers, which are rewritten in inline
code, see above): a URL in code is not a link. Removing a URL is not bad output and causes no retry.
It is a regex, not a Markdown parser, so it is the first layer and not the guarantee (Tech §12).

Snippets are the first ``SNIPPET_CHARS`` characters of the chunk, cut back to a word boundary
(see ``make_snippet``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from grounded.retrieval.types import RetrievedChunk
from grounded.schemas.api import Citation
from grounded.schemas.llm import LLMAnswer

SNIPPET_CHARS = 300

# Group 1 is the label of a well-formed marker; it is None for a bracket pair with several labels.
_MARKER = re.compile(r"\[(c\d+)\]|\[c\d+(?:\s*[,;]\s*c\d+)+\]")
_FENCE = re.compile(r"[ \t]*(`{3,}|~{3,})(.*)")

# An inline code span: a run of backticks, then the shortest text up to a run of the same length.
_CODE_SPAN = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)")
# ``[text](destination "title")`` or ``![alt](…)``. Group 1 is the text, with one level of nested
# brackets. The destination is ``<…>`` or has balanced parentheses (one level, as in
# ``javascript:alert(1)``); the title is ``"…"``, ``'…'`` or ``(…)``. Deeper shapes are left to
# ``_cut_destinations``.
_LINK = re.compile(
    r"!?\[((?:[^\[\]\n]|\[[^\[\]\n]*\])*)\]"
    r"\(\s*(?:<[^<>\n]*>|(?:[^\s()<>]|\([^\s()]*\))*)"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^()]*\)))?\s*\)"
)
# A link reference definition (CommonMark 0.31 §4.7) after optional ``>`` and list-item markers:
# ``[label]:``, an optional destination, an optional title (it may continue on the next line, so
# its closing quote is optional). The destination may be on the next line too, so a bare
# ``[label]:`` matches. Prose such as ``[c1]: the dependency runs first.`` does not: words after
# the destination are not a title. Matched against a whole line; ``prefix`` holds the container
# markers and ``end`` the line ending.
_DEFINITION = re.compile(
    r"(?P<prefix>(?:[ \t]*(?:>|[-+*][ \t]|\d{1,9}[.)][ \t]))*)[ \t]*"
    r"\[(?:[^\[\]\\\n]|\\.)+\]:[ \t]*(?:<[^<>\n]*>|[^\s<]\S*)?"
    r"(?:[ \t]+(?:\"[^\"\n]*\"?|'[^'\n]*'?|\([^()\n]*\)?))?[ \t]*(?P<end>\r?\n?)"
)
# ``<http://…>`` autolinks and bare ``http(s)://…`` / ``www.…`` URLs, with one space before them so
# the removal does not leave a double space. Group 1 is the space, group 2 the URL.
# Case-insensitive, as the scheme and ``www.`` are for a Markdown renderer (``HTTPS://`` links too).
_URL = re.compile(r"( ?)<?((?:https?://|www\.)[^\s<>()\[\]`]+)>?", re.IGNORECASE)
# The other CommonMark autolinks: ``<scheme:…>`` (a letter, then 1-31 letters, digits, ``+.-``)
# and ``<user@host>``. Matched after ``_URL``, so an http(s) loopback autolink is wrapped instead.
_AUTOLINK = re.compile(r"( ?)<(?:[A-Za-z][A-Za-z0-9+.\-]{1,31}:[^\s<>]*|[^\s<>@]+@[^\s<>@]+)>")
_TRAILING_PUNCTUATION = ".,;:!?"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})


@dataclass(frozen=True, slots=True)
class MappedClaim:
    """A claim with its citations as display numbers. Confidence is added later (3.10)."""

    text: str
    citations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MappedAnswer:
    answer_markdown: str  # markers rewritten to [n], invalid ones removed, URLs removed
    claims: tuple[MappedClaim, ...]
    citations: tuple[Citation, ...]  # ordered by n
    invalid_citation_count: int
    dropped_claim_count: int
    removed_url_count: int  # links and URLs the model wrote (strip_urls)
    bad_output: str | None  # why the answer is unusable, or None when it passed the checks


def map_citations(
    answer: LLMAnswer,
    labels: Mapping[str, RetrievedChunk],
    titles: Mapping[int, str],
    *,
    require_citations: bool = True,
) -> MappedAnswer:
    """Validate the labels of ``answer`` against this request's ``labels`` and build the citations.

    ``titles`` maps ``chunk_id`` to the page title for every chunk behind ``labels``.
    ``require_citations=False`` skips the zero-valid-citations check (``no_rag``, see above).
    """
    refusal = answer.status == "insufficient_context"
    numbers: dict[str, int] = {}  # label -> display number, in order of first use
    invalid = 0

    def number(label: str) -> int:
        return numbers.setdefault(label, len(numbers) + 1)

    def rewrite(match: re.Match[str]) -> str:
        nonlocal invalid
        label = match[1]
        if label is None or label not in labels:
            invalid += 1
            return ""
        if refusal:
            return ""  # a valid label, but a refusal cites nothing
        return f"[{number(label)}]"

    stripped, removed_urls = strip_urls(answer.answer_markdown)
    markdown = rewrite_markers(stripped, rewrite)

    claims: list[MappedClaim] = []
    dropped = 0
    if refusal:
        dropped = len(answer.claims)
    else:
        for claim in answer.claims:
            cited: dict[int, None] = {}  # ordered set: a repeated label is one citation
            for label in claim.citation_ids:
                if label in labels:
                    cited[number(label)] = None
                else:
                    invalid += 1
            claims.append(MappedClaim(text=claim.text, citations=tuple(cited)))

    citations = tuple(_citation(n, labels[label], titles) for label, n in numbers.items())
    bad_output = None
    if require_citations and not refusal and not citations:
        bad_output = (
            f"the status is '{answer.status}' but the answer cites no valid source label; "
            "cite the sources you used with markers such as [c1], using only the labels given"
        )
    return MappedAnswer(
        answer_markdown=markdown,
        claims=tuple(claims),
        citations=citations,
        invalid_citation_count=invalid,
        dropped_claim_count=dropped,
        removed_url_count=removed_urls,
        bad_output=bad_output,
    )


def rewrite_markers(markdown: str, replace: Callable[[re.Match[str]], str]) -> str:
    """Apply ``replace`` to every citation marker outside fenced code, in text order."""
    return _outside_fences(markdown, lambda line: _MARKER.sub(replace, line))


def strip_urls(markdown: str) -> tuple[str, int]:
    """Remove the links and URLs of ``markdown`` outside code; return the text and how many.

    A link or image becomes its text (a marker text such as ``c1`` stays ``[c1]``, so it is still
    a citation); a link reference definition line is removed; a ``](…)`` whose ``[`` is not in the
    same part of the line (a code span in the link text, or text over two lines) loses its
    destination; an autolink or a bare URL is removed together with the space before it; a
    loopback URL is wrapped in inline code instead and is not counted. Each removal counts once: a
    URL in a link's text is part of that link. Sentence punctuation right after a bare URL stays in
    the text. Inline code spans are matched per line.
    """
    removed = 0

    def link(match: re.Match[str]) -> str:
        nonlocal removed
        removed += 1
        text = match[1]
        # Any marker form, a group like ``c1, c2`` too: the rewrite then judges it as usual.
        if _MARKER.fullmatch(f"[{text}]"):
            return f"[{text}]"
        # A link or an image inside the text is a link of its own; a URL there belongs to this one,
        # already counted (a loopback URL stays, for the pass below to wrap).
        return _URL.sub(_drop_unless_loopback, _LINK.sub(link, text))

    def url(match: re.Match[str]) -> str:
        nonlocal removed
        space, target, kept = _split_url(match)
        if _is_loopback(target):
            return f"{space}`{target}`{kept}"
        removed += 1
        return kept

    def autolink(_match: re.Match[str]) -> str:
        nonlocal removed
        removed += 1
        return ""

    def text(part: str) -> str:
        nonlocal removed
        part, cut = _cut_destinations(_LINK.sub(link, part))
        removed += cut
        return _AUTOLINK.sub(autolink, _URL.sub(url, part))

    def line(markdown_line: str) -> str:
        nonlocal removed
        if (definition := _DEFINITION.fullmatch(markdown_line)) is not None:
            removed += 1
            return definition["prefix"].rstrip(" \t") + definition["end"]
        return _outside_code_spans(markdown_line, text)

    stripped = _outside_fences(markdown, line)
    return stripped, removed


def _split_url(match: re.Match[str]) -> tuple[str, str, str]:
    """A ``_URL`` match as (space before, URL, sentence punctuation after it)."""
    space, target = match[1], match[2]
    kept = ""
    while target and target[-1] in _TRAILING_PUNCTUATION:
        target, kept = target[:-1], target[-1] + kept
    return space, target, kept


def _drop_unless_loopback(match: re.Match[str]) -> str:
    _, target, kept = _split_url(match)
    return match[0] if _is_loopback(target) else kept


def _cut_destinations(part: str) -> tuple[str, int]:
    """Remove every ``(…)`` that follows a ``]`` in ``part``; return the text and how many.

    Runs after ``_LINK``, so what is left is a link ``_LINK`` could not see whole: its text holds
    a code span or runs over two lines (the ``[`` is in another part), or nests deeper than
    ``_LINK`` knows. Outside code a ``](`` only ever opens a link destination. The cut runs to the
    matching ``)``, or to the end of the part when there is none (the destination continues on
    the next line, which is then plain text).
    """
    out: list[str] = []
    count, done = 0, 0
    while (start := part.find("](", done)) != -1:
        depth, end = 1, start + 2
        while end < len(part) and depth:
            depth += {"(": 1, ")": -1}.get(part[end], 0)
            end += 1
        if depth:
            end = len(part.rstrip("\r\n"))
        out.append(part[done : start + 1])  # up to and with the ``]``
        done = end
        count += 1
    out.append(part[done:])
    return "".join(out), count


def _is_loopback(url: str) -> bool:
    try:
        host = urlsplit(url if "://" in url else f"http://{url}").hostname
    except ValueError:  # a malformed URL (e.g. an unclosed IPv6 bracket) is not loopback
        return False
    return host in _LOOPBACK_HOSTS


def _outside_code_spans(line: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to the parts of ``line`` that are not inline code spans."""
    out: list[str] = []
    start = 0
    for span in _CODE_SPAN.finditer(line):
        out.append(transform(line[start : span.start()]))
        out.append(span[0])
        start = span.end()
    out.append(transform(line[start:]))
    return "".join(out)


def _outside_fences(markdown: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to every line outside fenced code, in text order."""
    out: list[str] = []
    fence: tuple[str, int] | None = None  # (character, length) of the open fence
    for line in markdown.splitlines(keepends=True):
        opening = _FENCE.match(line)
        if fence is None:
            if opening and not (opening[1][0] == "`" and "`" in opening[2]):
                fence = (opening[1][0], len(opening[1]))
                out.append(line)
            else:
                out.append(transform(line))
        else:
            if (
                opening
                and opening[1][0] == fence[0]
                and len(opening[1]) >= fence[1]
                and not opening[2].strip()
            ):
                fence = None
            out.append(line)
    return "".join(out)


def make_snippet(content: str, limit: int = SNIPPET_CHARS) -> str:
    """The first ``limit`` characters of ``content``, cut back to a word boundary.

    Whitespace at the edges is stripped. Text that fits is returned whole. Otherwise the cut lands
    after the last whitespace inside the first ``limit`` characters, so a word is never split; a
    single unbroken run longer than ``limit`` (a URL, a long identifier) is cut at ``limit``.
    No ellipsis is added: showing truncation is the UI's job.
    """
    text = content.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if not text[limit].isspace():
        boundary = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
        if boundary > 0:
            cut = cut[:boundary]
    return cut.rstrip()


def _citation(n: int, chunk: RetrievedChunk, titles: Mapping[int, str]) -> Citation:
    return Citation.model_validate(
        {
            "n": n,
            "chunk_id": chunk.chunk_id,
            "url": chunk.url,
            "title": titles[chunk.chunk_id],
            "breadcrumb": chunk.breadcrumb_text,
            "snippet": make_snippet(chunk.content),
        }
    )
