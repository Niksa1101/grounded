"""Resolve the FastAPI docs' code include directives into literal code (Tech.md §5.3).

The docs keep their examples as real files and pull them in at build time. Two syntaxes exist at the
pinned tag, and both are expanded here, line by line, before Markdown parsing (as the mkdocs
preprocessors do):

- **Current** (``markdown-include-variants``): a line ``{* <path> hl[..] ln[..] title[".."] *}``
  becomes a fenced ``python`` block. The referenced file is the variant the site shows first, so it
  is the one we keep; the other variants sit in a collapsed "Other versions" panel and are skipped.
  ``ln[a:b,c:d]`` keeps those 1-based inclusive ranges and adds the site's "omitted" comments, so a
  reader (and the LLM) can tell the snippet is partial. ``hl[..]`` only highlights and is dropped.
  ``title[".."]`` stays on the fence as ``title="..."`` because it names the file.
- **Legacy** (``mdx_include``): ``{!path!}`` or ``{!> path!}`` inside a fence the page already has;
  the directive line is replaced by the file's lines.

Paths are relative to ``docs/en`` (where mkdocs runs), e.g. ``../../docs_src/x.py``. A path that
doesn't exist or leaves the repository is an ``IncludeError``: ingest fails instead of indexing a
page with a hole where the example should be.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# The directory include paths are relative to (the mkdocs project of the English docs).
INCLUDE_BASE_DIR: Final = "docs/en"

# Same shape as markdown-include-variants' own matcher: the directive is a whole line.
_VARIANT_RE = re.compile(r"^\{\*\s*(?P<path>\S+)\s*(?P<config>.*)\*\}$")
_CONFIG_RE = re.compile(r"(?P<name>\w+)\[(?P<value>[^\]]+)\]")
_LEGACY_RE = re.compile(r"^(?P<indent>\s*)\{!>?\s*(?P<path>[^!]+?)\s*!\}\s*$")
# Rendering-only highlight option on a fence opening line (legacy blocks carry it).
_HL_LINES_RE = re.compile(r'^(?P<head>\s{0,3}(?:`{3,}|~{3,})[^`]*?)\s+hl_lines="[^"]*"')

_ABOVE: Final = "# Code above omitted 👆"
_HERE: Final = "# Code here omitted 👈"
_BELOW: Final = "# Code below omitted 👇"


class IncludeError(Exception):
    """An include directive could not be resolved."""


def resolve_includes(text: str, root: Path, *, source_path: str) -> str:
    """Return ``text`` with every include directive replaced by the included code.

    ``root`` is the repository checkout; ``source_path`` is only used in error messages.
    """
    base = root / INCLUDE_BASE_DIR
    out: list[str] = []
    for number, line in enumerate(text.split("\n"), start=1):
        where = f"{source_path}:{number}"
        if match := _VARIANT_RE.match(line):
            out.extend(_expand_variant(match, root, base, where))
        elif match := _LEGACY_RE.match(line):
            lines = _read_lines(match["path"], root, base, where)
            logger.debug("legacy include resolved", extra={"at": where, "path": match["path"]})
            # Keep the directive's indentation, or a fence inside a list item would break.
            out.extend(f"{match['indent']}{code}" if code else code for code in lines)
        else:
            out.append(_HL_LINES_RE.sub(r"\g<head>", line))
    return "\n".join(out)


def _expand_variant(match: re.Match[str], root: Path, base: Path, where: str) -> list[str]:
    path = match["path"]
    ranges: list[tuple[int, int]] = []
    title: str | None = None
    for option in _CONFIG_RE.finditer(match["config"]):
        name, value = option["name"], option["value"]
        if name == "ln":
            ranges.extend(_parse_ranges(value, where))
        elif name == "title":
            if not (len(value) >= 2 and value.startswith('"') and value.endswith('"')):
                raise IncludeError(f"{where}: title must be quoted, got {value!r}")
            title = value[1:-1]
        elif name != "hl":
            raise IncludeError(f"{where}: unknown include option {name!r}")

    lines = _read_lines(path, root, base, where)
    body = _select(lines, sorted(ranges), where) if ranges else lines
    logger.debug("include resolved", extra={"at": where, "path": path, "ranges": ranges})

    fence = "`" * max(3, _longest_backtick_run(body) + 1)
    info = f'python title="{title}"' if title else "python"
    return [f"{fence}{info}", *body, fence]


def _parse_ranges(value: str, where: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in value.split(","):
        start, _, end = part.strip().partition(":")
        try:
            ranges.append((int(start), int(end or start)))
        except ValueError as exc:
            raise IncludeError(f"{where}: bad line range {part!r}") from exc
    return ranges


def _select(lines: list[str], ranges: list[tuple[int, int]], where: str) -> list[str]:
    # ln[0:0] means "no excerpt": the site then only shows the full file.
    if ranges == [(0, 0)]:
        return lines
    selected: list[str] = []
    for i, (start, end) in enumerate(ranges):
        if not 1 <= start <= end <= len(lines):
            raise IncludeError(f"{where}: line range {start}:{end} outside 1:{len(lines)}")
        if i == 0 and start > 1:
            selected.extend([_ABOVE, ""])
        elif i > 0:
            selected.extend(["", _HERE, ""])
        selected.extend(lines[start - 1 : end])
    if ranges[-1][1] < len(lines):
        selected.extend(["", _BELOW])
    return selected


def _read_lines(relative: str, root: Path, base: Path, where: str) -> list[str]:
    target = (base / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise IncludeError(f"{where}: include {relative!r} points outside the repository")
    if not target.is_file():
        raise IncludeError(f"{where}: included file not found: {relative}")
    text = target.read_text(encoding="utf-8").replace("\r\n", "\n")
    return text.removesuffix("\n").split("\n")


def _longest_backtick_run(lines: list[str]) -> int:
    runs = (len(m.group()) for line in lines for m in re.finditer(r"`+", line))
    return max(runs, default=0)
