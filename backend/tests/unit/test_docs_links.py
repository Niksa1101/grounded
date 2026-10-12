"""Relative links of the README and the MVP closeout runbook point at things that exist (5.12).

The README is what a reader meets first, and it links to runbooks, tickets, screenshots and its
own sections. A renamed file or heading would leave a dead link that nothing else notices. This
reads files only: no network, and http(s) links are not followed. Anchors are checked only inside
the same file, with GitHub's slug rule, which is enough for the README's own contents links.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
FILES = [
    ROOT / "README.md",
    ROOT / "docs" / "runbooks" / "mvp-closeout.md",
    ROOT / "tickets" / "phase-5" / "5.12-readme-v1.md",
]

_FENCE = re.compile(r"^(```|~~~)")
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")


def _prose_lines(text: str) -> list[str]:
    """The lines outside fenced code blocks (a command with brackets is not a link)."""
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line.strip()):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return lines


def _slug(heading: str) -> str:
    """GitHub's anchor for a heading: lowercase, drop punctuation, spaces to hyphens."""
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def _anchors(text: str) -> set[str]:
    return {_slug(m.group(1)) for line in _prose_lines(text) if (m := _HEADING.match(line))}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_relative_link_resolves(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    anchors = _anchors(text)
    broken: list[str] = []
    for line in _prose_lines(text):
        for target in _LINK.findall(line):
            if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
                continue  # http(s), mailto: not followed
            file_part, _, anchor = target.partition("#")
            if not file_part:
                if anchor and anchor not in anchors:
                    broken.append(f"#{anchor} (no such heading)")
                continue
            if not (path.parent / file_part).exists():
                broken.append(f"{file_part} (no such file)")
    assert not broken, f"{path.name} has dead links: {broken}"
