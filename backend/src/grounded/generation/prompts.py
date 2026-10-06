"""Versioned prompt files (Tech.md §9.2, AGENTS.md §6.9).

A prompt is one Markdown file in ``backend/prompts/`` with exactly two level-1 sections, in this
order:

    # System          the instructions; static text, no placeholders
    # User template   the per-request message; ``{{name}}`` placeholders

Nothing but blank lines may precede ``# System``, so the file is what the model gets. Other headings
(``##`` and deeper) inside a section are ordinary text.

``prompt_version`` is ``"<file stem>@<first 8 hex of sha256(file)>"``, computed from the file at
load time, so an edit cannot go unversioned. The hash is taken over the text with CRLF read as LF,
like the migration checksums (DB.md §10), so a Windows checkout and CI agree. The rendered prompt is
normalized the same way, which keeps the eval LLM cache key (Phase 4) identical across systems.

``answer_v1`` citation marker grammar (3.06 parses it; the prompt gives the model the same rules):

    marker := "[" label "]"          label := "c" digit-1-9       e.g. [c1]

One label per bracket pair, so two sources are ``[c1][c2]`` and never ``[c1, c2]``. No spaces, no
``[1]`` form, no markers inside fenced code blocks. ``citation_ids`` in a claim hold the bare labels
(``"c1"``), matching ``schemas.llm.CitationLabel``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

# This file is backend/src/grounded/generation/prompts.py, so parents[3] is backend/.
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

ANSWER_PLACEHOLDERS = frozenset({"question", "sources"})

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]*)\}\}")
_SYSTEM_HEADING = "# System"
_USER_HEADING = "# User template"
_VERSION_HEX_LEN = 8


class PromptError(Exception):
    """Base class for prompt failures."""


class PromptLoadError(PromptError):
    """The file is missing or malformed: caught the first time the prompt is loaded."""


class PromptRenderError(PromptError):
    """The variables passed to ``render_user`` do not match the template's placeholders."""


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: str  # "<name>@<8 hex>", recorded in every request log and eval run
    system: str
    user_template: str
    placeholders: frozenset[str]

    def render_user(self, **variables: str) -> str:
        """Fill the user template. The variables must match its placeholders exactly.

        Substitution is a single pass, so a value that itself contains ``{{sources}}`` (a question
        typed by a user, a code sample in a source) is inserted as text and never expanded.
        """
        missing = self.placeholders - variables.keys()
        unknown = variables.keys() - self.placeholders
        if missing or unknown:
            raise PromptRenderError(
                f"{self.name}: missing variables {sorted(missing)}, "
                f"unknown variables {sorted(unknown)}"
            )
        return _PLACEHOLDER_RE.sub(lambda m: variables[m[1]], self.user_template)


def prompt_version(name: str, text: str) -> str:
    normalized = _normalize(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{name}@{digest[:_VERSION_HEX_LEN]}"


def parse_prompt(name: str, text: str, *, placeholders: Collection[str]) -> Prompt:
    """Split, validate and version prompt text. ``load_prompt`` is this plus reading the file.

    ``placeholders`` is what the caller will pass to ``render_user``. The user template must use
    exactly that set: a typo (``{{sourcs}}``) or a dropped placeholder fails here, not at request
    time.
    """
    expected = frozenset(placeholders)
    normalized = _normalize(text)
    system, user_template = _split_sections(name, normalized)

    if "{{" in system or "}}" in system:
        raise PromptLoadError(f"{name}: the system section must not contain placeholders")

    # Whatever braces remain after removing well-formed placeholders are typos like "{{question}".
    leftover = _PLACEHOLDER_RE.sub("", user_template)
    if "{{" in leftover or "}}" in leftover:
        raise PromptLoadError(f"{name}: unbalanced double braces in the user template")
    found = frozenset(_PLACEHOLDER_RE.findall(user_template))
    if found != expected:
        raise PromptLoadError(
            f"{name}: user template placeholders {sorted(found)} do not match the expected "
            f"{sorted(expected)} (unknown: {sorted(found - expected)}, "
            f"missing: {sorted(expected - found)})"
        )

    return Prompt(name, prompt_version(name, text), system, user_template, found)


def load_prompt(
    name: str,
    *,
    placeholders: Collection[str],
    directory: Path = DEFAULT_PROMPTS_DIR,
) -> Prompt:
    if not _NAME_RE.match(name):
        raise PromptLoadError(f"bad prompt name {name!r}: expected lowercase letters, digits, _")
    path = directory / f"{name}.md"
    try:
        # Bytes, not read_text: decide the line-ending handling ourselves, in one place.
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        raise PromptLoadError(f"prompt file not found: {path}") from None
    except UnicodeDecodeError as exc:
        raise PromptLoadError(f"{path} is not valid UTF-8: {exc}") from None
    return parse_prompt(name, text, placeholders=placeholders)


def load_answer_prompt(directory: Path = DEFAULT_PROMPTS_DIR) -> Prompt:
    return load_prompt("answer_v1", placeholders=ANSWER_PLACEHOLDERS, directory=directory)


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n")


def _split_sections(name: str, text: str) -> tuple[str, str]:
    lines = text.split("\n")
    system_at = [i for i, line in enumerate(lines) if line.rstrip() == _SYSTEM_HEADING]
    user_at = [i for i, line in enumerate(lines) if line.rstrip() == _USER_HEADING]
    if len(system_at) != 1:
        raise PromptLoadError(f"{name}: expected exactly one {_SYSTEM_HEADING!r} heading")
    if len(user_at) != 1:
        raise PromptLoadError(f"{name}: expected exactly one {_USER_HEADING!r} heading")
    if system_at[0] > user_at[0]:
        raise PromptLoadError(f"{name}: {_SYSTEM_HEADING!r} must come before {_USER_HEADING!r}")
    if any(line.strip() for line in lines[: system_at[0]]):
        raise PromptLoadError(f"{name}: nothing but blank lines may precede {_SYSTEM_HEADING!r}")

    system = "\n".join(lines[system_at[0] + 1 : user_at[0]]).strip()
    user_template = "\n".join(lines[user_at[0] + 1 :]).strip()
    if not system:
        raise PromptLoadError(f"{name}: the system section is empty")
    if not user_template:
        raise PromptLoadError(f"{name}: the user template section is empty")
    return system, user_template
