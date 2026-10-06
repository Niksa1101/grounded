"""Versioned prompt files (Tech.md §9.2, AGENTS.md §6.9).

A prompt is one Markdown file in ``backend/prompts/`` with two required level-1 sections and one
optional, in this order:

    # System          the instructions; static text, no placeholders
    # User template   the per-request message; ``{{name}}`` placeholders
    # Retry feedback  optional; appended to the user message on the one retry after invalid output
                      (Tech §9.5 step 3); ``{{name}}`` placeholders

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
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

# This file is backend/src/grounded/generation/prompts.py, so parents[3] is backend/.
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

ANSWER_PLACEHOLDERS = frozenset({"question", "sources"})
ANSWER_RETRY_PLACEHOLDERS = frozenset({"error"})

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER_RE = re.compile(r"\{\{([^{}]*)\}\}")
_SYSTEM_HEADING = "# System"
_USER_HEADING = "# User template"
_RETRY_HEADING = "# Retry feedback"
_VERSION_HEX_LEN = 8


class PromptError(Exception):
    """Base class for prompt failures."""


class PromptLoadError(PromptError):
    """The file is missing or malformed: caught the first time the prompt is loaded."""


class PromptRenderError(PromptError):
    """The variables passed to ``render_user`` / ``render_retry`` do not match the template's
    placeholders, or the prompt has no retry section."""


@dataclass(frozen=True, slots=True)
class Prompt:
    name: str
    version: str  # "<name>@<8 hex>", recorded in every request log and eval run
    system: str
    user_template: str
    placeholders: frozenset[str]
    retry_template: str | None = None  # None: the file has no "# Retry feedback" section
    retry_placeholders: frozenset[str] = frozenset()

    def render_user(self, **variables: str) -> str:
        """Fill the user template. The variables must match its placeholders exactly.

        Substitution is a single pass, so a value that itself contains ``{{sources}}`` (a question
        typed by a user, a code sample in a source) is inserted as text and never expanded.
        """
        return _fill(self.name, self.user_template, self.placeholders, variables)

    def render_retry(self, user: str, **variables: str) -> str:
        """The retry's user message: the original one, then the feedback section filled in.

        Same rules as ``render_user``, so a validation error that quotes model output containing
        ``{{...}}`` is inserted as text.
        """
        if self.retry_template is None:
            raise PromptRenderError(f"{self.name}: the prompt has no '# Retry feedback' section")
        feedback = _fill(self.name, self.retry_template, self.retry_placeholders, variables)
        return f"{user}\n\n{feedback}"


def prompt_version(name: str, text: str) -> str:
    normalized = _normalize(text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{name}@{digest[:_VERSION_HEX_LEN]}"


def parse_prompt(
    name: str,
    text: str,
    *,
    placeholders: Collection[str],
    retry_placeholders: Collection[str] = (),
) -> Prompt:
    """Split, validate and version prompt text. ``load_prompt`` is this plus reading the file.

    ``placeholders`` is what the caller will pass to ``render_user``. The user template must use
    exactly that set: a typo (``{{sourcs}}``) or a dropped placeholder fails here, not at request
    time. The retry section is optional; when the file has one it must use exactly
    ``retry_placeholders``, by the same rule.
    """
    normalized = _normalize(text)
    system, user_template, retry_template = _split_sections(name, normalized)

    if "{{" in system or "}}" in system:
        raise PromptLoadError(f"{name}: the system section must not contain placeholders")

    found = _check_placeholders(name, "user template", user_template, placeholders)
    retry_found: frozenset[str] = frozenset()
    if retry_template is not None:
        retry_found = _check_placeholders(
            name, "retry feedback", retry_template, retry_placeholders
        )

    return Prompt(
        name, prompt_version(name, text), system, user_template, found, retry_template, retry_found
    )


def load_prompt(
    name: str,
    *,
    placeholders: Collection[str],
    retry_placeholders: Collection[str] = (),
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
    return parse_prompt(
        name, text, placeholders=placeholders, retry_placeholders=retry_placeholders
    )


def load_answer_prompt(directory: Path = DEFAULT_PROMPTS_DIR) -> Prompt:
    return load_prompt(
        "answer_v1",
        placeholders=ANSWER_PLACEHOLDERS,
        retry_placeholders=ANSWER_RETRY_PLACEHOLDERS,
        directory=directory,
    )


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n")


def _fill(
    name: str, template: str, placeholders: frozenset[str], variables: Mapping[str, str]
) -> str:
    missing = placeholders - variables.keys()
    unknown = variables.keys() - placeholders
    if missing or unknown:
        raise PromptRenderError(
            f"{name}: missing variables {sorted(missing)}, unknown variables {sorted(unknown)}"
        )
    return _PLACEHOLDER_RE.sub(lambda m: variables[m[1]], template)


def _check_placeholders(
    name: str, label: str, template: str, placeholders: Collection[str]
) -> frozenset[str]:
    expected = frozenset(placeholders)
    # Whatever braces remain after removing well-formed placeholders are typos like "{{question}".
    leftover = _PLACEHOLDER_RE.sub("", template)
    if "{{" in leftover or "}}" in leftover:
        raise PromptLoadError(f"{name}: unbalanced double braces in the {label}")
    found = frozenset(_PLACEHOLDER_RE.findall(template))
    if found != expected:
        raise PromptLoadError(
            f"{name}: {label} placeholders {sorted(found)} do not match the expected "
            f"{sorted(expected)} (unknown: {sorted(found - expected)}, "
            f"missing: {sorted(expected - found)})"
        )
    return found


def _split_sections(name: str, text: str) -> tuple[str, str, str | None]:
    lines = text.split("\n")
    system_at = [i for i, line in enumerate(lines) if line.rstrip() == _SYSTEM_HEADING]
    user_at = [i for i, line in enumerate(lines) if line.rstrip() == _USER_HEADING]
    retry_at = [i for i, line in enumerate(lines) if line.rstrip() == _RETRY_HEADING]
    if len(system_at) != 1:
        raise PromptLoadError(f"{name}: expected exactly one {_SYSTEM_HEADING!r} heading")
    if len(user_at) != 1:
        raise PromptLoadError(f"{name}: expected exactly one {_USER_HEADING!r} heading")
    if len(retry_at) > 1:
        raise PromptLoadError(f"{name}: at most one {_RETRY_HEADING!r} heading")
    if system_at[0] > user_at[0]:
        raise PromptLoadError(f"{name}: {_SYSTEM_HEADING!r} must come before {_USER_HEADING!r}")
    if retry_at and retry_at[0] < user_at[0]:
        raise PromptLoadError(f"{name}: {_RETRY_HEADING!r} must come after {_USER_HEADING!r}")
    if any(line.strip() for line in lines[: system_at[0]]):
        raise PromptLoadError(f"{name}: nothing but blank lines may precede {_SYSTEM_HEADING!r}")

    user_end = retry_at[0] if retry_at else len(lines)
    system = "\n".join(lines[system_at[0] + 1 : user_at[0]]).strip()
    user_template = "\n".join(lines[user_at[0] + 1 : user_end]).strip()
    if not system:
        raise PromptLoadError(f"{name}: the system section is empty")
    if not user_template:
        raise PromptLoadError(f"{name}: the user template section is empty")
    retry_template = None
    if retry_at:
        retry_template = "\n".join(lines[retry_at[0] + 1 :]).strip()
        if not retry_template:
            raise PromptLoadError(f"{name}: the retry feedback section is empty")
    return system, user_template, retry_template
