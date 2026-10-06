"""Context builder: ranked chunks -> ``<source>`` blocks and the label map (Tech.md §7, §9.3).

``build_context`` takes the retrieval output (rank order, best first), keeps the first ``k_context``
and renders them as the ``{{sources}}`` value of the answer prompt. The labels ``c1..cK`` exist for
this request only: the model cites labels, the server maps them back through ``BuiltContext.labels``
to DB-sourced chunks and URLs (AGENTS.md §6.3). The model never sees chunk ids.

**Order and labels.** Parts of one split section share a ``section_id``. When more than one part is
selected, the worse-ranked parts are pulled right behind the best-ranked part and put in document
order, so the model reads the section as it was written. Labels are assigned *after* that, in the
final render order, so they always ascend in the prompt and the label map stays one chunk per label.
Document order within a section is ``chunk_id`` ascending: ``RetrievedChunk`` carries no
``ordinal``, and ingest inserts a document's chunks by ordinal into an identity column
(``ingest/pipeline.py``, ``_store``), which ``tests/integration/test_ingest.py`` pins. It is a
property of ingest, not of the schema.

**Escaping.** Chunk content is data, and it is documentation: it may legitimately contain markup,
even a page about ``<source>``. Only the framing is protected:

- in content, a ``<`` that starts ``<source`` or ``</source`` (any case, optional whitespace inside)
  becomes ``&lt;``, so content cannot close its block or open a fake one. Everything else, ``&``
  included, is left alone: the prompt tells the model to copy code faithfully, and rewriting every
  ``&`` or ``<`` would corrupt it. The mapping is therefore one-way, which is fine because only the
  model reads it;
- in attribute values (``section``, ``url``), ``& " <`` become entities, and whitespace runs in
  ``section`` collapse to one space, so a heading cannot break the one-line opening tag. ``>`` stays
  as is: it is legal inside a quoted value and it is the breadcrumb separator in Tech §9.3.

The output is a pure function of the input, which the eval LLM cache (4.03) relies
on: it is keyed by the full rendered prompt.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from grounded.retrieval.types import RetrievedChunk

# ``schemas.llm.CitationLabel`` is ``^c[1-9]$``: a tenth label could never be cited.
MAX_CONTEXT_CHUNKS = 9

_BLOCK_SEPARATOR = "\n\n"
_FRAMING_OPEN = re.compile(r"<(?=\s*/?\s*source)", re.IGNORECASE)
_ATTRIBUTE_ESCAPES = {"&": "&amp;", '"': "&quot;", "<": "&lt;"}
_ATTRIBUTE_ESCAPE_RE = re.compile('[&"<]')
_WHITESPACE_RUN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class BuiltContext:
    """The rendered sources and what each label stands for."""

    text: str  # value for the ``{{sources}}`` placeholder; "" when there are no chunks
    labels: Mapping[str, RetrievedChunk]  # "c1" -> chunk, in render order


def build_context(chunks: Sequence[RetrievedChunk], *, k_context: int) -> BuiltContext:
    """Select, order, label and render the context for one request.

    ``chunks`` must be in rank order, best first (the retrieval contract). Fewer than
    ``k_context`` chunks is fine; the labels then stop at the last chunk.
    """
    if not 1 <= k_context <= MAX_CONTEXT_CHUNKS:
        raise ValueError(f"k_context must be in 1..{MAX_CONTEXT_CHUNKS}, got {k_context}")

    ordered = _group_by_section(chunks[:k_context])
    labels = {f"c{n}": chunk for n, chunk in enumerate(ordered, start=1)}
    text = _BLOCK_SEPARATOR.join(_render_block(label, chunk) for label, chunk in labels.items())
    return BuiltContext(text=text, labels=MappingProxyType(labels))


def _group_by_section(selected: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Sections in order of their best-ranked chunk; each section's parts in document order."""
    groups: dict[str, list[RetrievedChunk]] = {}  # dicts keep first-insertion order
    for chunk in selected:
        groups.setdefault(chunk.section_id, []).append(chunk)
    return [
        chunk
        for parts in groups.values()
        for chunk in sorted(parts, key=lambda part: part.chunk_id)
    ]


def _render_block(label: str, chunk: RetrievedChunk) -> str:
    section = escape_attribute(_WHITESPACE_RUN.sub(" ", chunk.breadcrumb_text).strip())
    url = escape_attribute(chunk.url)
    body = escape_content(chunk.content.strip("\n"))
    return f'<source id="{label}" section="{section}" url="{url}">\n{body}\n</source>'


def escape_content(text: str) -> str:
    """Neutralize ``<source`` / ``</source`` so chunk text cannot break the block framing."""
    return _FRAMING_OPEN.sub("&lt;", text)


def escape_attribute(value: str) -> str:
    return _ATTRIBUTE_ESCAPE_RE.sub(lambda m: _ATTRIBUTE_ESCAPES[m[0]], value)
