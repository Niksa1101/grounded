# R.17 — Reference links, link leftovers and doc wording

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-3/reference-links` · **Blocked by:** R.16 ·
> **Builds on:** `generation/citations.py`, `generation/params.py`

**What to build:** the findings of the Author's review of R.16 (2026-10-08). The Author asked for all of them to be
fixed. Decisions of the Agent, reported in the PR: markers inside inline code stay markers (only the docstring that
said otherwise changes, since a behavior change there would change citations), and the rerank note goes to PRD §12.

- **🟡 Reference-style links pass `strip_urls`.** It removes only `http(s)://` and `www.` URLs, so a link reference
  definition with another destination survives: `[c1]: //evil.example` stays, the rewrite turns it into
  `[1]: //evil.example`, and in CommonMark the citation marker `[1]` itself becomes a link to that host. The same
  holds for `mailto:` and `<//host>`. A regex can close this one, so the server does (R.16 left only what a regex
  cannot close to the frontend).
- **🟢 Link leftovers.** When `_LINK` does not match, only the bare `http(s)` part is removed, which leaves:
  `docs)` (parentheses in the URL), `[1])` (`javascript:alert(1)`), `[a [b] c]()` (nested brackets: an empty-href
  link), and a link whose text is its URL is counted twice. A link whose text holds a code span, or spans two lines,
  keeps a non-http destination (`[click `here`](//evil.example)`).
- **🟢 `<scheme:…>` and `<user@host>` autolinks** are links in CommonMark. PRD §12 said a regex cannot close them;
  it can.
- **🟢 Wording:** the `citations.py` docstring says inline code spans are left alone "as for markers" (markers in
  inline code are rewritten); Tech §11 and `params.py` say the cache key and the call "cannot disagree", which holds
  for temperature and max output tokens, not for `thinking_level` (PRD §12 already says it is read twice).
- **📚 Phase 6:** invariant 4 and its `Settings` guard assume rerank off. A PRD §12 item says to revisit them when
  `CONFIDENCE_W_RERANK` goes above 0.

**Read first:** Tech §9.6, §11, §12 (output rendering); PRD §12; CommonMark 0.31 "Link reference definitions",
"Links", "Autolinks".

**Scope notes**
- A line that is a link reference definition (after optional `>` / list-item markers: `[label]:`, then an optional
  destination and an optional title) is removed, and counted once. A `[label]:` with nothing after it is removed too,
  because CommonMark allows the destination on the next line. Prose such as `[c1]: the dependency runs first.` is not
  a definition and stays.
- `_LINK` accepts one level of nested brackets in the text, balanced parentheses and `<…>` in the destination, and
  `"…"`, `'…'` or `(…)` titles. A link's text is processed again (a link or an image inside a link) and the URLs in it
  are removed without being counted a second time (a loopback URL there is still wrapped).
- A `](` left outside code after `_LINK` is the start of a link destination whose `[` is not on this part of the
  line (a code span in the text, or text over two lines): everything up to the matching `)` (or the end of the part)
  is removed and counted.
- `<scheme:…>` (CommonMark scheme: a letter, then 1–31 of letters, digits, `+`, `.`, `-`) and `<user@host>`
  autolinks are removed and counted, after the http(s) pass so a loopback autolink is still wrapped.
- Still out of reach of a regex (PRD §12, the frontend of Phase 5): escaped backticks, an indented "fence", and a bare
  e-mail address (GFM turns it into a `mailto:` link).

**Acceptance criteria**
- [x] `test_citations.py` starts with failing tests for: definitions (with `//`, `mailto:`, `<…>` and a title, the
      destination on the next line, inside `>` and a list item, not touched in fenced code, prose not touched), the
      marker that would become a link (`map_citations`), every leftover above, the double count, and the autolinks.
- [x] `grounded ask --golden … --fake`: 30/30, 0 URLs removed.
- [x] Docs updated as listed.

**Verify:** standard checks and the fake golden run above.

**Eval impact:** none measured. The changes touch only answers that contain links, URLs or autolinks: 0 in the fake
and the real golden runs of R.14 (n=30 each).

**Docs to update:** Tech §9.6, §11, §12 (output rendering); PRD §12; the `citations.py` and `params.py` docstrings;
the phase README and the ticket index.
