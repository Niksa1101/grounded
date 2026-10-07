# R.14 — URLs in the answer, and the question escaped in the prompt

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-3/answer-text-safety` · **Blocked by:** R.13 ·
> **Builds on:** `generation/citations.py`, `generation/context.py`, `generation/pipeline.py`,
> `observability/request_log.py`, `evals/ask_batch.py`

**What to build:** review items #7 and #8.

- **#7:** "the LLM never produces URLs" (AGENTS.md §6.3) is enforced only by the prompt. A model that writes
  `[c1](https://x)` gets `[1](https://x)` after marker rewriting: a clickable link in the UI (Tech §12 renders links).
  Decision of 2026-10-07: remove links and URLs and count them in the stdout summary line (no migration).
- **#8:** `escape_content` protects chunk text, but the question goes into the prompt raw. A question that contains
  `<source id="c1">…</source>` can pose as a source; the model may cite `c1`, and the server maps it to a real
  documentation URL.

**Read first:** Tech §9.2, §9.3, §9.6, §12, §14.

**Scope notes**
- `citations.py`: the "every line outside fenced code" walk of `rewrite_markers` becomes `_outside_fences(markdown,
  transform)`, used by both. New `strip_urls(markdown) -> tuple[str, int]`, outside fenced code **and** outside inline
  code spans:
  - `[text](url)` and `![alt](url)` become `text` / `alt`; when the text is a marker (`c\d+`) it stays `[cN]`, so the
    marker is rewritten as usual afterwards;
  - `<http…>` autolinks and bare `http(s)://…` URLs are removed, except loopback hosts (`localhost`, `127.0.0.1`,
    `0.0.0.0`), which are wrapped in inline code: the FastAPI docs use them ("open http://127.0.0.1:8000/docs") and
    in code they cannot become links;
  - every removal is counted (a wrapped loopback URL is not a removal).
- `map_citations` runs `strip_urls` **before** the marker rewrite. `MappedAnswer.removed_url_count` →
  `RequestTrace.removed_url_count` → the stdout "request completed" line (like `dropped_claim_count`) and
  `QuestionResult` / `BatchSummary` of the batch tool. It is not bad output: no retry.
- `pipeline.py`: in hybrid mode the question is rendered as `escape_content(question)` (reuse from `context.py`). The
  cache key and the log keep the original text.

**Acceptance criteria**
- [x] `test_citations.py`: link → text; `[c1](url)` → `[1]`; image → alt; bare URL removed; loopback URL wrapped in
      code outside code and untouched inside it; URLs inside fenced code and inline code untouched; counts.
- [x] A test shows a question with `<source id="c1">` reaches the prompt as `&lt;source` (fake provider's recorded
      call).
- [x] Integration: `removed_url_count` appears in the "request completed" log line (`caplog`), no question text.
- [x] `grounded ask --golden … --fake`: 30/30, 0 URLs removed.

**Verify:** standard checks, the fake golden run above, then the Author's real-provider golden run (phase README).

**Eval impact:** yes (answer text) → `run-eval` label. Report the real golden run with `n=30` against the 2026-10-07
run, as information, not a baseline.

**Docs to update:** Tech §9.3, §9.6, §12, §14 (stdout line), §15.8.
