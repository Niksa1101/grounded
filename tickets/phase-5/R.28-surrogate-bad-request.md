# R.28 — A lone surrogate is a 422, not a 500

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#second-review-follow-ups-2026-10-12). Work only on this ticket.

> **Owner:** G · **Type:** Build (bug fix) · **Branch:** `phase-5/surrogate-bad-request` · **Blocked by:** R.27 ·
> **Builds on:** `infra/hashing.py`, `api/errors.py`, `tests/integration/test_ask_protection.py`

**What to build:** review item #2. `{"question": "...\ud800..."}` through the public `/api/ask` answers 500
`internal_error` instead of 422. `AskRequest` rejects the string (`string_unicode`), and the validation handler
(`api/errors.py:_bad_request`) then calls `question_hash(text)` on the rejected text; a lone surrogate cannot be encoded
as UTF-8, so `.encode("utf-8")` raises `UnicodeEncodeError` inside the handler.

**Read first:** Tech §11 (normalization), §12 (input limits); `infra/hashing.py`; `api/errors.py` (`_bad_request`).

**Scope notes**
- **Test first, and red.** In `tests/integration/test_ask_protection.py`, next to the parametrized rule test, send the
  body as raw bytes: `content=b'{"question": "How do I \\ud800 declare a path?"}'` (httpx cannot JSON-encode a
  surrogate, so the escape is written out by hand). Expected: 422 `bad_request`; one row with `outcome='bad_request'`,
  `question` NULL and `question_hash` not NULL; the text is not in the response.
- **Fix at the root, not in the handler.** `question_hash` encodes with `.encode("utf-8", "surrogatepass")`: for every
  well-formed string the bytes are identical, so no existing hash and no cache key changes; the function becomes total
  over every Python `str`. The docstring says why (the handler hashes text that failed validation, which is exactly the
  text that may be ill-formed). `hash_ip` is unchanged: its input is a canonical IP address.
- **No other change.** The schema, the error body and the HTTP contract stay as they are.

**Acceptance criteria**
- [ ] The integration test above, red before the fix and green after.
- [ ] `tests/unit/test_hashing.py`: a lone surrogate does not raise and hashes deterministically; the existing "sha256 of
      the normalized question" test still passes, which proves hashes of valid questions do not move.
- [ ] The in-process reproduction of the review (a script in the scratchpad) answers 422.

**Verify:** standard checks.

**Eval impact:** none: the hash of a valid question is byte-for-byte the same, so cache keys and eval cache keys hold.

**Docs to update:** Tech §12 (input limits): one sentence saying a lone surrogate is rejected as `bad_request` and its
hash is still recorded.
