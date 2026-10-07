# R.12 — Settings guard for confidence invariant 4

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md). Work only on this ticket.

> **Owner:** G (Author reviews) · **Type:** Build · **Branch:** `phase-3/confidence-weight-guard` · **Blocked by:** R.11 ·
> **Builds on:** `settings.py`, `generation/confidence.py` (Author-owned, docstring only)

**What to build:** review item #4. Invariant 4 ("self-report cannot carry a claim": at most 0.6 with the weakest
support) is guarded by `confidence_w_self <= 0.6`. The heuristic divides by the sum of the weights, so what counts is
the *share* of `w_self`, and the field cap guarantees nothing. Example: `W_SELF=0.6` and `W_RETRIEVAL`, `W_AGREEMENT`,
`W_CITATIONS` at 0.1 give (0.6 + 0.1·2/3 + 0.1·1/2) / 0.9 ≈ 0.80. Decision of 2026-10-07: a `Settings` validator checks
the worst-case bound; the Agent writes it, the Author reviews.

**Read first:** Tech §9.8; `generation/confidence.py` module docstring (invariant 4) and the `score_claims` docstring
(the proof of invariant 4).

**Scope notes**
- In `Settings._check_consistency`: the weights must not all be zero, and
  `(w_self·1 + w_retrieval·2/3 + w_citations·1/2) / Σw <= SELF_CARRY_CEILING` with `SELF_CARRY_CEILING = 0.6`, a module
  constant pointing to Tech §9.8. A comment says where 2/3 and 1/2 come from (the weakest support: one list, so
  `retrieval <= 2/3` and `agreement = 0`; one citation, so `citations = 1/2`; rerank off). The error message names the
  computed bound and the weights.
- The `le=0.6` field cap on `confidence_w_self` stays as the first fence.
- `generation/confidence.py`: only the docstring sentence "``Settings`` caps ``w_self`` at ``0.6`` as a guard" changes,
  to describe the new check. No change to the heuristic. The Author reviews the wording.

**Acceptance criteria**
- [ ] `test_settings.py`: the review's example is rejected; all-zero weights are rejected; the defaults pass; a
      non-default config at the bound passes.
- [ ] `test_confidence.py`: the invariant 4 test is parametrized over the defaults and one non-default config the
      validator accepts, so the bound is shown to be enough, not only the defaults.

**Verify:** standard checks.

**Eval impact:** none (the defaults do not change).

**Docs to update:** Tech §9.8, `.env.example` (the comment on the `CONFIDENCE_*` block).
