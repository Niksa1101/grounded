# R.19 — Neon runbook: set the `app` password in plain text from a psql variable

> **Before you start:** read [AGENTS.md](../../AGENTS.md), the rules in [tickets/README.md](../README.md) (how to work a ticket, ticket types, [standard checks](../README.md#standard-checks)) and the phase notes in [README.md](README.md#review-follow-ups). Work only on this ticket.

> **Owner:** G · **Type:** Build · **Branch:** `phase-5/neon-role-password` · **Blocked by:** R.18 ·
> **Builds on:** `docs/runbooks/neon-production.md`, `backend/tests/unit/test_neon_runbook.py`

**What to build:** review item #1. Step 6 of the Neon runbook sets the `app` password with psql `\password`, which
sends a client-side SCRAM hash. Neon's docs say *"Passwords must be supplied in plain text … Hashed passwords are not
supported"* (`neon.com/docs/manage/roles`, read 2026-10-12). The role must still be created **with SQL**: a role
created in the Console, CLI or API is a member of `neon_superuser` (`pg_read_all_data`, `pg_write_all_data`,
`CREATEROLE`, `BYPASSRLS`), which breaks least privilege.

**Read first:** DB §5; `docs/runbooks/neon-production.md` (preflight, step 6, the leak and rotation table);
AGENTS.md §5 (secrets), §6.13.

**Scope notes**
- **Unchanged:** the role SQL block of DB §5 and the runbook (`CREATE ROLE app LOGIN;` plus the GRANTs). The
  identity test between the two stays.
- **Replace the `\password app` step with:**
  1. In the shell, read the password silently: `read -rs APP_DB_PASSWORD && export APP_DB_PASSWORD`. Paste it from
     the secret manager. It must have at least 60 bits of entropy, or Neon refuses it.
  2. In psql: `\getenv app_password APP_DB_PASSWORD`, then `ALTER ROLE app PASSWORD :'app_password';`, then
     `\unset app_password`.
  3. Back in the shell: `unset APP_DB_PASSWORD`.
- **Why this form** (write it into the runbook): Neon requires plain text (cite the docs with the date); psql history
  stores `:'app_password'`, not the value, and the shell history has no value either; the value travels over TLS
  (`sslmode=require`). Unverified: whether Neon's server log records DDL text, so say so.
- Explicitly forbid creating `app` in the Console, CLI or API (`neon_superuser`, quote the docs).
- The preflight checks `psql --version` ≥ 15 (`\getenv` is psql 15+).
- Rotation (the leak and rotation table) uses the same three commands.

**Acceptance criteria**
- [x] Runbook step 6 uses the `\getenv` / `ALTER ROLE … PASSWORD :'app_password'` / `\unset` sequence, with the why.
- [x] The preflight checks psql ≥ 15; the rotation row uses the same commands; Console/CLI/API creation is forbidden.
- [x] `test_no_password_is_written_anywhere_in_the_role_sql`: `assert "\\password app" in RUNBOOK` is replaced by
      three assertions: the runbook uses `\getenv app_password` and `PASSWORD :'app_password'`; it contains no
      `\password` command; no literal `PASSWORD '` appears (the existing regex stays).
- [x] DB §5 prose matches the runbook.

**Verify:** standard checks.

**Eval impact:** none.

**Docs to update:** `docs/runbooks/neon-production.md`, DB §5 prose.
