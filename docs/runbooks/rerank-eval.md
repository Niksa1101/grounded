# Runbook: the `hybrid_rerank` evals (ticket 6.04)

Ticket 6.04 adds `hybrid_rerank` to both evals and, in a baseline PR, a `hybrid_rerank` row to both baselines.
The code is in the repository; **the real runs need your Cohere, Gemini and Groq keys** and are run by hand,
never in CI (PRD D55). An agent wrote this file and made no real call. Whoever runs the steps (you, or the
agent of 6.04b once the keys are in `.env`) records the numbers under [Results](#results).

What you end up with: a results file that compares `hybrid` and `hybrid_rerank` question by question, the two
`hybrid_rerank` rows (ungated, no thresholds), and the rerank latency and call counts for the lift analysis
(6.05).

**How the steps are marked.** Every command is for **PowerShell**, in the folder the step names (the
repository root, or `backend/`), unless the step says Git Bash. `COPY` means type it exactly as shown and
change nothing. `YOU TYPE` means something in that step comes from you (a key); `YOU PASTE` means you paste
program output into a file. The step says what and where. No command here has a placeholder in it.

## Facts that decide the design

- **Cohere Trial key** (Tech §8, verified 2026-10-12): 1,000 API calls a month, **10 Rerank requests a
  minute**, not for production. **The Author's budget for the whole of Phase 6 is 150 real calls**; 6.01, 6.02,
  6.03 and the code of 6.04 used 0. The runs below need about 30 (see [the tally](#the-call-tally)).
- **A rerank call is cached** by (model, question, the 40 candidates in order) in `.cache/rerank.sqlite`. A
  re-run of the same question on the same index costs no call. This file is the only thing that makes the second
  run free: do not delete `.cache/`.
- **Live calls are paced** at least `EVAL_RERANK_MIN_INTERVAL_S` (6 seconds) apart, so 25 questions take about
  2.5 minutes. A cache hit is not delayed. If Cohere still answers `429`, the run stops at once; it never
  waits out a `Retry-After`.
- **A reranker error is not a result.** The retrieval eval ends on the first one (exit 3, nothing written). The
  generation eval records it per case, and `grounded eval baseline` then refuses the row. Neither ever fills a
  gap with the plain `hybrid` order.
- **The model** is `rerank-v4.0-pro` (PRD D55). Whether the Trial key accepts it is **unverified** until step 3.
  Fallback: `rerank-v4.0-fast` (already priced).
- **`RERANK_PROVIDER` matters in both directions.** `hybrid_rerank` refuses to run unless it is `cohere`, and
  the plain `hybrid` entry of `promptfooconfig.yaml` refuses to run if it is `cohere` (its row would carry the
  wrong label). The deployed demo keeps `none`; so does CI.

## Order of steps

| # | Step | Real calls |
|---|---|---|
| 1 | Keys and settings in `.env` | 0 |
| 2 | Check the setup | 0 |
| 3 | One real call: does the Trial key accept the model? | 1 Cohere |
| 4 | Retrieval eval: `hybrid` next to `hybrid_rerank` | up to 24 Cohere |
| 5 | Read the result | 0 |
| 6 | Generation eval: a smoke run, then the full run | up to 5 Cohere, Gemini, Groq |
| 7 | The baseline PR | 0 |

## 1. Keys and settings in `.env`

The file is `.env` in the repository root (gitignored; `.env.example` lists every variable). Open it:

```powershell
notepad .env
```

COPY these lines into it if they are not there yet, **and YOU TYPE each key after its `=`** (from your password
manager; never paste a key into a chat, a PR or a file other than `.env`). Leave the other lines as they are:

```text
COHERE_API_KEY=
GEMINI_API_KEY=
GROQ_API_KEY=
GEMINI_MODEL=gemini-3.5-flash-lite
GROQ_MODEL=openai/gpt-oss-120b
JUDGE_MODEL=openai/gpt-oss-120b
RERANK_PROVIDER=cohere
RERANK_MODEL=rerank-v4.0-pro
```

`GEMINI_API_KEY` is needed only when a question is not yet in `.cache/embeddings.sqlite` (the golden questions
normally are). `GROQ_API_KEY` and the two Groq model lines are for the judge in step 6. Do not add
`EVAL_RERANK_MIN_INTERVAL_S` unless a step below tells you to.

When you are done with Phase 6 runs, set `RERANK_PROVIDER=none` again (or remove the line): the first promptfoo
config needs it off.

## 2. Check the setup

Nothing here calls an API. COPY, one block at a time:

```powershell
docker compose -f infra/docker-compose.yml up -d db
```

```powershell
cd backend
```

```powershell
uv run grounded migrate
```

(`migrate` only applies what the local database is missing; it does not touch Neon.)

```powershell
uv run python -c "from grounded.settings import get_settings as g; s = g(); k = lambda v: bool(v and v.get_secret_value().strip()); print(s.rerank_provider, s.rerank_model, k(s.cohere_api_key), k(s.gemini_api_key), k(s.groq_api_key))"
```

It prints `cohere rerank-v4.0-pro True True True`: the provider, the model, and whether each key is set (never the
keys). Anything else means step 1 is not saved or a line is blank.

```powershell
uv run grounded index list
```

The active index must be the one the committed rows were scored on: config hash starting `4949e8a3`. If it is
another, `--write-baseline` will refuse in step 7 (correctly: rows from different setups must not share a file).

## 3. One real call

This asks one golden question through the whole pipeline with the **stub** generator (`--fake`, so no Gemini
quota), and the real reranker. It costs 1 Cohere call, and the answer is cached for step 4.

COPY (still in `backend/`):

```powershell
uv run grounded ask "How do I install FastAPI with its standard optional dependencies but without the FastAPI Cloud CLI?" --fake
```

In the JSON, `"rerank_used": true` under `meta` means Cohere accepted the key and the model. Write the call in
[the tally](#the-call-tally).

- `"rerank_used": false`: the reason is in the WARNING line above the JSON (`rerank_error`). The codes are in
  [When something fails](#when-something-fails). A `rejected` with `404` or `403` means the Trial key does not
  accept `rerank-v4.0-pro`: change the line in `.env` to `RERANK_MODEL=rerank-v4.0-fast`, run this step again,
  and say so in the PR (PRD D55). A new model is a new cache key and a new `retrieval_config_hash`; nothing is
  lost but the one call.
- A question that is not in the embedding cache needs `GEMINI_API_KEY`.

## 4. Retrieval eval: `hybrid` next to `hybrid_rerank`

Up to 24 live Cohere calls (25 answerable questions, one already cached by step 3), about 2.5 minutes. It does
**not** write a baseline: `--write-baseline` here would rewrite the committed `hybrid` row.

COPY (in `backend/`):

```powershell
uv run grounded eval retrieval --config hybrid --config hybrid_rerank --out ../eval/results/rerank-compare.json
```

It prints the metric table (`hybrid` and `hybrid_rerank`, `n` 25, `k` 40), then
`Rerank rerank-v4.0-pro: N live calls, M from the rerank cache; live latency p50 .. / p95 .. ms`. **Write N in
the tally.** If it stops with `Eval stopped: rerank failed at q… (code, …)` it exited with 3 and wrote nothing;
read [When something fails](#when-something-fails), then run the same command again: the questions already
reranked are in the cache and cost nothing.

**Keep this file.** It is the only one with the per-question comparison and the live latency
(`rerank.latency_p50_ms`, `latency_p95_ms`), which the lift analysis (6.05) needs, and a re-run overwrites it
with a cache-only run that has `live_calls` 0 and no latency. COPY, right away (the folder is gitignored, so
back it up elsewhere too):

```powershell
Copy-Item ../eval/results/rerank-compare.json ../eval/results/rerank-compare-first.json
```

## 5. Read the result

The two rows differ only in the order of the same 40 candidates. MRR is taken over the whole reranked list;
Recall@5/10 and nDCG@5/10 over its first 5 and 10. The results file has, per question, the ranks of each label
in `hybrid` and in `hybrid_rerank`. One question is 0.04 of a mean (`n` = 25): do not call a smaller difference
a lift (AGENTS §7). "No lift" is a valid result. Nothing is decided here; 6.05 writes the analysis.

## 6. Generation eval

`hybrid_rerank` is in its own config file, `eval/promptfoo/promptfooconfig.rerank.yaml`; the file CI runs is
unchanged. It spends Gemini, Groq and (up to 5, the unanswerable questions the retrieval eval skipped) Cohere
calls, one pass of the 30 golden questions. **One real run is authorized for Phase 6**, so do the smoke run first.

### 6.1 The promptfoo setup

Once per PowerShell window (COPY). The first command goes to the repository root from wherever you are in it:

```powershell
Set-Location (git rev-parse --show-toplevel)
```

```powershell
$env:PROMPTFOO_PYTHON = (uv run --project backend python -c "import sys; print(sys.executable)")
$env:PROMPTFOO_DISABLE_TELEMETRY = "1"; $env:PROMPTFOO_DISABLE_UPDATE = "1"
```

In Git Bash the same is `export PROMPTFOO_PYTHON="$(uv run --project backend python -c 'import sys; print(sys.executable)')"`
and `PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1` in front of the `npx` line (Tech §15.3).

### 6.2 Smoke run: three questions

COPY. `q003` and `q036` are answerable (cached by step 4, so no Cohere call), `q045` is not (1 Cohere call).

```powershell
$env:EVAL_QUESTION_IDS = "q003,q045,q036"
```

```powershell
npx promptfoo@0.123.1 eval -c eval/promptfoo/promptfooconfig.rerank.yaml -j 1 --no-cache -o eval/results/rerank-smoke.json
```

```powershell
Remove-Item Env:EVAL_QUESTION_IDS
```

Check that no case degraded. No output from this command is good:

```powershell
Select-String -Path eval/results/rerank-smoke.json -Pattern '"rerank_error": "', '"rerank_used": false'
```

If it prints lines, a case stayed on the RRF order: read [When something fails](#when-something-fails) before the
full run. Whatever the smoke run costs goes in the tally.

### 6.3 Full run

COPY:

```powershell
npx promptfoo@0.123.1 eval -c eval/promptfoo/promptfooconfig.rerank.yaml -j 1 --no-cache -o eval/results/rerank-generation.json
```

```powershell
Select-String -Path eval/results/rerank-generation.json -Pattern '"rerank_error": "', '"rerank_used": false'
```

There is no gate to run on this file: the generation gate fails closed when the gated `hybrid` config is not in
the run, and this run has only `hybrid_rerank`. The check is `grounded eval baseline` in step 7: it refuses an
`inconclusive` run (more than 20% of cases errored for provider reasons), a case that errored for a provider
reason, a judge that could not run and a degraded rerank case, names them, and writes nothing. If the run is
one of those, wait for the quota, run the same promptfoo command again (the eval caches replay what was paid
for) and do not write it.

## 7. The baseline PR

Conditions the Author set (2026-10-12), all checked by the tooling or by `git diff`:

- the PR is titled `eval: update baseline (add hybrid_rerank rows)`, with a before and after table and `n`;
- it only **adds** `hybrid_rerank` rows; every existing row and every threshold stays byte-identical;
- the numbers come from the tools below, never typed;
- an `inconclusive` run or a run with a degraded rerank case is never written (the tools refuse);
- the new rows have no thresholds.

Branch from a clean `main` (`git status` must show nothing changed: the rows record `git_dirty`). COPY:

```powershell
Set-Location (git rev-parse --show-toplevel)
```

```powershell
git switch main
```

```powershell
git pull --ff-only
```

```powershell
git switch -c phase-6/baseline-rerank
```

Both writes, in `backend/`. COPY:

```powershell
cd backend
```

The first re-reads the rerank cache, so it makes 0 calls and writes only the new row. The second refuses, and
leaves the file alone, if the run is not fully scored or any case degraded:

```powershell
uv run grounded eval retrieval --config hybrid_rerank --write-baseline
```

```powershell
uv run grounded eval baseline --suite generation --results ../eval/results/rerank-generation.json
```

Then re-paste the README block from the new baselines (a test fails until you do):

```powershell
uv run grounded eval report
```

YOU PASTE that output between `<!-- eval-report:start -->` and `<!-- eval-report:end -->` in `README.md`
(replacing what is there), then prove the rule. COPY:

```powershell
git diff --stat
```

```powershell
git diff eval/baselines
```

The diff of each baseline file must show **only added lines** (the new row), no changed or removed ones.
Finally the standard checks (`uv run ruff check . && uv run ruff format --check . && uv run pyright`, then
`uv run pytest`), the PR with the `run-eval` label, and your approval of the rows.

## When something fails

| Symptom | What it means | What to do |
|---|---|---|
| `hybrid_rerank needs RERANK_PROVIDER=cohere` | `.env` has `none` or no line | Step 1 |
| `COHERE_API_KEY must be set` or `RERANK_MODEL must be set` | The variable is blank or missing | Step 1; the eval fails before it embeds anything |
| `provider mode hybrid would be answered with the reranker` | You ran the first promptfoo config (`promptfooconfig.yaml`) with `RERANK_PROVIDER=cohere` | Set it to `none` for that config, or use `promptfooconfig.rerank.yaml` |
| Retrieval eval: `rerank failed at q… (rate_limited …)`, exit 3 | Cohere answered 429: the Trial minute or month limit | Wait two minutes and run the same command again (the done questions are cached). If it repeats, add `EVAL_RERANK_MIN_INTERVAL_S=8` to `.env`; check the call count on the Cohere dashboard against the 1,000 a month |
| `… (timeout …)` | The call took more than `RERANK_TIMEOUT_S` (3 s) | Run again once. If it keeps happening, tell the Author: raising the timeout, or `rerank-v4.0-fast`, is a D55 decision, not a retry |
| `… (unavailable …)` | Cohere answered 5xx or the network failed | Wait and run again |
| `… (rejected …)` | 401: the key. 403 or 404: the Trial key does not accept the model | Fix the key, or switch to `RERANK_MODEL=rerank-v4.0-fast` and say so in the PR |
| `… (bad_output …)` | Cohere's answer does not have the documented shape | Stop and report it. Compare with `backend/tests/fixtures/cohere/rerank_success.json` (a fixture built from the docs, not recorded); the fixture is replaced by a real recording once one exists |
| promptfoo: a case with `"rerank_error": "…"` | The pipeline degraded that case to the RRF order (same codes as above, and `cap_reached`/`unavailable`) | The run is not a result. Stop it (Ctrl-C) at the first one, fix the cause, run again; cached questions cost nothing |
| `grounded eval baseline` says `the reranker degraded in N of 30 case(s)` | The same, found afterwards | Same: run again, then write |
| `grounded eval baseline` says `answered with a reranker` for `hybrid` | `RERANK_PROVIDER=cohere` was in `.env` for a plain `hybrid` run | Set it to `none` and run that config again |
| `--write-baseline` says `row … differs in … index_config_hash` | The active local index is not the one the committed rows were scored on | Do not force it. Activate the right index version (`grounded index list`), or ask |
| `grounded eval baseline` says `inconclusive` or `errored for a provider reason` | Gemini or Groq quota, a 5xx or a timeout hit more than a few cases | Wait for the reset or the outage; run the same promptfoo command again (the eval caches keep what was paid for); never loop |

## The call tally

The Phase 6 budget is **150 real Cohere calls** for all tickets. Each run prints its count (`N live calls`);
add it below the same day. Expected total for a clean pass: 1 (step 3) + 24 (step 4) + 1 (smoke, `q045`) + 4
(full run, the other unanswerable questions) = **30**, which leaves room for about five repeats.

## Results

Filled by whoever runs the steps. Numbers are copied from the tool output, never estimated.

| Date | Step | Live Cohere calls | Total so far | Note |
|---|---|---|---|---|
| 2026-10-12 | 6.01 to 6.03 | 0 | 0 | no key in `.env`; everything tested with fakes |
| _TBD_ | | | | |

_TBD_ (6.04b): the trial key's acceptance of `rerank-v4.0-pro`; the retrieval comparison (`n` = 25); live latency
p50 and p95; the generation run; the two baseline rows and their PR.
