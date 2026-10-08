# Evaluation

The eval harness that gates quality-affecting PRs (see [docs/Tech.md](../docs/Tech.md) and AGENTS.md §7).

| Path | Contents | Arrives in |
|---|---|---|
| `golden/` | versioned golden set (`golden_set.vN.jsonl`) and the labeling guide | Phase 1 |
| `baselines/` | committed baseline metrics; changed only by an `eval: update baseline (<reason>)` PR | Phase 1 (retrieval), Phase 4 (generation) |
| `promptfoo/` | generation eval: `promptfooconfig.yaml` and three thin shims (`provider.py`, `asserts.py`, `tests_loader.py`) over `backend/src/grounded/evals/promptfoo_*.py`, which is where the code is linted, type-checked and tested | Phase 4 (4.05) |
| `results/` | run outputs, **gitignored** | — |

Metric numbers in the README, PRs and baselines come only from committed eval output. Never edit them by hand.

## Running the generation eval (promptfoo)

It runs the real pipeline in-process for the `no_rag` and `hybrid` configs over the golden set and scores the
deterministic metrics (schema first-try validity, citation validity, refusal correctness, citation precision). From
the repository root, after `uv sync` in `backend/`, in Git Bash (Linux and macOS are the same):

```bash
export PROMPTFOO_PYTHON="$(uv run --project backend python -c 'import sys; print(sys.executable)')"
PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1 \
  npx promptfoo@0.123.1 eval -c eval/promptfoo/promptfooconfig.yaml -j 1 --no-cache -o eval/results/<name>.json
```

PowerShell, the CI form, the flags, what each variable is for and the result format: [docs/Tech.md §15.3](../docs/Tech.md).

- **It uses quota.** The real run makes up to 2 generator calls per golden question (one per config, plus a retry on
  invalid output). Check the provider's remaining daily quota first, and never loop on it. After a daily quota or a
  rejected request the provider stops calling and answers the rest of the run with the same tagged error.
- **A few questions:** `EVAL_QUESTION_IDS=q003,q045,q036` in front of the command runs only those (an unknown id is an
  error).
- **No network:** `GENERATOR_PROVIDERS=fake GEMINI_API_KEY=` runs the whole harness with the canned stub provider. A
  question whose embedding is not in `.cache/embeddings.sqlite` then fails as a tagged `EmbedderUnavailableError` instead
  of calling the API. The stub's answers are not an eval result.
- **The result file** (`-o <name>.json`; `.html` for the viewer) is promptfoo's own. How to read a row, the error tags and
  the not-applicable marker are in Tech §15.3, and a recorded sample is in `backend/tests/fixtures/promptfoo/`.
- The golden set is never edited by this run; `tests_loader.py` only reads it.

`grounded eval retrieval --write-baseline` refuses to put rows from different setups (golden-set version or bytes, index config) into one baseline file. To add or refresh a config after such a change, run every config together, e.g. `--config dense --config fts --write-baseline`.
