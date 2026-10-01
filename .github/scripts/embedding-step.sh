#!/usr/bin/env bash
# Run one step that needs embeddings and say in the job summary why it failed, if it did.
#
#   embedding-step.sh "<title>" <command> [args...]
#
# `grounded ingest` and `grounded eval retrieval` exit 3 when the vectors could not be had: a cold
# cache with no GEMINI_API_KEY, or the daily embedding quota (EXIT_EMBEDDINGS_UNAVAILABLE in cli.py,
# Tech.md §17). That is the provider's side, not a result about quality, so the summary names it.
# Any other non-zero exit is reported as a plain failure of the step. The step fails either way, so
# the gate step after it never runs and nothing is skipped or green by accident (AGENTS.md §7).
#
# The exit code is also written to the step output `exit_code`, for the cache-save conditions.
set -uo pipefail

title=$1
shift

errors=$(mktemp)
code=0
"$@" 2>"$errors" || code=$?
cat "$errors" >&2

echo "exit_code=$code" >> "$GITHUB_OUTPUT"

if [ "$code" -ne 0 ]; then
  {
    if [ "$code" -eq 3 ]; then
      echo "### ⚠️ Retrieval eval: infrastructure failure, not a quality result"
      echo
      echo "**$title** could not get embeddings: no \`GEMINI_API_KEY\` on a cold cache, or the daily"
      echo "quota is spent. There is no gate verdict. Run the \`warm-cache\` workflow, or re-run this"
      echo "job after the quota resets (midnight Pacific); cached texts are not sent again."
    else
      echo "### ❌ Retrieval eval: \`$title\` failed (exit $code)"
      echo
      echo "The eval did not run, so there is no gate verdict."
    fi
    echo
    echo '```text'
    cat "$errors"
    echo '```'
  } >> "$GITHUB_STEP_SUMMARY"
  if [ "$code" -eq 3 ]; then
    echo "::error title=Embeddings unavailable::$title: infrastructure failure, not a quality result."
  fi
fi

exit "$code"
