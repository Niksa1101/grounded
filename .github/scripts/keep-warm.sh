#!/usr/bin/env bash
# Wakes the backend and Neon: GET <BACKEND_URL>/readyz until it answers 200 (ticket 5.11, NFR-2).
#
# /readyz queries the database, so one request starts the function and resumes Neon. A cold start
# can take many seconds, so each try is bounded (curl --max-time) and there are a few tries with a
# pause between them. Never an open loop.
#
# /readyz needs the proxy secret (PRD D54c), so the script sends X-Proxy-Secret. The header reaches
# curl on stdin (-H @-), never as an argument: an argument is visible to every process on the runner
# (ps, /proc/*/cmdline), stdin is not. The value is never printed.
#
# BACKEND_URL          the backend's production origin (repository variable). Unset: skip, exit 0.
# PROXY_SHARED_SECRET  the backend's proxy secret (repository secret, the Vercel value). Required
#                      when BACKEND_URL is set.
# KEEP_WARM_ATTEMPTS   tries (default 4)
# KEEP_WARM_TIMEOUT    seconds per try (default 45)
# KEEP_WARM_PAUSE      seconds between tries (default 10)
set -u

base="${BACKEND_URL:-}"
secret="${PROXY_SHARED_SECRET:-}"
attempts="${KEEP_WARM_ATTEMPTS:-4}"
timeout="${KEEP_WARM_TIMEOUT:-45}"
pause="${KEEP_WARM_PAUSE:-10}"

if [ -z "$base" ]; then
  echo "::notice title=Keep-warm skipped::The repository variable BACKEND_URL is not set (runbook: Housekeeping). Nothing was requested."
  exit 0
fi
case "$base" in
  https://*|http://localhost*|http://127.0.0.1*) ;;
  *)
    echo "::error title=BACKEND_URL is not an https URL::Set it to the backend's production origin, e.g. https://example.vercel.app"
    exit 1
    ;;
esac
if [ -z "$secret" ]; then
  echo "::error title=PROXY_SHARED_SECRET is not set::/readyz needs the proxy secret. Create the repository secret PROXY_SHARED_SECRET with the backend's value (Neon runbook, step 10)."
  exit 1
fi

url="${base%/}/readyz"
for ((i = 1; i <= attempts; i++)); do
  code=$(printf 'X-Proxy-Secret: %s\n' "$secret" | curl --silent --show-error --output /dev/null \
    --write-out '%{http_code}' --max-time "$timeout" --user-agent "grounded-keep-warm" \
    -H @- "$url") || code="000"
  echo "attempt $i/$attempts: HTTP $code"
  if [ "$code" = "200" ]; then
    echo "The backend answered /readyz: the function and the database are awake."
    exit 0
  fi
  if [ "$code" = "401" ]; then
    echo "::error title=Keep-warm rejected (401)::/readyz refused the X-Proxy-Secret header: the secret differs from the backend's. Set the repository secret PROXY_SHARED_SECRET to the value in Vercel."
    exit 1
  fi
  if [ "$i" -lt "$attempts" ]; then
    sleep "$pause"
  fi
done

echo "::error title=Keep-warm failed::/readyz did not answer 200 in $attempts tries (last: HTTP $code). 000 = no answer; 503 = the database or the active index is not ready."
exit 1
