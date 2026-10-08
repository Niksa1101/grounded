#!/usr/bin/env bash
# Print the Settings values the CI caches are keyed on, as `name=value` lines for $GITHUB_OUTPUT:
#
#   bash .github/scripts/cache-keys.sh >> "$GITHUB_OUTPUT"
#
# Settings is the single source of the corpus ref, the embedding model and the tokenizer encoding
# (Tech.md §17), so the cache keys and the ingest follow it and can't drift apart. Needs the backend
# environment (`uv sync`) and does not read the database or any key.
set -euo pipefail

cd "$(dirname "$0")/../../backend"
uv run python - <<'PY'
from grounded.settings import get_settings

s = get_settings()
print(f"fastapi_ref={s.fastapi_ref}")
print(f"embedding_model={s.embedding_model}")
print(f"embedding_dim={s.embedding_dim}")
print(f"tokenizer_encoding={s.tokenizer_encoding}")
PY
