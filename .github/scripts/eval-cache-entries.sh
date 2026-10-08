#!/usr/bin/env bash
# Print how many entries an eval cache file holds (0 when the file does not exist).
#
#   eval-cache-entries.sh <sqlite-file>
#
# It first folds the write-ahead log into the main file (`wal_checkpoint(TRUNCATE)`). The caches
# run in WAL mode (infra/kvcache.py), and actions/cache saves the one file: a process that was
# killed (a cancelled run, a timeout) can leave committed rows in the `-wal` file, which would be
# missing from the saved copy. eval.yml counts before and after the eval and saves the cache only
# when the count changed, so a run that only replayed the cache does not add an identical copy.
# Exit 1 with a message when the file is not a readable cache.
set -euo pipefail

file=${1:?usage: eval-cache-entries.sh <sqlite-file>}
python3 -I - "$file" <<'PY'
import sqlite3
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
try:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        (count,) = conn.execute("SELECT count(*) FROM kv").fetchone()
    finally:
        conn.close()
except sqlite3.Error as exc:
    print(f"{path} is not a readable cache: {exc}", file=sys.stderr)
    raise SystemExit(1)
print(count)
PY
