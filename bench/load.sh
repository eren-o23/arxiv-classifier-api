#!/usr/bin/env bash
# Concurrency sweep with hey. Throughput and p95 at each level.
#
#   make load                          # spawn a server, sweep 1 2 4 8 16
#   NUM_THREADS=1 make load            # the thread comparison, same binary
#   URL=https://host bench/load.sh 8   # a box someone else runs (M6)
#
# Spawns its own server unless URL is set, same as bench/latency.py.
set -euo pipefail

N=${N:-200}
PORT=${PORT:-8001}
LEVELS=${*:-1 2 4 8 16}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

command -v hey >/dev/null || { echo "hey not installed — brew install hey" >&2; exit 1; }

# hey posts a fixed body; take the same paper bench/latency.py times.
body=$(mktemp)
trap 'rm -f "$body"' EXIT
python3 -c 'import json,sys
p = json.load(open(sys.argv[1]))["papers"][0]
json.dump({"title": p["title"], "abstract": p["abstract"]}, open(sys.argv[2], "w"))' \
  tests/golden.json "$body"

if [ -n "${URL:-}" ]; then
  url=${URL%/}
else
  url="http://127.0.0.1:$PORT"
  # >/dev/null: the app logs one JSON line per request to stdout, and 200 of
  # them per level would bury the table this script prints.
  uv run uvicorn serving.api:app --port "$PORT" --log-level warning >/dev/null 2>&1 & pid=$!
  # $pid, not $!, which is stale by the time the trap fires.
  trap 'kill $pid 2>/dev/null || true; rm -f "$body"' EXIT
  # uvicorn refuses connections until the model is loaded and warmed, so this
  # loop is the readiness gate — there is no racing a half-warm server.
  until curl -sf "$url/health" >/dev/null 2>&1; do sleep 0.5; done
fi

echo
echo "## hey sweep — NUM_THREADS=${NUM_THREADS:-default (all cores)}, $N requests per level"
echo
echo "| concurrency | throughput | p95 |"
echo "|---|---|---|"
for c in $LEVELS; do
  out=$(hey -m POST -T application/json -D "$body" -c "$c" -n "$N" "$url/predict")
  rps=$(echo "$out" | awk '/Requests\/sec/{printf "%.1f", $2}')
  # ' 95%' not '95% in': hey prints the literal "95%% in 0.2043 secs".
  p95=$(echo "$out" | awk '/ 95%/{printf "%.0f", $3 * 1000}')
  echo "| $c | $rps req/s | $p95 ms |"
done
