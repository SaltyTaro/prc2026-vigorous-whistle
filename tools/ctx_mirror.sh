#!/usr/bin/env bash
# adsb_ctx: copy finished runs of one output root back from the workers (tar over ssh), with the chain logs, then print local sha256 of
# every run.json and prediction file. usage: tools/ctx_mirror.sh ROOT (e.g. runs/adsb_ctx/design) [HOST ...] (default arcw0-arcw7)
set -euo pipefail
cd "$(dirname "$0")/.."
root="$1"; shift
hosts=${*:-"arcw0 arcw1 arcw2 arcw3 arcw4 arcw5 arcw6 arcw7"}
mkdir -p "$root" runs/adsb_ctx/logs
for host in $hosts; do
  timeout 600 ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar cf - $root runs/adsb_ctx/logs $(ls runs/december/learners/runs__adsb_ctx__* 2>/dev/null) 2>/dev/null" | tar xf - || echo "$host: mirror returned $?"
done
find "$root" -name run.json -o -name '*predictions.parquet' | sort | xargs sha256sum
