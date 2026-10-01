#!/usr/bin/env bash
# Mirror finished masked-tree design runs (run.json + stop_predictions.parquet; the feature caches stay on the workers) from arcw0-arcw7
# to runs/tree_mask/design/, checking every file's SHA-256 against the worker's copy. Usage: tools/mirror_tree_mask.sh [pattern, e.g. lgb_]
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026/runs/tree_mask/design
PAT=${1:-}
for n in 0 1 2 3 4 5 6 7; do
  for d in $(timeout 30 ssh -o BatchMode=yes arcw$n "cd $R 2>/dev/null && ls -d ${PAT}*/ 2>/dev/null | tr -d /" || true); do
    if ! timeout 30 ssh -o BatchMode=yes arcw$n "test -f $R/$d/run.json"; then continue; fi
    mkdir -p runs/tree_mask/design/$d
    for f in run.json stop_predictions.parquet; do
      scp -q arcw$n:$R/$d/$f runs/tree_mask/design/$d/$f
      remote=$(timeout 30 ssh -o BatchMode=yes arcw$n "sha256sum $R/$d/$f" | cut -d' ' -f1)
      local_=$(sha256sum runs/tree_mask/design/$d/$f | cut -d' ' -f1)
      [ "$remote" = "$local_" ] || { echo "HASH MISMATCH arcw$n $d/$f"; exit 1; }
    done
    echo "arcw$n $d ok"
  done
done
