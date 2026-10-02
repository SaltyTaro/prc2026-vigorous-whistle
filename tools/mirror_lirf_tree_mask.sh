#!/usr/bin/env bash
# Mirror finished masked LIRF tree design runs (seasonal/{run.json, predictions.parquet, model.*} + logs) from one worker to
# runs/lirf_tree_mask/design/, checking every file's SHA-256 against the worker's copy. Usage: tools/mirror_lirf_tree_mask.sh N
set -euo pipefail
cd "$(dirname "$0")/.."
n=$1
R=/home/azureuser/prc-flight-2026/runs/lirf_tree_mask/design
for d in $(timeout 30 ssh -o BatchMode=yes arcw$n "cd $R && for x in */seasonal/run.json; do echo \${x%%/*}; done"); do
  mkdir -p runs/lirf_tree_mask/design/$d/seasonal
  for f in $(timeout 30 ssh -o BatchMode=yes arcw$n "cd $R/$d/seasonal && ls"); do
    scp -q arcw$n:$R/$d/seasonal/$f runs/lirf_tree_mask/design/$d/seasonal/$f
    remote=$(timeout 60 ssh -o BatchMode=yes arcw$n "sha256sum $R/$d/seasonal/$f" | cut -d' ' -f1)
    local_=$(sha256sum runs/lirf_tree_mask/design/$d/seasonal/$f | cut -d' ' -f1)
    [ "$remote" = "$local_" ] || { echo "HASH MISMATCH arcw$n $d/$f"; exit 1; }
  done
  echo "arcw$n $d ok"
done
mkdir -p runs/lirf_tree_mask/design/logs
scp -q "arcw$n:$R/logs/*" runs/lirf_tree_mask/design/logs/ || true
