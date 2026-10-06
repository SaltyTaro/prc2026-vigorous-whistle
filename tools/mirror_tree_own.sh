#!/usr/bin/env bash
# Mirror finished tree own-points fits (<leaf>/<variant>/<fold>/{run.json, predictions.parquet, model.txt}) and logs from one worker to
# runs/tree_own/fits/, checking every file's SHA-256 against the worker's copy. Usage: tools/mirror_tree_own.sh N
set -euo pipefail
cd "$(dirname "$0")/.."
n=$1
h=arcw$n
R=/home/azureuser/prc-flight-2026/runs/tree_own/fits
L=runs/tree_own/fits
for d in $(timeout 60 ssh -o BatchMode=yes $h "cd $R && for x in */*/*/run.json; do [ -e \"\$x\" ] && echo \${x%/run.json}; done"); do
  mkdir -p $L/$d
  for f in run.json predictions.parquet model.txt; do
    scp -q $h:$R/$d/$f $L/$d/$f
    remote=$(timeout 60 ssh -o BatchMode=yes $h "sha256sum $R/$d/$f" | cut -d' ' -f1)
    local_=$(sha256sum $L/$d/$f | cut -d' ' -f1)
    [ "$remote" = "$local_" ] || { echo "HASH MISMATCH $h $d/$f"; exit 1; }
  done
  echo "$h $d ok"
done
mkdir -p $L/logs/$h
scp -q "$h:$R/logs/*" $L/logs/$h/ || true
