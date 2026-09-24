#!/usr/bin/env bash
# adsb_nn2: run folds of one network seed sequentially on a worker with tools/run_adsb_nn2.py, stopping at the first failure.
# usage: tools/nn2_chain.sh CONFIG VARIANT SEED OUTPUT_ROOT FOLD [FOLD ...]   (learner directory OUTPUT_ROOT/<variant>_s<seed>)
# Logs: runs/adsb_nn2/logs/<variant>_s<seed>_<fold>.log; progress markers: runs/adsb_nn2/logs/chain_<variant>_s<seed>.log
set -u
cd /home/azureuser/prc-flight-2026
config=$1; variant=$2; seed=$3; root=$4; shift 4
mkdir -p runs/adsb_nn2/logs
marker=runs/adsb_nn2/logs/chain_${variant}_s${seed}.log
echo "chain start $(date -u +%FT%TZ) folds: $*" >> "$marker"
for fold in "$@"; do
  log=runs/adsb_nn2/logs/${variant}_s${seed}_${fold}.log
  if .venv/bin/python -X utf8 tools/run_adsb_nn2.py --config "$config" --variant "$variant" --seed "$seed" --fold "$fold" --output "${root}/${variant}_s${seed}" --threads 8 > "$log" 2>&1; then
    echo "done $fold $(date -u +%FT%TZ)" >> "$marker"
  else
    echo "FAILED $fold $(date -u +%FT%TZ)" >> "$marker"
    exit 1
  fi
done
echo "chain done $(date -u +%FT%TZ)" >> "$marker"
