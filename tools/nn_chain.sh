#!/usr/bin/env bash
# Run the folds of one network seed sequentially on a worker (tools/run_adsb_nn.py), stopping at the first failure.
# usage: tools/nn_chain.sh CONFIG VARIANT SEED FOLD [FOLD ...]
# Logs: runs/adsb_nn/logs/<variant>_s<seed>_<fold>.log; progress markers: runs/adsb_nn/logs/chain_<variant>_s<seed>.log
set -u
cd /home/azureuser/prc-flight-2026
config=$1; variant=$2; seed=$3; shift 3
mkdir -p runs/adsb_nn/logs
marker=runs/adsb_nn/logs/chain_${variant}_s${seed}.log
echo "chain start $(date -u +%FT%TZ) folds: $*" >> "$marker"
for fold in "$@"; do
  log=runs/adsb_nn/logs/${variant}_s${seed}_${fold}.log
  if .venv/bin/python -X utf8 tools/run_adsb_nn.py --config "$config" --variant "$variant" --seed "$seed" --fold "$fold" --output "runs/adsb_nn/${variant}_s${seed}" --threads 8 > "$log" 2>&1; then
    echo "done $fold $(date -u +%FT%TZ)" >> "$marker"
  else
    echo "FAILED $fold $(date -u +%FT%TZ)" >> "$marker"
    exit 1
  fi
done
echo "chain done $(date -u +%FT%TZ)" >> "$marker"
