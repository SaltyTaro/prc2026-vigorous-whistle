#!/usr/bin/env bash
# adsb_rctx: run folds of one residual context-network seed sequentially on a worker with tools/run_adsb_rctx.py, stopping at the first
# failure. usage: tools/rctx_chain.sh CONFIG VARIANT SEED OUTPUT_ROOT MODE FOLD [FOLD ...]   (MODE design|lomo; FOLD a 2025 month or
# refit12; learner directory OUTPUT_ROOT/<variant>_s<seed>). Logs: runs/adsb_rctx/logs/<variant>_s<seed>_<fold>.log; markers:
# runs/adsb_rctx/logs/chain_<variant>_s<seed>.log
set -u
cd /home/azureuser/prc-flight-2026
config=$1; variant=$2; seed=$3; root=$4; mode=$5; shift 5
mkdir -p runs/adsb_rctx/logs
marker=runs/adsb_rctx/logs/chain_${variant}_s${seed}.log
echo "chain start $(date -u +%FT%TZ) mode $mode folds: $*" >> "$marker"
for fold in "$@"; do
  log=runs/adsb_rctx/logs/${variant}_s${seed}_${fold}.log
  if [ "$fold" = "refit12" ]; then
    what=(--refit12)
  elif [ "$mode" = "design" ]; then
    what=(--held-out "$fold" --design)
  else
    what=(--held-out "$fold")
  fi
  if .venv/bin/python -X utf8 tools/run_adsb_rctx.py --config "$config" --variant "$variant" --seed "$seed" "${what[@]}" --output "${root}/${variant}_s${seed}" --threads 8 > "$log" 2>&1; then
    echo "done $fold $(date -u +%FT%TZ)" >> "$marker"
  else
    echo "FAILED $fold $(date -u +%FT%TZ)" >> "$marker"
    exit 1
  fi
done
echo "chain done $(date -u +%FT%TZ)" >> "$marker"
