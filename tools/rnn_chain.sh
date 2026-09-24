#!/usr/bin/env bash
# adsb_rnn: run held-out months of one residual-network seed sequentially on a worker with tools/run_adsb_rnn.py, stopping at the first
# failure; the pseudo-month `refit12` runs tools/refit_adsb_rnn.py (twelve-month refit of the declared network NETWORK, default rnet).
# usage: tools/rnn_chain.sh CONFIG VARIANT SEED OUTPUT_ROOT MODE MONTH [MONTH ...]   (MODE design|lomo; learner directory
# OUTPUT_ROOT/<variant>_s<seed>). Logs: runs/adsb_rnn/logs/<variant>_s<seed>_<month>.log; markers: runs/adsb_rnn/logs/chain_<variant>_s<seed>.log
set -u
cd /home/azureuser/prc-flight-2026
config=$1; variant=$2; seed=$3; root=$4; mode=$5; shift 5
network=${NETWORK:-rnet}
mkdir -p runs/adsb_rnn/logs
marker=runs/adsb_rnn/logs/chain_${variant}_s${seed}.log
echo "chain start $(date -u +%FT%TZ) mode $mode months: $*" >> "$marker"
flag=""
[ "$mode" = "design" ] && flag="--design"
for month in "$@"; do
  log=runs/adsb_rnn/logs/${variant}_s${seed}_${month}.log
  if [ "$month" = "refit12" ]; then
    cmd=(.venv/bin/python -X utf8 tools/refit_adsb_rnn.py --declaration configs/adsb_rnn_v1.json --network "$network" --seed "$seed" --threads 8)
  else
    cmd=(.venv/bin/python -X utf8 tools/run_adsb_rnn.py --config "$config" --variant "$variant" --seed "$seed" --held-out "$month" $flag --output "${root}/${variant}_s${seed}" --threads 8)
  fi
  if "${cmd[@]}" > "$log" 2>&1; then
    echo "done $month $(date -u +%FT%TZ)" >> "$marker"
  else
    echo "FAILED $month $(date -u +%FT%TZ)" >> "$marker"
    exit 1
  fi
done
echo "chain done $(date -u +%FT%TZ)" >> "$marker"
