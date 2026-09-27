#!/usr/bin/env bash
# Time-gap test (configs/adsb_ctx_gap_v1.json) on one GPU node: seed SEED, both folds (gap, near) as two lanes, each lane running
# g32, g24, g16 in turn via tools/run_adsb_ctx_gap.py. Finished runs (run.json present) are skipped.
# Logs runs/adsb_ctx_gap/logs/<variant>_s<seed>_<fold>.log; markers in runs/adsb_ctx_gap/logs/chain.log. Usage: tools/ctx_gap_chain.sh SEED
set -u
cd ~/prc
seed=$1
mkdir -p runs/adsb_ctx_gap/logs
echo "chain start $(date -u +%FT%TZ) seed $seed" >> runs/adsb_ctx_gap/logs/chain.log
lane() {
  fold=$1
  for v in g32 g24 g16; do
    out=runs/adsb_ctx_gap/${v}_s${seed}
    [ -f "$out/$fold/run.json" ] && { echo "skip ${v}_s${seed}_${fold}" >> runs/adsb_ctx_gap/logs/chain.log; continue; }
    echo "start ${v}_s${seed}_${fold} $(date -u +%FT%TZ)" >> runs/adsb_ctx_gap/logs/chain.log
    if .venv/bin/python -X utf8 tools/run_adsb_ctx_gap.py --config configs/adsb_ctx_gap_v1.json --variant "$v" --seed "$seed" --fold "$fold" \
         --output "$out" --device cuda --threads 3 > runs/adsb_ctx_gap/logs/${v}_s${seed}_${fold}.log 2>&1; then
      echo "done ${v}_s${seed}_${fold} $(date -u +%FT%TZ)" >> runs/adsb_ctx_gap/logs/chain.log
    else
      echo "FAILED ${v}_s${seed}_${fold} $(date -u +%FT%TZ)" >> runs/adsb_ctx_gap/logs/chain.log
    fi
  done
}
lane gap & lane near & wait
echo "chain done $(date -u +%FT%TZ) seed $seed" >> runs/adsb_ctx_gap/logs/chain.log
