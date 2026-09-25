#!/usr/bin/env bash
# adsb_ctx2 design on the GPU worker: every declared (variant, seed) of configs/adsb_ctx2_design_v1.json "runs", LANES at a time
# (default 2) on the one GPU, via tools/run_adsb_ctx2.py --fold design. Existing finished runs are skipped (run.json present).
# Logs runs/adsb_ctx2/logs/<variant>_s<seed>.log; marker lines in runs/adsb_ctx2/logs/chain.log. Usage: tools/ctx2_design_chain.sh [LANES]
set -u
cd ~/prc
lanes=${1:-2}
mkdir -p runs/adsb_ctx2/logs
.venv/bin/python -c 'import json;c=json.load(open("configs/adsb_ctx2_design_v1.json"));[print(v,s) for v,ss in c["runs"].items() for s in ss]' > runs/adsb_ctx2/logs/jobs.txt
echo "chain start $(date -u +%FT%TZ) lanes $lanes jobs $(wc -l < runs/adsb_ctx2/logs/jobs.txt)" >> runs/adsb_ctx2/logs/chain.log
run_one() {
  v=$1; s=$2
  out=runs/adsb_ctx2/design/${v}_s${s}
  [ -f "$out/design/run.json" ] && { echo "skip ${v}_s${s}" >> runs/adsb_ctx2/logs/chain.log; return 0; }
  if .venv/bin/python -X utf8 tools/run_adsb_ctx2.py --config configs/adsb_ctx2_design_v1.json --variant "$v" --seed "$s" --fold design \
       --output "$out" --device cuda --threads 4 > runs/adsb_ctx2/logs/${v}_s${s}.log 2>&1; then
    echo "done ${v}_s${s} $(date -u +%FT%TZ)" >> runs/adsb_ctx2/logs/chain.log
  else
    echo "FAILED ${v}_s${s} $(date -u +%FT%TZ)" >> runs/adsb_ctx2/logs/chain.log
  fi
}
export -f run_one
xargs -P "$lanes" -L 1 bash -c 'run_one "$0" "$1"' < runs/adsb_ctx2/logs/jobs.txt
echo "chain done $(date -u +%FT%TZ)" >> runs/adsb_ctx2/logs/chain.log
