#!/usr/bin/env bash
# adsb_ctx round B design (configs/adsb_ctx_lirf_design_v1.json): one October design fit per worker as a third, 4-thread lane beside
# round A's two lanes. usage: tools/ctx_launch_lirf_design.sh N:VARIANT:SEED [...]   e.g. 2:ctx_lirf:1 3:ctx_lirf_noadsb:1
# Outputs runs/adsb_ctx/lirf_design/<variant>_s<seed>/design/; log runs/adsb_ctx/logs/lirf_<variant>_s<seed>_design.log; marker
# runs/adsb_ctx/logs/lirf_<variant>_s<seed>.marker ("done <time>" or "FAILED <time>"). Refuses a job whose log already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: $0 N:VARIANT:SEED [...]"; exit 2; }
for job in "$@"; do
  IFS=: read -r n variant seed <<< "$job"
  tag="lirf_${variant}_s${seed}"
  tar cf - configs/adsb_ctx_lirf_design_v1.json | timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && tar xf - && sha256sum configs/adsb_ctx_lirf_design_v1.json | cut -c1-16"
  run=".venv/bin/python -X utf8 tools/run_adsb_ctx.py --config configs/adsb_ctx_lirf_design_v1.json --variant $variant --seed $seed --fold design --output runs/adsb_ctx/lirf_design/${variant}_s${seed} --threads 4"
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_ctx/logs && if [ -e runs/adsb_ctx/logs/${tag}_design.log ]; then echo arcw$n $tag already launched; else (setsid nohup bash -c '$run > runs/adsb_ctx/logs/${tag}_design.log 2>&1 && echo done \$(date -u +%FT%TZ) > runs/adsb_ctx/logs/${tag}.marker || echo FAILED \$(date -u +%FT%TZ) > runs/adsb_ctx/logs/${tag}.marker' > /dev/null 2>&1 < /dev/null &); echo arcw$n launched $tag; fi" || echo "arcw$n: launch returned $?"
done
