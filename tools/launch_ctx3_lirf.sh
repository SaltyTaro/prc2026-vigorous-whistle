#!/usr/bin/env bash
# LIRF context-network design (configs/adsb_ctx3_lirf_design_v1.json): 8 October design fits on arcw0-arcw3, CPU, two lanes per
# worker (8 threads each): arcwN runs lref32 seed 21+N and lm25_32 seed 21+N side by side.
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
C=configs/adsb_ctx3_lirf_design_v1.json
for n in 0 1 2 3; do
  s=$((21+n))
  scp -q $C arcw$n:$R/configs/
  timeout 60 ssh -o BatchMode=yes arcw$n "mkdir -p $R/runs/adsb_ctx3_lirf/logs && cd $R && for v in lref32 lm25_32; do (setsid nohup .venv/bin/python -u tools/run_adsb_ctx3.py --config $C --variant \$v --seed $s --fold design --output runs/adsb_ctx3_lirf/design/\${v}_s$s --device cpu --threads 8 > runs/adsb_ctx3_lirf/logs/\${v}_s$s.log 2>&1 < /dev/null &); done; echo arcw$n launched seed $s"
done
