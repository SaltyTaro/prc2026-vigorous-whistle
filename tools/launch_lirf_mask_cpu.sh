#!/usr/bin/env bash
# LIRF masked-member round (configs/adsb_ctx3_lirf_mask_dev_v1.json, hardware amendment): the 16 refit12 fits on arcw0-arcw7, CPU,
# two lanes per worker (8 threads each): arcwN fits seeds 801+2N and 802+2N. Refuses a worker where a lane is already running.
# NODES="6 7" limits the workers (a retry after a dropped connection).
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
C=configs/adsb_ctx3_lirf_mask_network_v1.json
for n in ${NODES:-0 1 2 3 4 5 6 7}; do
  scp -q tools/run_adsb_ctx_lomo.py arcw$n:$R/tools/
  scp -q $C configs/adsb_ctx3_lirf_mask_dev_v1.json arcw$n:$R/configs/
  a=$((801 + 2 * n)); b=$((802 + 2 * n))
  timeout 60 ssh -o BatchMode=yes arcw$n "cd $R && if ps -eo args | grep -q '^.venv/bin/python.*run_adsb_ctx_lomo'; then echo arcw$n busy; exit 0; fi
    mkdir -p runs/adsb_ctx3_lirf_mask/logs_cpu
    for s in $a $b; do (setsid nohup .venv/bin/python -u tools/run_adsb_ctx_lomo.py --config $C --variant lm25_32 --seed \$s --fold refit12 --output runs/adsb_ctx3_lirf_mask/nets/lm25_32_s\$s --device cpu --threads 8 > runs/adsb_ctx3_lirf_mask/logs_cpu/lm25_32_s\${s}_refit12.log 2>&1 < /dev/null &); done
    echo arcw$n launched seeds $a $b"
done
