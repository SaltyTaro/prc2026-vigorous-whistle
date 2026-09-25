#!/usr/bin/env bash
# adsb_ctx development (configs/adsb_ctx_v1.json): two lanes per worker, seeds 101-108 (lane A) and 109-116 (lane B) on arcw0-arcw7;
# each lane runs seasonal, forward, refit and refit12 for its seed with configs/adsb_ctx_network_v1.json, 8 threads.
# Outputs runs/adsb_ctx/ctx_s<seed>/<fold>/.
set -euo pipefail
cd "$(dirname "$0")/.."
for n in 0 1 2 3 4 5 6 7; do
  a=$((101 + n)); b=$((109 + n))
  tar cf - configs/adsb_ctx_network_v1.json configs/adsb_ctx_v1.json | timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && tar xf - && sha256sum configs/adsb_ctx_network_v1.json | cut -c1-16"
  lanes=""
  for seed in $a $b; do
    lanes="$lanes (setsid nohup tools/ctx_chain.sh configs/adsb_ctx_network_v1.json ctx $seed runs/adsb_ctx seasonal forward refit refit12 > /dev/null 2>&1 < /dev/null &);"
  done
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_ctx/logs && $lanes echo arcw$n launched seeds $a $b" || echo "arcw$n: launch returned $?"
done
