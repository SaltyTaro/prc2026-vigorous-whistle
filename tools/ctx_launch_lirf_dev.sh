#!/usr/bin/env bash
# adsb_ctx round B development (configs/adsb_ctx_lirf_v1.json): two lanes per worker, worker arcwN runs seeds 301+N (lane A) and 309+N (lane B), each
# through seasonal, forward, refit and refit12 with configs/adsb_ctx_network_lirf_v1.json variant ctx_lirf, 8 threads.
# Outputs runs/adsb_ctx/ctx_lirf_s<seed>/<fold>/. usage: tools/ctx_launch_lirf_dev.sh N [N ...]  (worker indices)
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: $0 N [N ...]"; exit 2; }
for n in "$@"; do
  a=$((301 + n)); b=$((309 + n))
  tar cf - configs/adsb_ctx_network_lirf_v1.json configs/adsb_ctx_lirf_v1.json | timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && tar xf - && sha256sum configs/adsb_ctx_network_lirf_v1.json | cut -c1-16"
  lanes=""
  for seed in $a $b; do
    # never start a seed twice: an existing chain marker means it was launched already
    lanes="$lanes if [ -e runs/adsb_ctx/logs/chain_ctx_lirf_s$seed.log ]; then echo seed $seed already launched; else (setsid nohup tools/ctx_chain.sh configs/adsb_ctx_network_lirf_v1.json ctx_lirf $seed runs/adsb_ctx seasonal forward refit refit12 > /dev/null 2>&1 < /dev/null &); fi;"
  done
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_ctx/logs && $lanes echo arcw$n launched seeds $a $b" || echo "arcw$n: launch returned $?"
done
