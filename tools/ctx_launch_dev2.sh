#!/usr/bin/env bash
# adsb_ctx round A (configs/adsb_ctx_v2.json): two lanes per worker, worker arcwN runs seeds 117+N (lane A) and 125+N (lane B), each
# through seasonal, forward, refit and refit12 with configs/adsb_ctx_network_v2.json variant ctx_e24, 8 threads.
# Outputs runs/adsb_ctx/ctx_e24_s<seed>/<fold>/. usage: tools/ctx_launch_dev2.sh N [N ...]  (worker indices, e.g. 2 3 6)
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: $0 N [N ...]"; exit 2; }
for n in "$@"; do
  a=$((117 + n)); b=$((125 + n))
  tar cf - configs/adsb_ctx_network_v2.json configs/adsb_ctx_v2.json | timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && tar xf - && sha256sum configs/adsb_ctx_network_v2.json | cut -c1-16"
  lanes=""
  for seed in $a $b; do
    # never start a seed twice: an existing chain marker means it was launched already
    lanes="$lanes if [ -e runs/adsb_ctx/logs/chain_ctx_e24_s$seed.log ]; then echo seed $seed already launched; else (setsid nohup tools/ctx_chain.sh configs/adsb_ctx_network_v2.json ctx_e24 $seed runs/adsb_ctx seasonal forward refit refit12 > /dev/null 2>&1 < /dev/null &); fi;"
  done
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_ctx/logs && $lanes echo arcw$n launched seeds $a $b" || echo "arcw$n: launch returned $?"
done
