#!/usr/bin/env bash
# adsb_ctx round E (configs/adsb_ctx_v6.json): two lanes per worker, worker arcwN runs seeds 501+N (lane A) and 509+N (lane B), each
# through seasonal, forward, refit and refit12 with configs/adsb_ctx_network_v4.json variant ctx_e32c (32 epochs), 8 threads.
# Outputs runs/adsb_ctx/ctx_e32c_s<seed>/<fold>/. usage: tools/ctx_launch_dev6.sh N [N ...]  (worker indices, e.g. 0 2 6 7)
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -gt 0 ] || { echo "usage: $0 N [N ...]"; exit 2; }
for n in "$@"; do
  a=$((501 + n)); b=$((509 + n))
  tar cf - configs/adsb_ctx_network_v4.json configs/adsb_ctx_v6.json | timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && tar xf - && sha256sum configs/adsb_ctx_network_v4.json | cut -c1-16"
  lanes=""
  for seed in $a $b; do
    # never start a seed twice: an existing chain marker means it was launched already
    lanes="$lanes if [ -e runs/adsb_ctx/logs/chain_ctx_e32c_s$seed.log ]; then echo seed $seed already launched; else (setsid nohup tools/ctx_chain.sh configs/adsb_ctx_network_v4.json ctx_e32c $seed runs/adsb_ctx seasonal forward refit refit12 > /dev/null 2>&1 < /dev/null &); fi;"
  done
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_ctx/logs && $lanes echo arcw$n launched seeds $a $b" || echo "arcw$n: launch returned $?"
done
