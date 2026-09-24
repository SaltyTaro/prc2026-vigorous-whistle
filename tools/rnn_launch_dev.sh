#!/usr/bin/env bash
# adsb_rnn development (configs/adsb_rnn_v1.json): the 16 declared rnb seeds on the twelve-month leave-one-month-out, two lanes per
# worker (two 8-thread jobs). Worker n runs seeds 501+n and 509+n; each lane fits the recipe months first, then the other eight months,
# then its twelve-month refit (scheduling only; used only if the selection needs it).
set -euo pipefail
cd "$(dirname "$0")/.."
months="2025-01 2025-07 2025-11 2025-12 2025-02 2025-03 2025-04 2025-05 2025-06 2025-08 2025-09 2025-10 refit12"
for n in 0 1 2 3 4 5 6 7; do
  lanes=""
  for k in 0 8; do
    s=$((501 + n + k))
    lanes="$lanes (setsid nohup tools/rnn_chain.sh configs/adsb_rnn_network_v1.json rnb $s runs/adsb_rnn lomo $months > /dev/null 2>&1 < /dev/null &);"
  done
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_rnn/logs && $lanes echo arcw$n launched" || echo "arcw$n: launch command returned $?"
done
