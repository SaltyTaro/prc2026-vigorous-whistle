#!/usr/bin/env bash
# adsb_nn3: launch the declared nnlr development chains on arcw0-arcw7, two lanes per worker (two 8-thread jobs, 1.33x throughput
# measured in adsb_nn2). Worker n runs seeds 401+n and 409+n, each: seasonal forward refit, then its twelve-month refit (scheduling
# only; used only if the selection needs it).
set -euo pipefail
cd "$(dirname "$0")/.."
for n in 0 1 2 3 4 5 6 7; do
  lanes=""
  for k in 0 8; do
    s=$((401 + n + k))
    lanes="$lanes (setsid nohup tools/nn2_chain.sh configs/adsb_nn3_network_v1.json nnlr $s runs/adsb_nn3 seasonal forward refit refit12 > /dev/null 2>&1 < /dev/null &);"
  done
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_nn2/logs && $lanes echo arcw$n launched" || echo "arcw$n: launch command returned $?"
done
