#!/usr/bin/env bash
# adsb_rnn design (configs/adsb_rnn_design_v1.json): one October design fit per worker -- rnb seeds 1-4 on arcw0-arcw3, rwide seeds
# 1-4 on arcw4-arcw7, 8 threads each.
set -euo pipefail
cd "$(dirname "$0")/.."
for n in 0 1 2 3 4 5 6 7; do
  if [ $n -lt 4 ]; then variant=rnb; seed=$((n + 1)); else variant=rwide; seed=$((n - 3)); fi
  lane="(setsid nohup tools/rnn_chain.sh configs/adsb_rnn_network_v1.json $variant $seed runs/adsb_rnn/design design 2025-10 > /dev/null 2>&1 < /dev/null &);"
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_rnn/logs && $lane echo arcw$n launched $variant s$seed" || echo "arcw$n: launch command returned $?"
done
