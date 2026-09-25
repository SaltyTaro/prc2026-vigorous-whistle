#!/usr/bin/env bash
# adsb_ctx design (configs/adsb_ctx_design_v1.json round_1_hosts): ctx seeds 1-6 on arcw0-arcw5, ctx_noadsb s1 on arcw6, ctx_dep s1 on
# arcw7; one October design fit per worker, 8 threads. Outputs runs/adsb_ctx/design/<variant>_s<seed>/design/.
set -euo pipefail
cd "$(dirname "$0")/.."
jobs=("ctx 1" "ctx 2" "ctx 3" "ctx 4" "ctx 5" "ctx 6" "ctx_noadsb 1" "ctx_dep 1")
for n in 0 1 2 3 4 5 6 7; do
  set -- ${jobs[$n]}
  lane="(setsid nohup tools/ctx_chain.sh configs/adsb_ctx_design_v1.json $1 $2 runs/adsb_ctx/design design > /dev/null 2>&1 < /dev/null &);"
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_ctx/logs && $lane echo arcw$n launched $1 s$2" || echo "arcw$n: launch command returned $?"
done
