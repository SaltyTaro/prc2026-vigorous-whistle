#!/usr/bin/env bash
# adsb_rctx design (configs/adsb_rctx_design_v1.json): October design fits of rctx, 8 threads per lane.
# usage: tools/rctx_launch_design.sh HOST LANE [LANE ...]   (LANE = comma-separated seeds run one after another in one lane, e.g. 1,3;
# a lane containing a seed whose chain marker exists is refused)
set -euo pipefail
cd "$(dirname "$0")/.."
host=$1; shift
tar cf - tools/run_adsb_rctx.py tools/rctx_chain.sh configs/adsb_rctx_network_v1.json configs/adsb_rctx_design_v1.json | timeout 60 ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar xf - && sha256sum configs/adsb_rctx_network_v1.json | cut -c1-16"
lanes=""
for lane in "$@"; do
  guard="" run=""
  for seed in ${lane//,/ }; do
    guard="$guard [ -e runs/adsb_rctx/logs/chain_rctx_s$seed.log ] ||"
    run="$run tools/rctx_chain.sh configs/adsb_rctx_network_v1.json rctx $seed runs/adsb_rctx/design design 2025-10;"
  done
  lanes="$lanes if$guard false; then echo lane $lane refused: a seed was already launched; else (setsid nohup bash -c '$run' > /dev/null 2>&1 < /dev/null &); echo launched lane $lane; fi;"
done
timeout 60 ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_rctx/logs && chmod +x tools/rctx_chain.sh && $lanes echo $host done" || echo "$host: launch returned $?"
