#!/usr/bin/env bash
# adsb_nn2: launch the declared development chains on arcw0-arcw7, two lanes per worker (two 8-thread jobs; measured 1.2x one job's
# throughput in adsb_nn). Worker n runs committee-pool seeds C[n], C[n+8] and LIRF-inclusive seeds L[n], L[n+8]:
#   lane 1: C[n] seasonal forward refit ; L[n] seasonal forward refit ; C[n] refit12 ; L[n] refit12
#   lane 2: C[n+8] ... ; L[n+8] ...
# The development folds of every seed come before any twelve-month refit (refits are for scheduling only; used only if selected).
# usage: tools/nn2_launch_dev.sh COMMITTEE_VARIANT   (nnb -> seeds 117-132 with tools/nn_chain.sh; nmres -> seeds 301-316 with nn2_chain.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
variant=$1
case $variant in
  nnb) cfirst=117 ;;
  nmres) cfirst=301 ;;
  *) echo "unknown committee variant $variant"; exit 1 ;;
esac
committee_chain() {  # seed folds...
  local seed=$1; shift
  if [ "$variant" = nnb ]; then echo "tools/nn_chain.sh configs/adsb_nn_network_v1.json nnb $seed $*"
  else echo "tools/nn2_chain.sh configs/adsb_nn2_network_v1.json nmres $seed runs/adsb_nn2 $*"; fi
}
nnl_chain() { local seed=$1; shift; echo "tools/nn2_chain.sh configs/adsb_nn2_network_v1.json nnl $seed runs/adsb_nn2 $*"; }
for n in 0 1 2 3 4 5 6 7; do
  lanes=""
  for k in 0 8; do
    c=$((cfirst + n + k)); l=$((201 + n + k))
    lane="$(committee_chain $c seasonal forward refit); $(nnl_chain $l seasonal forward refit); $(committee_chain $c refit12); $(nnl_chain $l refit12)"
    lanes="$lanes (setsid nohup bash -c '$lane' > /dev/null 2>&1 < /dev/null &);"
  done
  timeout 60 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026 && mkdir -p runs/adsb_nn/logs runs/adsb_nn2/logs && $lanes echo arcw$n launched" || echo "arcw$n: launch command returned $?"
done
