#!/usr/bin/env bash
# Lever 5 (fork E): launch the us-east-2 nodes allowed by amendment 2 (fork B NO GO): prc-ctx4i / prc-ctx4j on-demand (STOP-mode autostop after
# 30 idle minutes), prc-ctx4k / prc-ctx4l spot (autodown after 30 idle minutes), all worker yamls (data by the dispatcher's fanout from a ready
# node). Capacity failures are retried every 3 minutes for RETRY_MIN minutes (default 60), then the loop ends by itself.
# Separate from tools/ctx_own2_nodes.sh so that file is not edited while its launch loop runs. Usage (WSL, project root): tools/ctx_own2_up_use2.sh
set -u
cd "$(dirname "$0")/.."
source ~/.venv-sky/bin/activate
R=runs/ctx_own2
mkdir -p $R/logs
end=$(( $(date +%s) + ${RETRY_MIN:-60} * 60 ))
one() {
  local n=$1 y=$2 flags=$3 k=0 log
  while true; do
    k=$((k + 1)); log="$R/logs/launch_${n}_$(date -u +%H%M%S)_try${k}.log"
    if sky launch -c "$n" "$y" $flags -y > "$log" 2>&1; then echo "$n up ($(date -u +%T), try $k, $y)"; return 0; fi
    if grep -q -E "InsufficientInstanceCapacity|Failed to acquire resources|capacity-not-available|MaxSpotInstanceCountExceeded" "$log" && [ "$(date +%s)" -lt "$end" ]; then
      sky down "$n" -y > /dev/null 2>&1 || true; sleep 180; continue
    fi
    echo "$n launch FAILED ($(date -u +%T), try $k; see $log)"; return 1
  done
}
one prc-ctx4i infra/aws/sky_ctx4_worker_use2.yaml "-i 30" &
one prc-ctx4j infra/aws/sky_ctx4_worker_use2.yaml "-i 30" &
one prc-ctx4k infra/aws/sky_ctx4_worker_use2_spot.yaml "-i 30 --down" &
one prc-ctx4l infra/aws/sky_ctx4_worker_use2_spot.yaml "-i 30 --down" &
wait
