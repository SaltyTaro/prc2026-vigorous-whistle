#!/usr/bin/env bash
# Lever 5 (fork E): bring up the nodes of the development round (configs/ctx_own2_dev_v1.json): restart the STOPPED design nodes (prc-ctx4b
# us-east-1, prc-ctx4i / prc-ctx4j us-east-2; their disks keep the data, the token build and the venv) and launch two more on-demand nodes
# in us-east-2 (prc-ctx4k / prc-ctx4l with the on-demand worker yaml, STOP-mode autostop after 30 idle minutes; data by the dispatcher's
# fanout). Capacity failures are retried every 3 minutes for RETRY_MIN minutes (default 60), then each loop ends by itself.
# Usage (WSL, project root): setsid nohup bash tools/ctx_own2_up_dev.sh >> runs/ctx_own2/up_dev.stdout 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
source ~/.venv-sky/bin/activate
R=runs/ctx_own2
mkdir -p $R/logs
end=$(( $(date +%s) + ${RETRY_MIN:-60} * 60 ))
retry() {  # name, command...
  local n=$1 k=0 log; shift
  while true; do
    k=$((k + 1)); log="$R/logs/dev_up_${n}_$(date -u +%H%M%S)_try${k}.log"
    if "$@" > "$log" 2>&1; then echo "$n up ($(date -u +%T), try $k)"; return 0; fi
    if grep -q -E "InsufficientInstanceCapacity|Failed to acquire resources|capacity-not-available" "$log" && [ "$(date +%s)" -lt "$end" ]; then sleep 180; continue; fi
    echo "$n FAILED ($(date -u +%T), try $k; see $log)"; return 1
  done
}
for n in prc-ctx4b prc-ctx4i prc-ctx4j; do retry "$n" sky start "$n" -y & done
for n in prc-ctx4k prc-ctx4l; do retry "$n" sky launch -c "$n" infra/aws/sky_ctx4_worker_use2.yaml -i 30 -y & done
wait
