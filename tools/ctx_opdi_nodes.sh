#!/usr/bin/env bash
# Lever 9 (fork J; configs/ctx_opdi_design_v1.json, infra/aws/gpu_authorization_20261007.json): node operations, run in WSL from the project
# root. A copy of tools/ctx_own2_nodes.sh for this lever's own clusters prc-ctx5a-l (never prc-ctx3*/prc-ctx4* or sky-jobs-controller-*):
#   on-demand  prc-ctx5a (head) / b / c / d us-east-2, prc-ctx5e (head) / f us-east-1, prc-ctx5g (head) / h us-west-2
#   spot       prc-ctx5i (head) / j us-east-2, prc-ctx5k (head) us-east-1, prc-ctx5l us-west-2
# Heads launch with their region's head yaml (data uploaded from the workstation by sky, token build, tests); the others with the worker yaml
# and get the data by `fanout` from a ready node (FROM).
#   up       sky launch the clusters in NODES, autodown after 45 idle minutes (the guard then sets STOP mode for on-demand); RETRY_MIN > 0 retries a
#            launch that failed for capacity every 3 minutes for that many minutes
#   fanout   copy FROM's data/ and runs/adsb_nn/features to the other NODES (agent-forwarded rsync), then run the checks on every node
#   check    token hashes, own-points + OPDI table hashes, context tests; a node that passes gets ~/prc/.opdi_ready
#   sync     push src/ tools/ configs/ tests/ and the two feature manifests to every node
#   down     sky down the clusters in NODES (only prc-ctx5*)
# Usage: NODES="prc-ctx5a prc-ctx5b" [FROM=prc-ctx5a] [RETRY_MIN=60] tools/ctx_opdi_nodes.sh up|fanout|check|sync|down
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a nodes <<< "${NODES:-prc-ctx5a prc-ctx5b prc-ctx5c prc-ctx5d}"
for n in "${nodes[@]}"; do case "$n" in prc-ctx5[a-l]) ;; *) echo "refusing node $n"; exit 2 ;; esac; done
R=runs/ctx_opdi
A=infra/aws
yaml_for() {
  case "$1" in
    prc-ctx5a) echo $A/sky_ctx5_head_use2.yaml ;;
    prc-ctx5b|prc-ctx5c|prc-ctx5d) echo $A/sky_ctx5_worker_use2.yaml ;;
    prc-ctx5e) echo $A/sky_ctx5_head_use1.yaml ;;
    prc-ctx5f) echo $A/sky_ctx5_worker_use1.yaml ;;
    prc-ctx5g) echo $A/sky_ctx5_head_usw2.yaml ;;
    prc-ctx5h) echo $A/sky_ctx5_worker_usw2.yaml ;;
    prc-ctx5i) echo $A/sky_ctx5_head_use2_spot.yaml ;;
    prc-ctx5j) echo $A/sky_ctx5_worker_use2_spot.yaml ;;
    prc-ctx5k) echo $A/sky_ctx5_head_use1_spot.yaml ;;
    prc-ctx5l) echo $A/sky_ctx5_worker_usw2_spot.yaml ;;
  esac
}
checks="cd ~/prc && .venv/bin/python -X utf8 tools/ctx3_check_tokens.py | head -c 120 && echo && .venv/bin/python -X utf8 tools/ctx_opdi_check_features.py && .venv/bin/python -m pytest tests/test_ctx.py tests/test_ctx2.py tests/test_ctx3.py -q -p no:cacheprovider >/dev/null 2>&1 && touch ~/prc/.opdi_ready && echo READY"
case "${1:-}" in
  up)
    source ~/.venv-sky/bin/activate
    mkdir -p "$R/logs"
    for n in "${nodes[@]}"; do
      ( y=$(yaml_for "$n"); end=$(( $(date +%s) + ${RETRY_MIN:-0} * 60 )); k=0
        while true; do
          k=$((k + 1))
          log="$R/logs/launch_${n}_$(date -u +%H%M%S)_try${k}.log"
          if sky launch -c "$n" "$y" -i 45 --down -y > "$log" 2>&1; then echo "$n up ($(date -u +%T), try $k, $y)"; break; fi
          if grep -q -E "InsufficientInstanceCapacity|Failed to acquire resources|SpotMaxPriceTooLow|MaxSpotInstanceCountExceeded|capacity-not-available|VcpuLimitExceeded" "$log" && [ "$(date +%s)" -lt "$end" ]; then
            source ~/.venv-sky/bin/activate; sky down "$n" -y > /dev/null 2>&1 || true; sleep 180; continue
          fi
          echo "$n launch FAILED ($(date -u +%T), try $k; see $log)"; break
        done ) &
    done
    wait ;;
  fanout)
    src=${FROM:?FROM=<node with data>}
    eval "$(ssh-agent -s)" > /dev/null
    trap 'ssh-agent -k > /dev/null' EXIT
    ssh-add -q ~/.sky/generated/ssh-keys/${src}.key 2>/dev/null || ssh-add -q ~/.sky/generated/ssh-keys/*.key
    src_ip=$(awk '/HostName/{print $2}' ~/.sky/generated/ssh/${src})
    opts="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    for w in "${nodes[@]}"; do
      [ "$w" = "$src" ] && continue
      ( ssh -A "$w" "cd ~/prc && mkdir -p data runs/adsb_nn/features && rsync -a -e '$opts' ubuntu@$src_ip:prc/data/ data/ && rsync -a -e '$opts' ubuntu@$src_ip:prc/runs/adsb_nn/features/ runs/adsb_nn/features/ && echo copied" \
          && echo "$w copy ok" || echo "$w copy FAILED" ) &
    done
    wait
    for n in "${nodes[@]}"; do ( ssh "$n" "$checks" 2>&1 | tail -3 | sed "s/^/$n: /" ) & done
    wait ;;
  check)
    for n in "${nodes[@]}"; do ( ssh "$n" "$checks" 2>&1 | tail -3 | sed "s/^/$n: /" ) & done
    wait ;;
  sync)
    for n in "${nodes[@]}"; do
      ( rsync -a --delete -e ssh src/ "$n":prc/src/ --exclude __pycache__ && rsync -a -e ssh tools/ "$n":prc/tools/ --exclude __pycache__ \
        && rsync -a -e ssh configs/ "$n":prc/configs/ && rsync -a --delete -e ssh tests/ "$n":prc/tests/ --exclude __pycache__ \
        && ssh "$n" "mkdir -p ~/prc/docs/dig ~/prc/docs/ctx_opdi" && rsync -a -e ssh docs/dig/ctx_ownpoints_features_manifest.json "$n":prc/docs/dig/ \
        && { [ ! -f docs/ctx_opdi/features_manifest.json ] || rsync -a -e ssh docs/ctx_opdi/features_manifest.json "$n":prc/docs/ctx_opdi/; } && echo "$n synced" ) &
    done
    wait ;;
  down)
    source ~/.venv-sky/bin/activate
    sky down "${nodes[@]}" -y ;;
  *) echo "usage: NODES=\"...\" $0 up|fanout|check|sync|down"; exit 2 ;;
esac
