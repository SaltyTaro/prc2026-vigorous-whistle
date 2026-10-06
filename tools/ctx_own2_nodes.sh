#!/usr/bin/env bash
# Lever 5 (fork E; configs/ctx_own2_design_v1.json, infra/aws/gpu_authorization_20261006b.json + amendment 1): node operations, run in WSL from
# the project root. A copy of tools/combo_nodes.sh for the lever's own clusters (never prc-ctx3a-d, fork B's, nor sky-jobs-controller-*):
#   on-demand  prc-ctx4a / prc-ctx4b us-east-1, prc-ctx4c / prc-ctx4d us-west-2
#   spot       prc-ctx4e / prc-ctx4f us-east-1, prc-ctx4g / prc-ctx4h us-west-2, prc-ctx4i-l us-east-2 (only after fork B's nodes are up or B is NO GO)
# Nodes named in HEADS (default: prc-ctx4a prc-ctx4c prc-ctx4e prc-ctx4g) launch with their region's head yaml (data uploaded from the workstation,
# token build, tests); the others with the worker yaml and get the data by `fanout` from a node that has it (FROM).
#   up       sky launch the clusters in NODES, autodown after 45 idle minutes; RETRY_MIN > 0 retries a launch that failed for capacity
#            (InsufficientInstanceCapacity / spot unavailability) every 3 minutes for that many minutes
#   fanout   copy FROM's data/ (incl. the token build and the own-points tables) and runs/adsb_nn/features to the other NODES (agent-forwarded rsync,
#            across regions when needed), then on every node check the token hashes, the own-points table hashes and run the context tests;
#            a node that passes gets the marker ~/prc/.own2_ready
#   check    the same checks without copying (for heads: marks them ready)
#   sync     push src/ tools/ configs/ tests/ docs/dig/ctx_ownpoints_features_manifest.json to every node
#   down     sky down the clusters in NODES (only prc-ctx4*)
# Fits are dispatched, mirrored and re-run after pre-emption by tools/ctx_own2_dispatch.py (one sky exec per fit, two per GPU).
# Usage: NODES="prc-ctx4a prc-ctx4e" [HEADS=...] [FROM=prc-ctx4a] [RETRY_MIN=60] tools/ctx_own2_nodes.sh up|fanout|check|sync|down
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a nodes <<< "${NODES:-prc-ctx4a prc-ctx4b prc-ctx4c prc-ctx4d}"
for n in "${nodes[@]}"; do case "$n" in prc-ctx4[a-l]) ;; *) echo "refusing node $n"; exit 2 ;; esac; done
HEADS=" ${HEADS:-prc-ctx4a prc-ctx4c prc-ctx4e prc-ctx4g} "
R=runs/ctx_own2
yaml_for() {
  local n=$1 role=worker
  [[ "$HEADS" == *" $n "* ]] && role=head
  case "$n" in
    prc-ctx4a|prc-ctx4b) [ $role = head ] && echo infra/aws/sky_ctx4_head_use1.yaml || echo infra/aws/sky_ctx4_worker_use1.yaml ;;
    prc-ctx4c|prc-ctx4d) [ $role = head ] && echo infra/aws/sky_ctx4_head_usw2.yaml || echo infra/aws/sky_ctx4_worker_usw2.yaml ;;
    prc-ctx4e|prc-ctx4f) [ $role = head ] && echo infra/aws/sky_ctx4_head_use1_spot.yaml || echo infra/aws/sky_ctx4_worker_use1_spot.yaml ;;
    prc-ctx4g|prc-ctx4h) [ $role = head ] && echo infra/aws/sky_ctx4_head_usw2_spot.yaml || echo infra/aws/sky_ctx4_worker_usw2_spot.yaml ;;
    prc-ctx4i|prc-ctx4j|prc-ctx4k|prc-ctx4l)
      [ -f $R/use2_allowed ] || { echo "us-east-2 spot is not allowed yet ($R/use2_allowed absent)" >&2; exit 2; }
      [ $role = head ] && echo infra/aws/sky_ctx4_head_use2_spot.yaml || echo infra/aws/sky_ctx4_worker_use2_spot.yaml ;;
  esac
}
checks="cd ~/prc && .venv/bin/python -X utf8 tools/ctx3_check_tokens.py | head -c 120 && echo && .venv/bin/python -X utf8 tools/ctx_own2_check_features.py && .venv/bin/python -m pytest tests/test_ctx.py tests/test_ctx2.py tests/test_ctx3.py -q -p no:cacheprovider >/dev/null 2>&1                 && touch ~/prc/.own2_ready && echo READY"
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
          if grep -q -E "InsufficientInstanceCapacity|Failed to acquire resources|SpotMaxPriceTooLow|MaxSpotInstanceCountExceeded|capacity-not-available" "$log" && [ "$(date +%s)" -lt "$end" ]; then
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
        && ssh "$n" "mkdir -p ~/prc/docs/dig" && rsync -a -e ssh docs/dig/ctx_ownpoints_features_manifest.json "$n":prc/docs/dig/ && echo "$n synced" ) &
    done
    wait ;;
  down)
    source ~/.venv-sky/bin/activate
    sky down "${nodes[@]}" -y ;;
  *) echo "usage: NODES=\"...\" $0 up|fanout|check|sync|down"; exit 2 ;;
esac
