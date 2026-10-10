#!/usr/bin/env bash
# Tier B network half (lane G; configs/febjun_tierb_v1.json, infra/aws/gpu_authorization_20261010.json): node operations, run in WSL from the
# project root. tools/season_net_nodes.sh's pattern with this lane's own clusters prc-ctx8a-l (never prc-ctx3*-prc-ctx7*, sky-jobs-controller-*
# or another project's managed job):
#   on-demand  prc-ctx8a (head) / b / c / d us-east-2, prc-ctx8e (head) / f us-east-1, prc-ctx8g (head) / h us-west-2
#   spot       prc-ctx8i (head) / j us-east-2, prc-ctx8k (head) us-east-1, prc-ctx8l us-west-2
# Heads launch with their region's head yaml (data uploaded from the workstation by sky, token build, tests); the others with the worker yaml
# and get the data by `fanout` from a ready node (FROM).
#   up       sky launch the clusters in NODES, autodown after 45 idle minutes; RETRY_MIN > 0 retries a capacity failure every 3 minutes
#   fanout   copy FROM's data/ and runs/adsb_nn/features to the other NODES (agent-forwarded rsync), then run the checks on every node
#   check    token hashes, own-points table hashes, context tests; a node that passes gets ~/prc/.febjun_ready
#   sync     push src/ tools/ configs/ tests/, the own-points manifest and runs/febjun_nets/jobs to every node
#   launch   start tools/febjun_net_queue.sh on node i of NODES (job list node<i>.txt) as a SkyPilot job, 2 lanes; refuses a running queue
#   status   one line per node: done / failed / running / last marker
#   mirror   rsync runs/febjun_nets/nets and the node logs back, then write each node's file hashes to runs/febjun_nets/node_hashes/<node>.txt
#   down     sky down the clusters in NODES (only prc-ctx8*)
# Usage: NODES="prc-ctx8a prc-ctx8b" [FROM=prc-ctx8a] [RETRY_MIN=60] tools/febjun_net_nodes.sh up|fanout|check|sync|launch|status|mirror|down
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a nodes <<< "${NODES:-prc-ctx8a prc-ctx8b prc-ctx8c prc-ctx8d}"
for n in "${nodes[@]}"; do case "$n" in prc-ctx8[a-l]) ;; *) echo "refusing node $n"; exit 2 ;; esac; done
R=runs/febjun_nets
A=infra/aws
yaml_for() {
  case "$1" in
    prc-ctx8a) echo $A/sky_ctx8_head_use2.yaml ;;
    prc-ctx8b|prc-ctx8c|prc-ctx8d) echo $A/sky_ctx8_worker_use2.yaml ;;
    prc-ctx8e) echo $A/sky_ctx8_head_use1.yaml ;;
    prc-ctx8f) echo $A/sky_ctx8_worker_use1.yaml ;;
    prc-ctx8g) echo $A/sky_ctx8_head_usw2.yaml ;;
    prc-ctx8h) echo $A/sky_ctx8_worker_usw2.yaml ;;
    prc-ctx8i) echo $A/sky_ctx8_head_use2_spot.yaml ;;
    prc-ctx8j) echo $A/sky_ctx8_worker_use2_spot.yaml ;;
    prc-ctx8k) echo $A/sky_ctx8_head_use1_spot.yaml ;;
    prc-ctx8l) echo $A/sky_ctx8_worker_usw2_spot.yaml ;;
  esac
}
checks="cd ~/prc && .venv/bin/python -X utf8 tools/ctx3_check_tokens.py | head -c 120 && echo && .venv/bin/python -X utf8 tools/ctx_own2_check_features.py && .venv/bin/python -m pytest tests/test_ctx.py tests/test_ctx2.py tests/test_ctx3.py -q -p no:cacheprovider >/dev/null 2>&1 && touch ~/prc/.febjun_ready && echo READY"
case "${1:-}" in
  up)
    source ~/.venv-sky/bin/activate
    mkdir -p "$R/launch_logs"
    for n in "${nodes[@]}"; do
      ( y=$(yaml_for "$n"); end=$(( $(date +%s) + ${RETRY_MIN:-0} * 60 )); k=0
        while true; do
          k=$((k + 1))
          log="$R/launch_logs/launch_${n}_$(date -u +%H%M%S)_try${k}.log"
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
        && ssh "$n" "mkdir -p ~/prc/docs/dig ~/prc/$R/jobs" && rsync -a -e ssh docs/dig/ctx_ownpoints_features_manifest.json "$n":prc/docs/dig/ \
        && rsync -a --delete -e ssh $R/jobs/ "$n":prc/$R/jobs/ && echo "$n synced" ) &
    done
    wait ;;
  launch)
    source ~/.venv-sky/bin/activate
    for i in "${!nodes[@]}"; do
      n=${nodes[$i]}
      if ssh "$n" "pgrep -f '[f]ebjun_net_queue.sh' > /dev/null"; then echo "$n: queue already running"; continue; fi
      sky exec "$n" --gpus L40S:1 -d --name "febjun-queue-$i" "cd ~/prc && bash tools/febjun_net_queue.sh $i 2" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Job (ID|submitted)|Error' | sed "s/^/$n: /"
    done ;;
  status)
    for n in "${nodes[@]}"; do
      ssh -o ConnectTimeout=15 "$n" "cd ~/prc/$R && echo done \$(find nets -name run.json 2>/dev/null | wc -l) failed \$(grep -c '^FAILED' logs/chain.log 2>/dev/null) running \$(pgrep -fc '[r]un_adsb_ctx_lomo.py') last: \$(tail -1 logs/chain.log 2>/dev/null)" 2>&1 | sed "s/^/$n: /" || echo "$n: unreachable"
    done ;;
  mirror)
    mkdir -p $R/nets $R/node_logs $R/node_hashes
    for n in "${nodes[@]}"; do
      rsync -a -e ssh "$n":prc/$R/nets/ $R/nets/ && mkdir -p $R/node_logs/$n && rsync -a -e ssh "$n":prc/$R/logs/ $R/node_logs/$n/ \
        && ssh "$n" "cd ~/prc && find $R/nets -type f -print0 | sort -z | xargs -0 -r sha256sum" > $R/node_hashes/$n.txt && echo "$n mirrored ($(wc -l < $R/node_hashes/$n.txt) files hashed)"
    done ;;
  down)
    source ~/.venv-sky/bin/activate
    sky down "${nodes[@]}" -y ;;
  *) echo "usage: NODES=\"...\" $0 up|fanout|check|sync|launch|status|mirror|down"; exit 2 ;;
esac
