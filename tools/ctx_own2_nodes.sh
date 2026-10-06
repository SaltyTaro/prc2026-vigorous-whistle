#!/usr/bin/env bash
# Lever 5 (fork E; configs/ctx_own2_design_v1.json, infra/aws/gpu_authorization_20261006b.json): node operations, run in WSL from the project root.
# A copy of tools/combo_nodes.sh for the clusters prc-ctx4a (head, us-east-1), prc-ctx4b (us-east-1), prc-ctx4c / prc-ctx4d (us-west-2).
# Never touches prc-ctx3a-d (fork B) or sky-jobs-controller-*.
#   up       sky launch the clusters in NODES (head yaml for HEAD, region worker yamls for the others), autodown after 45 idle minutes;
#            RETRY_MIN > 0 retries each failed launch every 3 minutes for that many minutes (AWS g6e capacity comes and goes)
#   fanout   copy the head's data/ (incl. the token build and the own-points tables) and runs/adsb_nn/features to the other nodes (agent-forwarded
#            rsync), then on every node check the token hashes, the own-points table hashes (docs/dig/ctx_ownpoints_features_manifest.json) and
#            run the context tests
#   sync     push src/ tools/ configs/ tests/ docs/dig/ctx_ownpoints_features_manifest.json and the job lists to every node
#   launch   start tools/ctx_own2_queue.sh on every node as a SkyPilot job (node i of NODES, 2 lanes); refuses if a queue is running
#   status   one line per node: runs done / failed / running
#   mirror   copy $ROUND/$STAGE and the node logs back from every node
#   down     sky down the clusters in NODES (only prc-ctx4*)
# ROUND (default runs/ctx_own2) and STAGE (default design) name the job lists: $ROUND/jobs_$STAGE/node<i>.txt, outputs under $ROUND/$STAGE.
# Usage: [ROUND=...] [STAGE=design|dev] NODES="prc-ctx4a prc-ctx4b ..." tools/ctx_own2_nodes.sh up|fanout|sync|launch|status|mirror|down
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a nodes <<< "${NODES:-prc-ctx4a prc-ctx4b prc-ctx4c prc-ctx4d}"
for n in "${nodes[@]}"; do case "$n" in prc-ctx4a|prc-ctx4b|prc-ctx4c|prc-ctx4d) ;; *) echo "refusing node $n"; exit 2 ;; esac; done
head=${HEAD:-prc-ctx4a}  # prc-ctx4c (us-west-2) when us-east-1 has no g6e capacity
R=${ROUND:-runs/ctx_own2}
S=${STAGE:-design}
yaml_for() {
  if [ "$1" = "$head" ]; then case "$1" in prc-ctx4a) echo infra/aws/sky_ctx4_head_use1.yaml ;; prc-ctx4c) echo infra/aws/sky_ctx4_head_usw2.yaml ;; *) echo "no head yaml for $1" >&2; exit 2 ;; esac; return; fi
  case "$1" in prc-ctx4a|prc-ctx4b) echo infra/aws/sky_ctx4_worker_use1.yaml ;; *) echo infra/aws/sky_ctx4_worker_usw2.yaml ;; esac
}
case "${1:-}" in
  up)
    source ~/.venv-sky/bin/activate
    mkdir -p "$R/logs"
    for n in "${nodes[@]}"; do
      ( y=$(yaml_for "$n"); end=$(( $(date +%s) + ${RETRY_MIN:-0} * 60 )); k=0
        while true; do
          k=$((k + 1))
          if sky launch -c "$n" "$y" -i 45 --down -y > "$R/logs/launch_${n}_try${k}.log" 2>&1; then echo "$n up ($(date -u +%T), try $k)"; break; fi
          if grep -q InsufficientInstanceCapacity "$R/logs/launch_${n}_try${k}.log" && [ "$(date +%s)" -lt "$end" ]; then sleep 180; continue; fi
          echo "$n launch FAILED ($(date -u +%T), try $k; see $R/logs/launch_${n}_try${k}.log)"; break
        done ) &
    done
    wait ;;
  fanout)
    eval "$(ssh-agent -s)" > /dev/null
    trap 'ssh-agent -k > /dev/null' EXIT
    ssh-add -q ~/.sky/generated/ssh-keys/${head}.key
    head_ip=$(awk '/HostName/{print $2}' ~/.sky/generated/ssh/${head})
    opts="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    for w in "${nodes[@]}"; do
      [ "$w" = "$head" ] && continue
      ( ssh -A "$w" "cd ~/prc && mkdir -p data runs/adsb_nn/features && rsync -a -e '$opts' ubuntu@$head_ip:prc/data/ data/ && rsync -a -e '$opts' ubuntu@$head_ip:prc/runs/adsb_nn/features/ runs/adsb_nn/features/ && echo copied" \
          && echo "$w copy ok" || echo "$w copy FAILED" ) &
    done
    wait
    for n in "${nodes[@]}"; do
      ( ssh "$n" "cd ~/prc && .venv/bin/python -X utf8 tools/ctx3_check_tokens.py | head -c 120 && echo && .venv/bin/python -X utf8 tools/ctx_own2_check_features.py && .venv/bin/python -m pytest tests/test_ctx.py tests/test_ctx2.py tests/test_ctx3.py -q -p no:cacheprovider 2>&1 | tail -2" \
          | sed "s/^/$n: /" ) &
    done
    wait ;;
  sync)
    for n in "${nodes[@]}"; do
      ( rsync -a --delete -e ssh src/ "$n":prc/src/ --exclude __pycache__ && rsync -a -e ssh tools/ "$n":prc/tools/ --exclude __pycache__ \
        && rsync -a -e ssh configs/ "$n":prc/configs/ && rsync -a --delete -e ssh tests/ "$n":prc/tests/ --exclude __pycache__ \
        && ssh "$n" "mkdir -p ~/prc/docs/dig ~/prc/$R/jobs_$S" && rsync -a -e ssh docs/dig/ctx_ownpoints_features_manifest.json "$n":prc/docs/dig/ \
        && rsync -a --delete -e ssh "$R/jobs_$S/" "$n":prc/$R/jobs_$S/ && echo "$n synced" ) &
    done
    wait ;;
  launch)
    source ~/.venv-sky/bin/activate
    for i in "${!nodes[@]}"; do
      n=${nodes[$i]}
      if ssh "$n" "pgrep -f '[c]tx_own2_queue.sh' > /dev/null"; then echo "$n: queue already running"; continue; fi
      sky exec "$n" --gpus L40S:1 -d --name "own2-$S-$i" "cd ~/prc && ROUND=$R STAGE=$S bash tools/ctx_own2_queue.sh $i 2" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Job (ID|submitted)|Error' | sed "s/^/$n: /"
    done ;;
  status)
    for n in "${nodes[@]}"; do
      ssh -o ConnectTimeout=15 "$n" "cd ~/prc/$R && echo done \$(find $S -name run.json 2>/dev/null | wc -l) failed \$(grep -c '^FAILED' logs_$S/chain.log 2>/dev/null) running \$(pgrep -fc '[r]un_adsb_ctx') last: \$(tail -1 logs_$S/chain.log 2>/dev/null)" 2>&1 | sed "s/^/$n: /" || echo "$n: unreachable"
    done ;;
  mirror)
    mkdir -p "$R/$S" "$R/node_logs_$S"
    for n in "${nodes[@]}"; do
      rsync -a -e ssh "$n":prc/$R/$S/ "$R/$S/" && mkdir -p "$R/node_logs_$S/$n" \
        && rsync -a -e ssh "$n":prc/$R/logs_$S/ "$R/node_logs_$S/$n/" && echo "$n mirrored"
    done ;;
  down)
    source ~/.venv-sky/bin/activate
    sky down "${nodes[@]}" -y ;;
  *) echo "usage: NODES=\"...\" $0 up|fanout|sync|launch|status|mirror|down"; exit 2 ;;
esac
