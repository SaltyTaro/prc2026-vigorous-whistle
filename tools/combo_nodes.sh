#!/usr/bin/env bash
# Node operations of the combination's development round, run in WSL from the project root (SkyPilot SSH aliases; head = prc-ctx3a).
# tools/ctx3_nodes.sh with the node list taken from NODES (default: the four nodes), because AWS may supply fewer than four.
#   fanout   copy the head's data/ (incl. the token build) and runs/adsb_nn/features to the other nodes inside AWS (agent-forwarded SSH),
#            then on every node check the token content hashes and run the context tests
#   sync     push src/ tools/ configs/ tests/ and the job lists from the workstation to every node
#   launch   start tools/combo_queue.sh on every node as a SkyPilot job (node i of the list, 2 lanes); refuses if a queue is running
#   status   one line per node: jobs done / failed / running
#   mirror   copy runs/adsb_ctx3_combo/nets and the node logs back from every node
# ROUND (default runs/adsb_ctx3_combo) names another round's directory (jobs, nets, logs), e.g. ROUND=runs/adsb_ctx3_mask.
# Usage: [ROUND=runs/...] NODES="prc-ctx3a prc-ctx3c" tools/combo_nodes.sh fanout|sync|launch|status|mirror
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a nodes <<< "${NODES:-prc-ctx3a prc-ctx3b prc-ctx3c prc-ctx3d}"
head=prc-ctx3a
R=${ROUND:-runs/adsb_ctx3_combo}
case "${1:-}" in
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
      ( ssh "$n" "cd ~/prc && .venv/bin/python -X utf8 tools/ctx3_check_tokens.py | head -c 200 && echo && .venv/bin/python -m pytest tests/test_ctx.py tests/test_ctx2.py tests/test_ctx3.py -q -p no:cacheprovider 2>&1 | tail -2" \
          | sed "s/^/$n: /" ) &
    done
    wait ;;
  sync)
    for n in "${nodes[@]}"; do
      ( rsync -a --delete -e ssh src/ "$n":prc/src/ --exclude __pycache__ && rsync -a -e ssh tools/ "$n":prc/tools/ --exclude __pycache__ \
        && rsync -a -e ssh configs/ "$n":prc/configs/ && rsync -a --delete -e ssh tests/ "$n":prc/tests/ --exclude __pycache__ \
        && ssh "$n" "mkdir -p ~/prc/$R/jobs" && rsync -a --delete -e ssh $R/jobs/ "$n":prc/$R/jobs/ && echo "$n synced" ) &
    done
    wait ;;
  launch)
    source ~/.venv-sky/bin/activate
    for i in "${!nodes[@]}"; do
      n=${nodes[$i]}
      if ssh "$n" "pgrep -f '[c]ombo_queue.sh' > /dev/null"; then echo "$n: queue already running"; continue; fi
      sky exec "$n" --gpus L40S:1 -d --name "combo-queue-$i" "cd ~/prc && ROUND=$R bash tools/combo_queue.sh $i 2" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Job (ID|submitted)|Error' | sed "s/^/$n: /"
    done ;;
  status)
    for n in "${nodes[@]}"; do
      ssh -o ConnectTimeout=15 "$n" "cd ~/prc/$R && echo done \$(find nets -name run.json 2>/dev/null | wc -l) failed \$(grep -c '^FAILED' logs_dev/chain.log 2>/dev/null) running \$(pgrep -fc '[r]un_adsb_ctx_lomo.py') last: \$(tail -1 logs_dev/chain.log 2>/dev/null)" 2>&1 | sed "s/^/$n: /" || echo "$n: unreachable"
    done ;;
  mirror)
    mkdir -p $R/nets $R/node_logs
    for n in "${nodes[@]}"; do
      rsync -a -e ssh "$n":prc/$R/nets/ $R/nets/ && mkdir -p $R/node_logs/$n \
        && rsync -a -e ssh "$n":prc/$R/logs_dev/ $R/node_logs/$n/ && echo "$n mirrored"
    done ;;
  *) echo "usage: NODES=\"...\" $0 fanout|sync|launch|status|mirror"; exit 2 ;;
esac
