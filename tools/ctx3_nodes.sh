#!/usr/bin/env bash
# GPU sprint node operations, run in WSL from the project root (SkyPilot SSH aliases prc-ctx3a..d; head = a).
#   fanout   copy the head's data/ (incl. the token build) and runs/adsb_nn/features to the workers inside AWS (agent-forwarded SSH;
#            nothing passes through the workstation), then on every node check the token content hashes and run the context tests
#   sync     push src/ tools/ configs/ tests/ from the workstation to every node (after the declaration commit)
#   launch   start tools/ctx3_design_chain.sh on every node (node i of 4, 2 lanes), detached; refuses if a chain is already running
#   mirror   copy runs/adsb_ctx3/ back from every node into the local runs/adsb_ctx3/
# Usage: tools/ctx3_nodes.sh fanout|sync|launch|mirror
set -euo pipefail
cd "$(dirname "$0")/.."
nodes=(prc-ctx3a prc-ctx3b prc-ctx3c prc-ctx3d)
head=${nodes[0]}
case "${1:-}" in
  fanout)
    eval "$(ssh-agent -s)" > /dev/null
    trap 'ssh-agent -k > /dev/null' EXIT
    ssh-add -q ~/.sky/generated/ssh-keys/${head}.key
    head_ip=$(awk '/HostName/{print $2}' ~/.sky/generated/ssh/${head})
    opts="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    for w in "${nodes[@]:1}"; do
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
      rsync -a --delete -e ssh src/ "$n":prc/src/ --exclude __pycache__ && rsync -a -e ssh tools/ "$n":prc/tools/ --exclude __pycache__ \
        && rsync -a -e ssh configs/ "$n":prc/configs/ && rsync -a --delete -e ssh tests/ "$n":prc/tests/ --exclude __pycache__ && echo "$n synced" &
    done
    wait ;;
  launch)
    # submitted as a SkyPilot job (so the cluster counts as busy for autodown); --gpus is required or CUDA is hidden from the job
    source ~/.venv-sky/bin/activate
    for i in "${!nodes[@]}"; do
      n=${nodes[$i]}
      if ssh "$n" "pgrep -f '[c]tx3_design_chain.sh' > /dev/null"; then echo "$n: chain already running"; continue; fi
      sky exec "$n" --gpus L40S:1 -d --name "ctx3-chain-$i" "cd ~/prc && bash tools/ctx3_design_chain.sh $i 4 2" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Job (ID|submitted)|Error' | sed "s/^/$n: /"
    done ;;
  mirror)
    mkdir -p runs/adsb_ctx3
    for n in "${nodes[@]}"; do
      rsync -a -e ssh "$n":prc/runs/adsb_ctx3/ runs/adsb_ctx3/ --exclude 'logs/jobs.txt' --exclude 'logs/chain.log' --exclude 'logs/nohup_chain.log' && \
        mkdir -p runs/adsb_ctx3/node_logs/$n && rsync -a -e ssh "$n":prc/runs/adsb_ctx3/logs/ runs/adsb_ctx3/node_logs/$n/ && echo "$n mirrored"
    done ;;
  launch2)
    # the evening work (tools/ctx3_jobs.py): 32-epoch development seeds and the second design batch, via tools/ctx3_queue.sh as sky jobs
    source ~/.venv-sky/bin/activate
    for i in "${!nodes[@]}"; do
      n=${nodes[$i]}
      if ssh "$n" "pgrep -f '[c]tx3_queue.sh' > /dev/null"; then echo "$n: queue already running"; continue; fi
      sky exec "$n" --gpus L40S:1 -d --name "ctx3-queue-$i" "cd ~/prc && .venv/bin/python tools/ctx3_jobs.py && bash tools/ctx3_queue.sh $i 2" 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'Job (ID|submitted)|Error' | sed "s/^/$n: /"
    done ;;
  mirror2)
    # development learners and batch-2 design runs to the real E: drive (mounted in WSL at /mnt/edrive; WSL's /mnt/e is NOT E:),
    # December leaves (small) into the local runs/december/learners, node logs to E:
    e=/mnt/edrive/prc/runs
    mountpoint -q /mnt/edrive || { echo "mount E: first: wsl -u root mount -t drvfs E: /mnt/edrive -o metadata,uid=1000,gid=1000"; exit 1; }
    mkdir -p "$e/adsb_ctx_gpu" "$e/adsb_ctx3/design2" "$e/adsb_ctx3/node_logs2" runs/december/learners
    for n in "${nodes[@]}"; do
      rsync -a -e ssh "$n":prc/runs/adsb_ctx_gpu/ "$e/adsb_ctx_gpu/" 2>/dev/null; a=$?
      rsync -a -e ssh "$n":prc/runs/adsb_ctx3/design2/ "$e/adsb_ctx3/design2/" 2>/dev/null; b=$?
      rsync -a -e ssh --include 'runs__adsb_ctx_gpu__*' --exclude '*' "$n":prc/runs/december/learners/ runs/december/learners/ 2>/dev/null; c=$?
      mkdir -p "$e/adsb_ctx3/node_logs2/$n" && rsync -a -e ssh "$n":prc/runs/adsb_ctx3/logs2/ "$e/adsb_ctx3/node_logs2/$n/" 2>/dev/null; d=$?
      echo "$n mirrored (rsync codes $a $b $c $d)"
    done ;;
  *) echo "usage: $0 fanout|sync|launch|mirror|launch2|mirror2"; exit 2 ;;
esac
