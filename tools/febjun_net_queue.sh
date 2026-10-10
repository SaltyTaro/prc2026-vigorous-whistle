#!/usr/bin/env bash
# Run one node's Tier B network job list (runs/febjun_nets/jobs/node<N>.txt from tools/febjun_net_jobs.py; lines: config variant seed fold output
# extra) LANES at a time on this node's GPU via tools/run_adsb_ctx_lomo.py, in file order; finished jobs (run.json present) are skipped.
# tools/combo_queue.sh (unchanged) with the --extra-features directory taken per job (6th column, '-' = none) instead of per queue.
# Logs runs/febjun_nets/logs/<output basename>_<fold>.log; marker lines in runs/febjun_nets/logs/chain.log. No held-month RMSE is computed.
# Usage: tools/febjun_net_queue.sh NODE [LANES]
set -u
cd ~/prc
node=$1; lanes=${2:-2}
export ROUND=runs/febjun_nets
mkdir -p $ROUND/logs
jobs=$ROUND/jobs/node${node}.txt
echo "queue start $(date -u +%FT%TZ) node $node lanes $lanes jobs $(wc -l < "$jobs")" >> $ROUND/logs/chain.log
run_one() {
  config=$1; variant=$2; seed=$3; fold=$4; output=$5; extra=$6
  tag=$(basename "$output")_${fold}
  [ -f "$output/$fold/run.json" ] && { echo "skip $tag" >> $ROUND/logs/chain.log; return 0; }
  [ "$extra" = "-" ] && extra=""
  echo "start $tag $(date -u +%FT%TZ)" >> $ROUND/logs/chain.log
  if .venv/bin/python -X utf8 tools/run_adsb_ctx_lomo.py --config "$config" --variant "$variant" --seed "$seed" --fold "$fold" --output "$output" \
       --device cuda --threads 3 ${extra:+--extra-features "$extra"} > $ROUND/logs/${tag}.log 2>&1; then
    echo "done $tag $(date -u +%FT%TZ)" >> $ROUND/logs/chain.log
  else
    echo "FAILED $tag $(date -u +%FT%TZ)" >> $ROUND/logs/chain.log
  fi
}
export -f run_one
xargs -P "$lanes" -L 1 bash -c 'run_one "$0" "$1" "$2" "$3" "$4" "$5"' < "$jobs"
echo "queue done $(date -u +%FT%TZ) node $node" >> $ROUND/logs/chain.log
