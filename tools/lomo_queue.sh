#!/usr/bin/env bash
# Run one node's LOMO job list (runs/lomo_dev/jobs/node<N>.txt from tools/lomo_jobs.py; lines: config variant seed fold output) LANES at a
# time on this node's GPU via tools/run_adsb_ctx_lomo.py, in file order; finished jobs (run.json present) are skipped. Logs
# runs/lomo_dev/logs/<output basename>_<fold>.log; marker lines in runs/lomo_dev/logs/chain.log. Usage: tools/lomo_queue.sh NODE [LANES]
set -u
cd ~/prc
node=$1; lanes=${2:-2}
mkdir -p runs/lomo_dev/logs
jobs=runs/lomo_dev/jobs/node${node}.txt
echo "queue start $(date -u +%FT%TZ) node $node lanes $lanes jobs $(wc -l < "$jobs")" >> runs/lomo_dev/logs/chain.log
run_one() {
  config=$1; variant=$2; seed=$3; fold=$4; output=$5
  tag=$(basename "$output")_${fold}
  [ -f "$output/$fold/run.json" ] && { echo "skip $tag" >> runs/lomo_dev/logs/chain.log; return 0; }
  echo "start $tag $(date -u +%FT%TZ)" >> runs/lomo_dev/logs/chain.log
  if .venv/bin/python -X utf8 tools/run_adsb_ctx_lomo.py --config "$config" --variant "$variant" --seed "$seed" --fold "$fold" --output "$output" \
       --device cuda --threads 3 > runs/lomo_dev/logs/${tag}.log 2>&1; then
    echo "done $tag $(date -u +%FT%TZ)" >> runs/lomo_dev/logs/chain.log
  else
    echo "FAILED $tag $(date -u +%FT%TZ)" >> runs/lomo_dev/logs/chain.log
  fi
}
export -f run_one
xargs -P "$lanes" -L 1 bash -c 'run_one "$0" "$1" "$2" "$3" "$4"' < "$jobs"
echo "queue done $(date -u +%FT%TZ) node $node" >> runs/lomo_dev/logs/chain.log
