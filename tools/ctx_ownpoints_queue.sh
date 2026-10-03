#!/usr/bin/env bash
# Dig tunnel 21 (configs/ctx_ownpoints_design_v1.json): run this worker's job list runs/ctx_ownpoints/jobs/<host>.txt (lines: config variant
# seed fold output extra_features_dir) two lanes at a time on CPU via tools/run_adsb_ctx3.py --threads 8; finished jobs (run.json present)
# are skipped. Logs runs/ctx_ownpoints/logs/<output basename>_<fold>.log; markers in runs/ctx_ownpoints/logs/chain.log.
# Usage (on the worker, from the project root): bash tools/ctx_ownpoints_queue.sh HOST [LIST] (LIST defaults to HOST)
set -u
host=$1; list=${2:-$1}
mkdir -p runs/ctx_ownpoints/logs
jobs=runs/ctx_ownpoints/jobs/${list}.txt
echo "queue start $(date -u +%FT%TZ) $host jobs $(wc -l < "$jobs")" >> runs/ctx_ownpoints/logs/chain.log
run_one() {
  config=$1; variant=$2; seed=$3; fold=$4; output=$5; extra=$6
  tag=$(basename "$output")_${fold}
  [ -f "$output/$fold/run.json" ] && { echo "skip $tag" >> runs/ctx_ownpoints/logs/chain.log; return 0; }
  echo "start $tag $(date -u +%FT%TZ)" >> runs/ctx_ownpoints/logs/chain.log
  if .venv/bin/python -X utf8 tools/run_adsb_ctx3.py --config "$config" --variant "$variant" --seed "$seed" --fold "$fold" --output "$output" \
       --extra-features "$extra" --device cpu --threads 8 > runs/ctx_ownpoints/logs/${tag}.log 2>&1; then
    echo "done $tag $(date -u +%FT%TZ)" >> runs/ctx_ownpoints/logs/chain.log
  else
    echo "FAILED $tag $(date -u +%FT%TZ)" >> runs/ctx_ownpoints/logs/chain.log
  fi
}
export -f run_one
xargs -P 2 -L 1 bash -c 'run_one "$0" "$1" "$2" "$3" "$4" "$5"' < "$jobs"
echo "queue done $(date -u +%FT%TZ) $host" >> runs/ctx_ownpoints/logs/chain.log
