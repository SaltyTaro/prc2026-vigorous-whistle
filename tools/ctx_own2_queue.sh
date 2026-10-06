#!/usr/bin/env bash
# Lever 5 (fork E): run one node's job list $ROUND/jobs_$STAGE/node<N>.txt LANES at a time on this node's GPU, in file order; finished jobs
# (run.json present) are skipped. Lines: runner config variant seed fold output extra_features_dir, where runner is ctx3 (tools/run_adsb_ctx3.py,
# design fold) or lomo (tools/run_adsb_ctx_lomo.py, development folds). Logs $ROUND/logs_$STAGE/<output basename>_<fold>.log; markers in
# $ROUND/logs_$STAGE/chain.log. A copy of tools/combo_queue.sh / tools/ctx_ownpoints_queue.sh with the runner named on each line.
# Usage (on the node, from ~/prc): ROUND=runs/ctx_own2 STAGE=design bash tools/ctx_own2_queue.sh NODE [LANES]
set -u
cd ~/prc
node=$1; lanes=${2:-2}
export ROUND=${ROUND:-runs/ctx_own2}
export STAGE=${STAGE:-design}
mkdir -p "$ROUND/logs_$STAGE"
jobs=$ROUND/jobs_$STAGE/node${node}.txt
echo "queue start $(date -u +%FT%TZ) node $node lanes $lanes jobs $(wc -l < "$jobs")" >> "$ROUND/logs_$STAGE/chain.log"
run_one() {
  runner=$1; config=$2; variant=$3; seed=$4; fold=$5; output=$6; extra=$7
  tag=$(basename "$output")_${fold}
  [ -f "$output/$fold/run.json" ] && { echo "skip $tag" >> "$ROUND/logs_$STAGE/chain.log"; return 0; }
  case "$runner" in
    ctx3) script=tools/run_adsb_ctx3.py ;;
    lomo) script=tools/run_adsb_ctx_lomo.py ;;
    *) echo "FAILED $tag bad runner" >> "$ROUND/logs_$STAGE/chain.log"; return 1 ;;
  esac
  echo "start $tag $(date -u +%FT%TZ)" >> "$ROUND/logs_$STAGE/chain.log"
  if .venv/bin/python -X utf8 $script --config "$config" --variant "$variant" --seed "$seed" --fold "$fold" --output "$output" \
       --extra-features "$extra" --device cuda --threads 3 > "$ROUND/logs_$STAGE/${tag}.log" 2>&1; then
    echo "done $tag $(date -u +%FT%TZ)" >> "$ROUND/logs_$STAGE/chain.log"
  else
    echo "FAILED $tag $(date -u +%FT%TZ)" >> "$ROUND/logs_$STAGE/chain.log"
  fi
}
export -f run_one
xargs -P "$lanes" -L 1 bash -c 'run_one "$0" "$1" "$2" "$3" "$4" "$5" "$6"' < "$jobs"
echo "queue done $(date -u +%FT%TZ) node $node" >> "$ROUND/logs_$STAGE/chain.log"
