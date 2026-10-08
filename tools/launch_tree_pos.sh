#!/usr/bin/env bash
# Tree position screen (configs/tree_pos_screen_v1.json): 18 LightGBM fits on arcw4-arcw7, one at a time per worker, 16 threads.
# Reuses each worker's ADS-B feature cache from the tree_own screen (runs/tree_own/fits/_features, symlinked) when present.
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
O=runs/tree_pos/fits
LEAVES="runs/adsb_v20/lgb_x19_s24 runs/adsb_v20/lgb_x19_s25 runs/adsb_v20/lgb_x19_s26"
WORKERS=${*:-"4 5 6 7"}
W=($WORKERS); NW=${#W[@]}; i=0
declare -A JOBS
for f in refit12 lomo_2025-01 lomo_2025-07 forward jan_nov; do for l in $LEAVES; do
  w=${W[$((i % NW))]}; JOBS[$w]="${JOBS[$w]:-} $l,pos,$f"; i=$((i + 1))
done; done
for l in $LEAVES; do w=${W[$((i % NW))]}; JOBS[$w]="${JOBS[$w]:-} $l,ctl,refit12"; i=$((i + 1)); done
for n in $WORKERS; do
  h=arcw$n
  timeout 60 ssh -o BatchMode=yes $h "mkdir -p $R/$O/logs; [ -d $R/runs/tree_own/fits/_features ] && [ ! -e $R/$O/_features ] && ln -s $R/runs/tree_own/fits/_features $R/$O/_features; true"
  scp -q tools/run_tree_pos.py $h:$R/tools/
  scp -q configs/tree_pos_screen_v1.json $h:$R/configs/
  for f in data/external/adsblol/matches/matches2_*.parquet; do
    lh=$(sha256sum "$f" | cut -d' ' -f1)
    rh=$(timeout 60 ssh -o BatchMode=yes $h "sha256sum $R/$f 2>/dev/null | cut -d' ' -f1" || true)
    [ "$lh" = "$rh" ] || scp -q "$f" $h:$R/$f
  done
  {
    echo "cd $R"
    echo "[ -e $O/_features ] || .venv/bin/python -u tools/run_tree_pos.py --build-cache-only --threads 16 > $O/logs/cache_$n.log 2>&1 || { echo cache failed > $O/logs/failures_$n; exit 1; }"
    for j in ${JOBS[$n]}; do
      IFS=, read -r l v f <<< "$j"
      tag=$(echo "$l" | sed 's|runs/||; s|/|__|g')_${v}_${f}
      echo ".venv/bin/python -u tools/run_tree_pos.py --leaf $l --variant $v --fold $f --threads 16 > $O/logs/$tag.log 2>&1 || echo failed $tag >> $O/logs/failures_$n"
    done
    echo "echo done > $O/logs/done_$n"
  } > /tmp/tree_pos_chain_$n.sh
  scp -q /tmp/tree_pos_chain_$n.sh $h:$R/$O/chain_$n.sh
  timeout 75 ssh -o BatchMode=yes $h "cd $R && (setsid nohup bash $O/chain_$n.sh > /dev/null 2>&1 < /dev/null &); echo $h: \$(echo '${JOBS[$n]}' | wc -w) jobs" || true
done
