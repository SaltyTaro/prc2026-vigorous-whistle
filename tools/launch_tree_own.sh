#!/usr/bin/env bash
# Tree own-points screen (configs/tree_own_screen_v1.json): 48 LightGBM fits on arcw4-arcw7 (fork C's lane), one fit at a time per worker,
# 16 threads. Per worker: code + declaration copied, the own-points tables hash-checked against the local manifest, the ADS-B feature
# cache built once, then its twelve jobs in sequence (round-robin over fold x leaf x variant, so every fold advances on every worker).
# Usage: tools/launch_tree_own.sh [workers...]   (default 4 5 6 7)
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
O=runs/tree_own/fits
LEAVES="runs/adsb_v21/lgb_lirf_s31 runs/adsb_v21/lgb_lirf_cap_s32 runs/adsb_v21/lgb_lirf_nm_s33 runs/adsb_v20/lgb_x19_s24 runs/adsb_v20/lgb_x19_s25 runs/adsb_v20/lgb_x19_s26"
FOLDS="lomo_2025-01 lomo_2025-07 forward jan_nov"
WORKERS=${*:-"4 5 6 7"}
W=($WORKERS)
NW=${#W[@]}
i=0
declare -A JOBS
for f in $FOLDS; do for l in $LEAVES; do for v in ctl own; do
  w=${W[$((i % NW))]}
  JOBS[$w]="${JOBS[$w]:-} $l,$v,$f"
  i=$((i + 1))
done; done; done
for n in $WORKERS; do
  h=arcw$n
  timeout 60 ssh -o BatchMode=yes $h "mkdir -p $R/$O/logs $R/data/external/adsblol/features_v19own"
  scp -q tools/run_tree_own.py $h:$R/tools/
  scp -q configs/tree_own_screen_v1.json $h:$R/configs/
  # the own-points tables: copy any file whose hash differs from the local copy
  for f in data/external/adsblol/features_v19own/*; do
    lh=$(sha256sum "$f" | cut -d' ' -f1)
    rh=$(timeout 60 ssh -o BatchMode=yes $h "sha256sum $R/$f 2>/dev/null | cut -d' ' -f1" || true)
    [ "$lh" = "$rh" ] || scp -q "$f" $h:$R/$f
  done
  {
    echo "cd $R"
    echo ".venv/bin/python -u tools/run_tree_own.py --build-cache-only --threads 16 > $O/logs/cache_$n.log 2>&1 || { echo cache failed > $O/logs/failures_$n; exit 1; }"
    for j in ${JOBS[$n]}; do
      IFS=, read -r l v f <<< "$j"
      tag=$(echo "$l" | sed 's|runs/||; s|/|__|g')_${v}_${f}
      echo ".venv/bin/python -u tools/run_tree_own.py --leaf $l --variant $v --fold $f --threads 16 > $O/logs/$tag.log 2>&1 || echo failed $tag >> $O/logs/failures_$n"
    done
    echo "echo done > $O/logs/done_$n"
  } > /tmp/tree_own_chain_$n.sh
  scp -q /tmp/tree_own_chain_$n.sh $h:$R/$O/chain_$n.sh
  timeout 75 ssh -o BatchMode=yes $h "cd $R && (setsid nohup bash $O/chain_$n.sh > /dev/null 2>&1 < /dev/null &); echo $h: $(echo ${JOBS[$n]} | wc -w) jobs" || true
done
