#!/usr/bin/env bash
# Masked LIRF tree design (configs/lirf_tree_mask_design_v1.json): 15 seasonal-fold fits on arcw0-arcw3, 8 threads each, two lanes per worker.
# Per worker: the shared ADS-B feature cache is built first (16 threads), then lane A (one CatBoost) and lane B (a CatBoost or a LightGBM chain).
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
O=runs/lirf_tree_mask/design
for n in 0 1 2 3; do
  case $n in
    0) A="cat_ctr2_s34_dark50"; B="cat_ctr2_s34_ctrl";;
    1) A="cat_ctr3_s35_dark50"; B="lgb_s31_ctrl lgb_s31_focal25 lgb_s31_focal50 lgb_s31_dark50";;
    2) A="cat_ctr2_nm_s36_dark50"; B="lgb_cap_s32_focal25 lgb_cap_s32_focal50 lgb_cap_s32_dark50";;
    3) A="cat_ctr3_nm_s37_dark50"; B="lgb_nm_s33_focal25 lgb_nm_s33_focal50 lgb_nm_s33_dark50";;
  esac
  timeout 60 ssh -o BatchMode=yes arcw$n "mkdir -p $R/$O/logs"
  scp -q tools/run_lirf_tree_mask.py arcw$n:$R/tools/
  scp -q configs/lirf_tree_mask_design_v1.json arcw$n:$R/configs/
  {
    echo "cd $R"
    echo ".venv/bin/python -u tools/run_lirf_tree_mask.py --build-cache-only --threads 16 > $O/logs/cache_$n.log 2>&1 || { echo cache failed > $O/logs/failures_$n; exit 1; }"
    for lane in A B; do
      runs=${!lane}
      echo "("
      for r in $runs; do
        echo "  .venv/bin/python -u tools/run_lirf_tree_mask.py --run $r --threads 8 > $O/logs/$r.log 2>&1 || echo failed $r >> $O/logs/failures_$n"
      done
      echo "  echo done > $O/logs/lane_${lane}_done_$n"
      echo ") &"
    done
    echo "wait"
  } > /tmp/ltm_chain_$n.sh
  scp -q /tmp/ltm_chain_$n.sh arcw$n:$R/$O/chain_$n.sh
  timeout 75 ssh -o BatchMode=yes arcw$n "cd $R && (setsid nohup bash $O/chain_$n.sh > /dev/null 2>&1 < /dev/null &); echo arcw$n: A=$A B=$B" || true
done
