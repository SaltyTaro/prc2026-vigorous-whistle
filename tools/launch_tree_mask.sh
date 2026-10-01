#!/usr/bin/env bash
# Masked-tree design (configs/tree_mask_design_v1.json): 24 October design fits on arcw0-arcw7, 16 threads each, sequential per worker.
# arcw0-3: lgb_m25 seed n+1, lgb_m50 seed n+1, then cat_ref s3 / cat_ref s4 / cat_m25 s3 / cat_m25 s4; arcw4-7: cat_ref s1, cat_ref s2, cat_m25 s1, cat_m25 s2.
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
for n in 0 1 2 3 4 5 6 7; do
  case $n in
    0) chain="lgb_m25:1 lgb_m50:1 cat_ref:3";; 1) chain="lgb_m25:2 lgb_m50:2 cat_ref:4";;
    2) chain="lgb_m25:3 lgb_m50:3 cat_m25:3";; 3) chain="lgb_m25:4 lgb_m50:4 cat_m25:4";;
    4) chain="cat_ref:1";; 5) chain="cat_ref:2";; 6) chain="cat_m25:1";; 7) chain="cat_m25:2";;
  esac
  timeout 60 ssh -o BatchMode=yes arcw$n "mkdir -p $R/src/prc2026_tree_mask $R/runs/tree_mask/logs"
  scp -q tools/run_tree_mask_design.py arcw$n:$R/tools/
  scp -q src/prc2026_tree_mask/__init__.py arcw$n:$R/src/prc2026_tree_mask/
  scp -q configs/tree_mask_design_v1.json arcw$n:$R/configs/
  {
    echo "cd $R"
    for item in $chain; do v=${item%%:*}; s=${item##*:}
      echo ".venv/bin/python -u tools/run_tree_mask_design.py --variant $v --seed $s --threads 16 > runs/tree_mask/logs/${v}_s$s.log 2>&1 || echo failed $v $s >> runs/tree_mask/logs/failures_$n"
    done
    echo "echo done > runs/tree_mask/logs/chain_done_$n"
  } > /tmp/tm_chain_$n.sh
  scp -q /tmp/tm_chain_$n.sh arcw$n:$R/runs_chain_tm.sh
  timeout 60 ssh -o BatchMode=yes arcw$n "cd $R && (setsid nohup bash runs_chain_tm.sh > /dev/null 2>&1 < /dev/null &); echo arcw$n: $chain"
done
