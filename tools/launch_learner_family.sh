#!/usr/bin/env bash
# Learner-family design (configs/learner_family_design_v1.json): 16 October design fits on arcw0-arcw7, two sequential fits per worker, 16 threads.
# arcw0-3: ref_lgb seed s then dart seed s (dart rounds = that control's best iteration); arcw4-7: linear_tree then huber_tail, seed s = n-3.
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
for n in 0 1 2 3 4 5 6 7; do
  if [ $n -lt 4 ]; then s=$((n+1)); first="ref_lgb"; second="dart"; else s=$((n-3)); first="linear_tree"; second="huber_tail"; fi
  case $first in ref_lgb) p1="";; linear_tree) p1="--param linear_tree=true --param linear_lambda=1.0";; esac
  case $second in dart) p2='--param boosting_type=dart --param drop_rate=0.1 --param skip_drop=0.5 --rounds $(.venv/bin/python -c "import json;print(json.load(open(\"runs/learner_family/design/ref_lgb_s'$s'/run.json\"))[\"best_iteration\"])")';; huber_tail) p2="--param objective=huber --param alpha=600";; esac
  scp -q tools/run_learner_family_design.py arcw$n:$R/tools/
  cat > /tmp/chain_$n.sh <<EOF
cd $R && mkdir -p runs/learner_family/logs
.venv/bin/python -u tools/run_learner_family_design.py --variant $first --seed $s --threads 16 $p1 > runs/learner_family/logs/${first}_s$s.log 2>&1
.venv/bin/python -u tools/run_learner_family_design.py --variant $second --seed $s --threads 16 $p2 > runs/learner_family/logs/${second}_s$s.log 2>&1
echo done > runs/learner_family/logs/chain_done_$n
EOF
  scp -q /tmp/chain_$n.sh arcw$n:$R/runs_chain_lf.sh
  timeout 60 ssh -o BatchMode=yes arcw$n "cd $R && (setsid nohup bash runs_chain_lf.sh > /dev/null 2>&1 < /dev/null &); echo arcw$n $first/$second s$s launched"
done
