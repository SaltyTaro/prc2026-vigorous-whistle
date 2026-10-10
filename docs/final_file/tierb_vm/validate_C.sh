#!/bin/bash
# arcw6: lane C's code-path stand-in (never read), to compare its output hashes with the workstation's run (logs/standin2.log).
R=/home/azureuser/prc-flight-2026; cd $R; V=runs/febjun_tierb/vm; PY=".venv/bin/python -X utf8 -B tools/febjun_tierb_chain.py"
B=runs/febjun_tierb/chain
[ -f $B/root_standin/input.json ] || bash $V/vmrun.sh vC_root $PY root --standin || exit 1
[ -f $B/work_standin/steps/model_part/step1.json ] || bash $V/vmrun.sh vC_c1 $PY c1 --standin || exit 2
mkdir -p $B/work_standin/steps; [ -d $B/work_standin/steps/rematch ] || cp -a $B/work/steps/rematch $B/work_standin/steps/rematch
[ -f $B/work_standin/chain_tierb.json ] || bash $V/vmrun.sh vC_early $PY early --standin || exit 3
bash $V/vmrun.sh vC_late $PY late --standin || exit 4
echo VALIDATE_C_DONE >> $V/logs/vC_late.log
