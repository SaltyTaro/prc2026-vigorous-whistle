#!/bin/bash
# arcw6: lane M's January / July reproduction checks (exact reproductions on the workstation; outputs runs/febjun_tierb/model/checks/*.json).
R=/home/azureuser/prc-flight-2026; cd $R; V=runs/febjun_tierb/vm; PY=".venv/bin/python -X utf8 -B tools/febjun_tierb_model.py"
for s in ${@:-trees nets rows rematch chain}; do bash $V/vmrun.sh vM_$s $PY $s --check || { echo "VALIDATE_M_FAILED $s" >> $V/logs/vM_$s.log; exit 1; }; done
echo VALIDATE_M_DONE >> $V/logs/vM_chain.log
