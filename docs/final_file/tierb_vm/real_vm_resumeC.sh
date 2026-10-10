#!/bin/bash
# arcw6, detached: resume the real run's lane C after the c1 fix (03c476e); lane M and root are done and kept. Appends to real_vm.log.
R=/home/azureuser/prc-flight-2026; cd $R
V=runs/febjun_tierb/vm; L=$V/logs; LOG=$L/real_vm.log
PY=".venv/bin/python -X utf8 -B"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
step() {
  if bash $V/vmrun.sh "$name" $PY "$@"; then say "STEP_DONE $name ($(tail -1 $L/$name.log))"
  else say "STEP_FAILED $name"; tail -30 $L/$name.log >> $LOG; exit 1; fi
}
say "resume lane C (pid $$)"
[ -d $L/real_C_c1_failed1 ] || { mkdir -p $L/real_C_c1_failed1; mv $L/real_C_c1.* $L/real_C_c1_failed1/ 2>/dev/null; }
[ -f runs/febjun_tierb/chain/work/steps/model_part/step1.json ] || step real_C_c1 tools/febjun_tierb_chain.py c1
[ -f runs/febjun_tierb/chain/work/chain_tierb.json ] || step real_C_early tools/febjun_tierb_chain.py early
step real_C_late tools/febjun_tierb_chain.py late
step real_C_read tools/febjun_tierb_chain.py read
say "REAL_DONE"
