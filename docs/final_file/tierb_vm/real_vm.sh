#!/bin/bash
# arcw6, detached: the real Tier B run (lane M then lane C), every step under vmrun.sh (strace). Log runs/febjun_tierb/vm/logs/real_vm.log.
# Starts lane M once runs/febjun_tierb/vm/TREES_READY exists (written by the workstation relay after the 58 tree fits are uploaded and
# hash-checked). Ends with REAL_DONE, or STEP_FAILED <step> / TREE_CHECK_FAILED.
R=/home/azureuser/prc-flight-2026; cd $R
V=runs/febjun_tierb/vm; L=$V/logs; LOG=$L/real_vm.log; mkdir -p $L
PY=".venv/bin/python -X utf8 -B"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
step() {  # step NAME TOOL ACTION [args]
  if bash $V/vmrun.sh "$name" $PY "$@"; then say "STEP_DONE $name ($(tail -1 $L/$name.log))"
  else say "STEP_FAILED $name"; tail -30 $L/$name.log >> $LOG; exit 1; fi
}
say "start (pid $$)"
[ -f runs/febjun_tierb/chain/own/member_build.json ] || step real_own_member tools/febjun_tierb_chain.py own-member
n=0
until [ -f $V/TREES_READY ]; do [ $((n % 30)) -eq 0 ] && say "waiting for TREES_READY"; n=$((n + 1)); sleep 60; done
$PY -c "
import json, os, sys
plan = json.load(open('runs/febjun_lomo/plan.json'))
jobs = [j for h in plan['chains'].values() for j in h]
bad = []
for j in jobs:
    p = os.path.join(j['output'], 'lomo_' + j['month'], 'run.json')
    if not os.path.exists(p) or not os.path.exists(p.replace('run.json', 'predictions.parquet')):
        bad.append(('missing', j['name'])); continue
    r = json.load(open(p))
    if r['held_month'] != j['month'] or r['held_month'] in r['fit_months'] or not r['held_rows_equal_reference'] or r['declaration'] != 'configs/febjun_tierb_v1.json' or r['smoke']:
        bad.append(('record', j['name']))
print('TREE_CHECK jobs', len(jobs), 'bad', bad)
sys.exit(1 if bad or len(jobs) != 58 else 0)
" >> $LOG 2>&1 || { say "TREE_CHECK_FAILED"; exit 4; }
for s in trees rows chain read; do
  [ "$s" = read ] && [ -f docs/final_file/tierb_model_results.json ] && { say "skip M read (exists)"; continue; }
  step real_M_$s tools/febjun_tierb_model.py $s
done
[ -f runs/febjun_tierb/chain/root/input.json ] || step real_C_root tools/febjun_tierb_chain.py root
[ -f runs/febjun_tierb/chain/work/steps/model_part/step1.json ] || step real_C_c1 tools/febjun_tierb_chain.py c1
[ -f runs/febjun_tierb/chain/work/chain_tierb.json ] || step real_C_early tools/febjun_tierb_chain.py early
step real_C_late tools/febjun_tierb_chain.py late
step real_C_read tools/febjun_tierb_chain.py read
say "REAL_DONE"
