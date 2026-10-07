#!/usr/bin/env bash
# Fork V (configs/var_alloc_bag_v1.json): run steps of tools/run_var_alloc_bag.py on an Azure VM, in order, stopping at the first failure.
# Workspace /mnt/scratch/prc (temp disk; a copy of ~/prc-v67 with the current tools/src/configs). Missing inputs arrive lazily from the
# workstation's read-only file server (tools/var_alloc_bag_fileserver.py, port 8767) through an SSH reverse tunnel
# (tools/ctx_own2_vmfetch/sitecustomize.py); fetched files are hash-listed in runs/var_alloc_bag/vm_fetch.jsonl.
# Markers runs/var_alloc_bag/vm/<step>.ok|.failed; logs runs/var_alloc_bag/vm/<step>.log.
# Usage (on the VM): setsid nohup bash tools/var_alloc_bag_vmdriver.sh LANE STEP [STEP ...] > /dev/null 2>&1 < /dev/null &
set -u
W=/mnt/scratch/prc
cd "$W"
export PRC_ROOT="$W" PRC_FETCH_URL=http://127.0.0.1:8767 PRC_FETCH_LOG="$W/runs/var_alloc_bag/vm_fetch.jsonl"
export PYTHONPATH="$W/tools/ctx_own2_vmfetch"
export OMP_NUM_THREADS=4
mkdir -p runs/var_alloc_bag/vm
lane=$1; shift
log=runs/var_alloc_bag/vm/driver_$lane.log
PY=(.venv/bin/python -X utf8 -B tools/run_var_alloc_bag.py)
for step in "$@"; do
  if [ -f runs/var_alloc_bag/vm/$step.ok ]; then echo "$(date -u +%FT%TZ) skip $step (ok)" >> $log; continue; fi
  echo "$(date -u +%FT%TZ) start $step" >> $log
  case "$step" in
    lay_ctl_*|lay_cand_*) k=${step#lay_}; args=(--dev-layers "${k%%_*}" "${k#*_}") ;;
    chain) args=(--dev-chain) ;;
    stack_*) args=(--dev-stack "${step#stack_}") ;;
    score) args=(--dev-score) ;;
    ranky) args=(--rank-y) ;;
    rident) args=(--rank-final s1 69) ;;
    decide) args=(--decide) ;;
    rdry) args=(--rank-final b10 70) ;;
    *) echo "unknown step $step" >> $log; exit 2 ;;
  esac
  if "${PY[@]}" "${args[@]}" > runs/var_alloc_bag/vm/$step.log 2>&1; then
    echo "$(date -u +%FT%TZ) ok $step" >> $log; touch runs/var_alloc_bag/vm/$step.ok
  else
    echo "$(date -u +%FT%TZ) FAILED $step" >> $log; touch runs/var_alloc_bag/vm/$step.failed; exit 1
  fi
done
echo "$(date -u +%FT%TZ) lane $lane done: $*" >> $log
