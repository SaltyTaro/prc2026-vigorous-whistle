#!/usr/bin/env bash
# Lever 12b (fork N): run steps of tools/run_lfpg_net_dev.py on an Azure VM (user: "use the azure vm's please keep as much compute off this
# workstation"), in order, stopping at the first failure. Workspace /mnt/scratch/prc (the VM's temp disk; a copy of arcw1's ~/prc-v67 with the
# current tools/src/configs). Inputs arrive lazily from the workstation's read-only file server (tools/lfpg_net_fileserver.py, port 8766) through
# an SSH reverse tunnel (tools/ctx_own2_vmfetch/sitecustomize.py); every fetched file is hash-listed in runs/lfpg_net/vm_fetch.jsonl.
# Markers runs/lfpg_net/vm/<step>.ok|.failed; logs runs/lfpg_net/vm/<step>.log; lane log runs/lfpg_net/vm/driver_<lane>.log.
# Usage (on the VM): setsid nohup bash tools/lfpg_net_vmdriver.sh LANE STEP [STEP ...] > /dev/null 2>&1 < /dev/null &
set -u
W=/mnt/scratch/prc
cd "$W"
export PRC_ROOT="$W" PRC_FETCH_URL=http://127.0.0.1:8766 PRC_FETCH_LOG="$W/runs/lfpg_net/vm_fetch.jsonl"
export PYTHONPATH="$W/tools/ctx_own2_vmfetch"
export OMP_NUM_THREADS=4
mkdir -p runs/lfpg_net/vm
lane=$1; shift
log=runs/lfpg_net/vm/driver_$lane.log
PY=(.venv/bin/python -X utf8 -B tools/run_lfpg_net_dev.py)
for step in "$@"; do
  if [ -f runs/lfpg_net/vm/$step.ok ]; then echo "$(date -u +%FT%TZ) skip $step (ok)" >> $log; continue; fi
  echo "$(date -u +%FT%TZ) start $step" >> $log
  case "$step" in
    chain_own) args=(--chain own_all_w100) ;;
    chain_cand) args=(--chain lfpg_mix_w100) ;;
    stage1) args=(--stage1) ;;
    devm_2025-01|devm_2025-07|devm_2025-11|devm_2025-12) args=(--dev-month lfpg_mix_w100 "${step#devm_}") ;;
    devmown_2025-01|devmown_2025-07|devmown_2025-11|devmown_2025-12) args=(--dev-month own_all_w100 "${step#devmown_}") ;;
    score) args=(--score) ;;
    checkctl) args=(--check-control) ;;
    rsteps_unpinned_own) args=(--rank-steps own_all_w100) ;;
    rsteps_pinned_own) args=(--rank-steps own_all_w100 --pinned) ;;
    rsteps_pinned_cand) args=(--rank-steps lfpg_mix_w100 --pinned) ;;
    rident) args=(--rank-identity-final) ;;
    rfinal_own) args=(--rank-final own_all_w100) ;;
    rfinal_cand) args=(--rank-final lfpg_mix_w100) ;;
    rdry) args=(--rank-dry) ;;
    *) echo "unknown step $step" >> $log; exit 2 ;;
  esac
  if "${PY[@]}" "${args[@]}" > runs/lfpg_net/vm/$step.log 2>&1; then
    echo "$(date -u +%FT%TZ) ok $step" >> $log; touch runs/lfpg_net/vm/$step.ok
  else
    echo "$(date -u +%FT%TZ) FAILED $step" >> $log; touch runs/lfpg_net/vm/$step.failed; exit 1
  fi
done
echo "$(date -u +%FT%TZ) lane $lane done: $*" >> $log
