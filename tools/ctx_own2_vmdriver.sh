#!/usr/bin/env bash
# Lever 5 (fork E), v67 on Azure (coordinator / user: "use the azure vm's please keep as much compute off this workstation"): run the v67 build
# steps on the VM in order, each a separate invocation of tools/export_ctx_own2_override.py (or the audit), stopping at the first failure.
# Inputs arrive lazily from the workstation's read-only file server through the SSH reverse tunnel (tools/ctx_own2_vmfetch/sitecustomize.py);
# every fetched file is listed with its SHA-256 in runs/ctx_own2/vm_fetch.jsonl. Steps: identity (v66 byte for byte), dry (unpinned),
# pinned (dry run with the variance re-allocation pinned), build (rerun, byte identity with the dry run, the VM's copy of
# runs/submissions/vigorous-whistle_v67.parquet), audit. Marker files runs/ctx_own2/vm_<step>.ok / vm_<step>.failed; log
# runs/ctx_own2/logs/vm_driver.log. Usage (on the VM, from ~/prc-v67): nohup bash tools/ctx_own2_vmdriver.sh STEP [STEP ...] &
set -u
cd "$HOME/prc-v67"
export PRC_ROOT="$HOME/prc-v67" PRC_FETCH_URL=http://127.0.0.1:8765 PRC_FETCH_LOG="$HOME/prc-v67/runs/ctx_own2/vm_fetch.jsonl"
export PYTHONPATH="$HOME/prc-v67/tools/ctx_own2_vmfetch"
mkdir -p runs/ctx_own2/logs
log=runs/ctx_own2/logs/vm_driver.log
PY=(.venv/bin/python -X utf8 -B)
DECISION="override it (the user, 6 October 2026 ~17:45 UTC, relayed by the coordinator; docs/ctx_own2/override_v67.json)"
for step in "$@"; do
  echo "$(date -u +%FT%TZ) start $step" >> $log
  case "$step" in
    identity) cmd=("${PY[@]}" tools/export_ctx_own2_override.py --identity --version 67) ;;
    dry) cmd=("${PY[@]}" tools/export_ctx_own2_override.py --dry-run --version 67) ;;
    pinned) cmd=("${PY[@]}" tools/export_ctx_own2_override.py --dry-run --pinned --version 67) ;;
    build) cmd=("${PY[@]}" tools/export_ctx_own2_override.py --build --version 67 --user-decision "$DECISION") ;;
    audit) cmd=("${PY[@]}" tools/audit_export.py --version 67 --compare 66 --report docs/export_audit/v67_audit.json) ;;
    *) echo "unknown step $step" >> $log; exit 2 ;;
  esac
  if "${cmd[@]}" >> runs/ctx_own2/logs/vm_$step.log 2>&1; then
    echo "$(date -u +%FT%TZ) ok $step" >> $log; touch runs/ctx_own2/vm_$step.ok
  else
    echo "$(date -u +%FT%TZ) FAILED $step" >> $log; touch runs/ctx_own2/vm_$step.failed; exit 1
  fi
done
echo "$(date -u +%FT%TZ) driver done: $*" >> $log
