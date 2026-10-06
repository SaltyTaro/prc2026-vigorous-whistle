#!/usr/bin/env bash
# Lever 7 (configs/opdi_stack_v1.json) on arcw4, detached: verify + extract the ranking package and the new code, then the governing run
# and the ranking dry run (tools/run_opdi_stack_decl.py). Markers runs/opdi_stack/state2/*.done; a rerun skips finished steps.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S=runs/opdi_stack/state2
mkdir -p "$S" runs/opdi_stack/logs
step() { local name=$1; shift; if [ -f "$S/$name.done" ]; then echo "skip $name"; return; fi; echo "$(date -u +%FT%TZ) start $name"; "$@"; date -u +%FT%TZ > "$S/$name.done"; echo "$(date -u +%FT%TZ) done $name"; }
extract() {
  (cd incoming && sha256sum -c ../runs/opdi_stack/packages2.sha256)
  tar -xf incoming/code2.tar
  tar -xf incoming/rank_deps.tar
  (sha256sum -c runs/opdi_stack/rank_deps_files.sha256 | tail -3)
}
step extract extract
step govern $PY -u tools/run_opdi_stack_decl.py --govern
step dry_run $PY -u tools/run_opdi_stack_decl.py --dry-run
echo "$(date -u +%FT%TZ) ALL DONE"
date -u +%FT%TZ > "$S/all.done"
