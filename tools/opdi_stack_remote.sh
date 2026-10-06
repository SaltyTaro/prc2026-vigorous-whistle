#!/usr/bin/env bash
# Lever 7 (docs/opdi_stack/README.md): the OPDI stacker screen on an Azure worker (arcw4), detached.
# Usage on the VM, from the repository root: setsid nohup bash tools/opdi_stack_remote.sh > runs/opdi_stack/logs/remote.log 2>&1 < /dev/null &
# Steps: verify and extract the uploaded packages -> identity (must reproduce the pinned v66 control exactly) -> OPDI features (dev, rank)
# -> coverage -> screen variants opdi, opdi3 -> read. Each step writes runs/opdi_stack/state/<step>.done; a rerun skips finished steps.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
S=runs/opdi_stack/state
mkdir -p "$S" runs/opdi_stack/logs
step() { local name=$1; shift; if [ -f "$S/$name.done" ]; then echo "skip $name"; return; fi; echo "$(date -u +%FT%TZ) start $name"; "$@"; date -u +%FT%TZ > "$S/$name.done"; echo "$(date -u +%FT%TZ) done $name"; }
extract() {
  (cd incoming && sha256sum -c ../runs/opdi_stack/packages.sha256)
  tar -xf incoming/code.tar
  tar -xf incoming/runs_deps.tar
  tar -xf incoming/surface_missing.tar -C data/external/adsblol/surface
  (cd data/external/adsblol/surface && sha256sum -c ../../../../runs/opdi_stack/surface_missing.sha256)
}
step extract extract
step identity $PY -u tools/run_opdi_stack.py --score identity
step features_dev $PY -u tools/run_opdi_stack.py --features dev
step features_rank $PY -u tools/run_opdi_stack.py --features rank
step coverage $PY -u tools/run_opdi_stack.py --coverage
step score_opdi $PY -u tools/run_opdi_stack.py --score opdi
step score_opdi3 $PY -u tools/run_opdi_stack.py --score opdi3
step read $PY -u tools/run_opdi_stack.py --read
echo "$(date -u +%FT%TZ) ALL DONE"
date -u +%FT%TZ > "$S/all.done"
