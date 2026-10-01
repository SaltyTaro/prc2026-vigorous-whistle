#!/usr/bin/env bash
# Mirror finished LIRF context-network design runs (configs/adsb_ctx3_lirf_design_v1.json) from arcw0-arcw3: every file of
# runs/adsb_ctx3_lirf/design/<variant>_s<seed>/design (model, preprocessor, architecture, run record, October predictions), each
# checked against the worker's SHA-256. Usage: tools/mirror_ctx3_lirf.sh [worker numbers, default 0 1 2 3]
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026/runs/adsb_ctx3_lirf/design
for n in ${@:-0 1 2 3}; do
  for d in $(timeout 30 ssh -o BatchMode=yes arcw$n "cd $R 2>/dev/null && for x in */design; do [ -f \$x/run.json ] && echo \$x; done" || true); do
    mkdir -p runs/adsb_ctx3_lirf/design/$d
    timeout 600 ssh -o BatchMode=yes arcw$n "cd $R/$d && tar cf - ." | tar xf - -C runs/adsb_ctx3_lirf/design/$d
    remote=$(timeout 60 ssh -o BatchMode=yes arcw$n "cd $R/$d && sha256sum *" | awk '{sub(/^[*]/, "", $2); print $1, $2}' | sort -k2)
    local_=$(cd runs/adsb_ctx3_lirf/design/$d && sha256sum * | awk '{sub(/^[*]/, "", $2); print $1, $2}' | sort -k2)
    [ "$remote" = "$local_" ] || { echo "HASH MISMATCH arcw$n $d"; exit 1; }
    echo "arcw$n $d ok ($(echo "$remote" | wc -l) files)"
  done
done
