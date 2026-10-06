#!/usr/bin/env bash
# Mirror finished LIRF own-points design runs (configs/lirf_own_design_v1.json) from arcw0-arcw3: every file of
# runs/lirf_own/design/<variant>_s<seed>/design, each checked against the worker's SHA-256, plus the logs.
# Usage: tools/mirror_lirf_own_design.sh [worker numbers, default 0 1 2 3]
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026/runs/lirf_own
for n in ${@:-0 1 2 3}; do
  for d in $(timeout 30 ssh -o BatchMode=yes arcw$n "cd $R/design 2>/dev/null && for x in */design; do [ -f \$x/run.json ] && echo \$x; done" < /dev/null || true); do
    mkdir -p runs/lirf_own/design/$d
    timeout 600 ssh -o BatchMode=yes arcw$n "cd $R/design/$d && tar cf - ." < /dev/null | tar xf - -C runs/lirf_own/design/$d
    remote=$(timeout 60 ssh -o BatchMode=yes arcw$n "cd $R/design/$d && sha256sum *" < /dev/null | awk '{sub(/^[*]/, "", $2); print $1, $2}' | sort -k2)
    local_=$(cd runs/lirf_own/design/$d && sha256sum * | awk '{sub(/^[*]/, "", $2); print $1, $2}' | sort -k2)
    [ "$remote" = "$local_" ] || { echo "HASH MISMATCH arcw$n $d"; exit 1; }
    echo "arcw$n $d ok ($(echo "$remote" | wc -l) files)"
  done
  mkdir -p runs/lirf_own/logs_mirror/arcw$n
  timeout 120 ssh -o BatchMode=yes arcw$n "cd $R && tar cf - logs" < /dev/null | tar xf - -C runs/lirf_own/logs_mirror/arcw$n
done
