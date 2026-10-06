#!/usr/bin/env bash
# LIRF own-points development round: after tools/combo_nodes.sh mirror (ROUND=runs/lirf_own), check every mirrored file of
# runs/lirf_own/nets against each node's SHA-256 (file by file). Run in WSL from the project root. Exit 1 on any mismatch.
# Usage: NODES="prc-ctx3a ..." tools/lirf_own_mirror_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a nodes <<< "${NODES:-prc-ctx3a prc-ctx3b prc-ctx3c prc-ctx3d}"
status=0
for n in "${nodes[@]}"; do
  remote=$(ssh -o ConnectTimeout=20 "$n" "cd ~/prc/runs/lirf_own/nets && find . -type f -print0 | sort -z | xargs -0 sha256sum")
  count=$(echo "$remote" | grep -c . || true)
  bad=$(cd runs/lirf_own/nets && echo "$remote" | sha256sum -c --quiet 2>&1 | grep -c . || true)
  if [ "$bad" -ne 0 ]; then status=1; echo "$n: MISMATCH ($bad of $count files)"; else echo "$n: ok ($count files)"; fi
done
exit $status
