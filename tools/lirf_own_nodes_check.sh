#!/usr/bin/env bash
# LIRF own-points development round (configs/lirf_own_dev_v1.json): before any fit and after any node incident, compare the SHA-256 of
# every input file the fits read on each GPU node with the workstation's: the timeline bundle's features, the own-column tables
# (data/external/adsblol/features_v19own), the v19 tables, the committee ADS-B cache (runs/adsb_nn/features on the nodes =
# runs/adsb_ctx2/inputs/adsb_features_arcw0 here) and the December build. Run in WSL from the project root.
# Usage: NODES="prc-ctx3a prc-ctx3b ..." tools/lirf_own_nodes_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a nodes <<< "${NODES:-prc-ctx3a prc-ctx3b prc-ctx3c prc-ctx3d}"
bundle=$(python3 -c "import json;print(json.load(open('data/timeline/latest.json'))['bundle'])" | tr '\\' '/')
declare -A pairs=(
  ["$bundle/features"]="$bundle/features"
  ["data/external/adsblol/features_v19own"]="data/external/adsblol/features_v19own"
  ["data/external/adsblol/features_v19"]="data/external/adsblol/features_v19"
  ["runs/adsb_ctx2/inputs/adsb_features_arcw0"]="runs/adsb_nn/features"
  ["data/december/d198dea10625b0a75977/features"]="data/december/d198dea10625b0a75977/features"
)
tmp=$(mktemp -d)
for local_dir in "${!pairs[@]}"; do
  (cd "$local_dir" && sha256sum *.parquet *.json 2>/dev/null | sort -k2) > "$tmp/$(echo "$local_dir" | tr / _).local"
done
status=0
for n in "${nodes[@]}"; do
  for local_dir in "${!pairs[@]}"; do
    remote_dir=${pairs[$local_dir]}
    key=$(echo "$local_dir" | tr / _)
    ssh -o ConnectTimeout=20 "$n" "cd ~/prc/$remote_dir && sha256sum *.parquet *.json 2>/dev/null | sort -k2" > "$tmp/$key.$n" || true
    # every local file must exist remotely with the same hash (remote may hold extra files, e.g. manifests)
    missing=$(join -1 2 -2 2 -v 1 <(sort -k2 "$tmp/$key.local") <(sort -k2 "$tmp/$key.$n") | wc -l)
    differ=$(join -1 2 -2 2 <(sort -k2 "$tmp/$key.local") <(sort -k2 "$tmp/$key.$n") | awk '$2 != $3' | wc -l)
    total=$(wc -l < "$tmp/$key.local")
    if [ "$missing" -ne 0 ] || [ "$differ" -ne 0 ]; then status=1; echo "$n $local_dir: MISMATCH missing $missing differ $differ of $total"; else echo "$n $local_dir: ok ($total files)"; fi
  done
done
rm -rf "$tmp"
exit $status
