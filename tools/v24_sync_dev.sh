#!/usr/bin/env bash
# Sync the adsb_v24 development inputs to a worker: sync_list.txt (declaration parents + records) plus scratch/v24, the residual feature cache,
# the matches_v11 month tables, the December scoring package and the development tools. Usage: sync_dev.sh HOST
set -euo pipefail
host="$1"
cd "$(dirname "$0")/../.."
{
  cat runs/adsb_v24/sync_list.txt
  ls scratch/v24/*.parquet
  ls runs/adsb_v24/features/*.parquet
  ls data/external/adsblol/matches_v11/matches2_2025-*.parquet data/external/adsblol/matches_v11/matches2_2025-*.json
  ls src/prc2026_december/*.py src/prc2026_postprocess/*.py src/prc2026_lirf_mixture/*.py
  echo tools/run_adsb_v24.py; echo tools/run_adsb_v11.py; echo tools/run_lirf_mixture.py; echo tools/run_postprocess.py; echo configs/adsb_v24_v1.json
  echo docs/adsb_v24/scratch_build.json
} | sort -u > runs/adsb_v24/sync_files.txt
wc -l runs/adsb_v24/sync_files.txt
tar cf - -T runs/adsb_v24/sync_files.txt | ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar xf - && echo synced"
