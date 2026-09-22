#!/usr/bin/env bash
# Mirror a worker's adsb_v25 development outputs (month reports, predictions, launch and chain logs) into runs/adsb_v25/dev. Usage: v25_mirror_dev.sh HOST
set -euo pipefail
host="$1"
cd "$(dirname "$0")/.."
mkdir -p runs/adsb_v25/dev
ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar cf - runs/adsb_v25/dev/month_*.json runs/adsb_v25/dev/predictions_*.parquet runs/adsb_v25/dev/launch_*.log runs/adsb_v25/chain_*.log" | tar xf -
ls runs/adsb_v25/dev/
