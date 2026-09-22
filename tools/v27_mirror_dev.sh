#!/usr/bin/env bash
# Mirror a worker's adsb_v27 development outputs (partials and logs) into runs/adsb_v27/dev. Usage: v27_mirror_dev.sh HOST
set -euo pipefail
host="$1"
cd "$(dirname "$0")/.."
mkdir -p runs/adsb_v27/dev
ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar cf - runs/adsb_v27/dev/partial_*.json runs/adsb_v27/dev/partial_*.parquet runs/adsb_v27/dev/launch_*.log runs/adsb_v27/chain_*.log 2>/dev/null; true" | tar xf -
ls runs/adsb_v27/dev/ | grep -c partial || true
