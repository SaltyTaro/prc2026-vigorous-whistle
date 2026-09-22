#!/usr/bin/env bash
# Sync the adsb_v26 development inputs to a worker (runs/adsb_v26/sync_files.txt: every declaration parent, the feature caches, tools, packages). Usage: v26_sync_dev.sh HOST
set -euo pipefail
host="$1"
cd "$(dirname "$0")/.."
tar cf - -T runs/adsb_v26/sync_files.txt | ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar xf - && echo synced"
