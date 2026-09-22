#!/usr/bin/env bash
# Sync the adsb_v25 development inputs to a worker: every declaration parent, the label-free matches_v11 feature caches, the declaration,
# the development tools and the source packages they import (runs/adsb_v25/sync_files.txt). Usage: v25_sync_dev.sh HOST
set -euo pipefail
host="$1"
cd "$(dirname "$0")/.."
wc -l runs/adsb_v25/sync_files.txt
tar cf - -T runs/adsb_v25/sync_files.txt | ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar xf - && echo synced"
