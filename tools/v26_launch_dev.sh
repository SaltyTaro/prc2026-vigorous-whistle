#!/usr/bin/env bash
# Launch a worker's prepared (month, variant) chain (runs/adsb_v26/chain_dev_HOST.sh) under nohup. Usage: v26_launch_dev.sh HOST
set -euo pipefail
host="$1"
remote=/home/azureuser/prc-flight-2026
timeout 30 ssh -o BatchMode=yes "$host" "mkdir -p $remote/runs/adsb_v26/dev"
scp -q "runs/adsb_v26/chain_dev_${host}.sh" "$host:$remote/runs/adsb_v26/chain_dev.sh"
timeout 40 ssh -o BatchMode=yes "$host" "cd $remote && nohup setsid bash runs/adsb_v26/chain_dev.sh < /dev/null > /dev/null 2>&1 & sleep 2; ps -eo cmd | grep -c '[r]un_adsb_v26.py'" || echo "launched (ssh channel held by the detached session)"
