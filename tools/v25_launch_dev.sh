#!/usr/bin/env bash
# Launch a chain of held-out months of tools/run_adsb_v25.py on a worker under nohup. Usage: v25_launch_dev.sh HOST MONTH [MONTH ...]
set -euo pipefail
host="$1"; shift
remote=/home/azureuser/prc-flight-2026
threads=16
{
  echo "cd $remote && mkdir -p runs/adsb_v25/dev"
  for month in "$@"; do
    echo "echo \"=== dev $month threads $threads \$(date -u +%FT%TZ)\" >> runs/adsb_v25/dev/launch_$month.log"
    echo ".venv/bin/python -u tools/run_adsb_v25.py --held-out $month --threads $threads >> runs/adsb_v25/dev/launch_$month.log 2>&1"
    echo "echo \"=== dev $month done \$(date -u +%FT%TZ)\" >> runs/adsb_v25/dev/launch_$month.log"
  done
  echo "echo \"chain done \$(date -u +%FT%TZ)\" >> runs/adsb_v25/chain_$host.log"
} > "runs/adsb_v25/chain_dev_${host}.sh"
timeout 30 ssh -o BatchMode=yes "$host" "mkdir -p $remote/runs/adsb_v25/dev"
scp -q "runs/adsb_v25/chain_dev_${host}.sh" "$host:$remote/runs/adsb_v25/chain_dev.sh"
timeout 40 ssh -o BatchMode=yes "$host" "cd $remote && nohup setsid bash runs/adsb_v25/chain_dev.sh < /dev/null > /dev/null 2>&1 & sleep 2; ps -eo cmd | grep -c '[r]un_adsb_v25.py'" || echo "launched (ssh channel held by the detached session)"
