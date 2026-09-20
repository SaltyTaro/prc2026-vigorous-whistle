#!/usr/bin/env bash
# Launch one held-out month of tools/run_adsb_v24.py on a worker under nohup. Usage: launch_dev.sh HOST MONTH [THREADS]
set -euo pipefail
host="$1"; month="$2"; threads="${3:-16}"
remote=/home/azureuser/prc-flight-2026
script="cd $remote && mkdir -p runs/adsb_v24/dev && (echo \"=== dev $month threads $threads \$(date -u +%FT%TZ)\" >> runs/adsb_v24/dev/launch_$month.log && .venv/bin/python -u tools/run_adsb_v24.py --held-out $month --threads $threads >> runs/adsb_v24/dev/launch_$month.log 2>&1; echo \"=== dev $month done \$(date -u +%FT%TZ)\" >> runs/adsb_v24/dev/launch_$month.log) > /dev/null 2>&1"
printf '%s\n' "$script" > "runs/adsb_v24/chain_dev_${host}_${month}.sh"
scp -q "runs/adsb_v24/chain_dev_${host}_${month}.sh" "$host:$remote/runs/adsb_v24/chain_dev_$month.sh"
timeout 40 ssh -o BatchMode=yes "$host" "cd $remote && nohup setsid bash runs/adsb_v24/chain_dev_$month.sh < /dev/null > /dev/null 2>&1 & sleep 2; ps -eo cmd | grep -c '[r]un_adsb_v24.py'" || echo "launched (ssh channel held by the detached session)"
