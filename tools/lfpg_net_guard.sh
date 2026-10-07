#!/usr/bin/env bash
# Lever 12 (fork N): billing guard (a copy of tools/ctx_own2_guard.sh for prc-ctx6a-l), detached from any agent (nohup + setsid, pid-file singleton
# runs/lfpg_net/guard.pid, log runs/lfpg_net/guard.log). Every 2 minutes, for every prc-ctx6* cluster that sky lists as UP:
#   on-demand (prc-ctx6a-h): autostop in STOP mode after 30 idle minutes, so fits not yet mirrored survive on the disk;
#   spot (prc-ctx6i-l): autodown after 30 idle minutes (AWS spot cannot be stopped; the dispatcher mirrors each fit within minutes).
# Exits at GUARD_HOURS (default 30) after start or after 6 consecutive checks without any prc-ctx6* cluster once no launch retry runs.
# Never touches prc-ctx3*/prc-ctx4*/prc-ctx5* or sky-jobs-controller-*. Usage (WSL, project root): setsid nohup bash tools/lfpg_net_guard.sh &
set -u
cd "$(dirname "$0")/.."
pidf=runs/lfpg_net/guard.pid
if [ -f $pidf ] && kill -0 "$(cat $pidf)" 2>/dev/null; then echo "guard already running ($(cat $pidf))"; exit 0; fi
echo $$ > $pidf
log=runs/lfpg_net/guard.log
source ~/.venv-sky/bin/activate
until=$(( $(date +%s) + ${GUARD_HOURS:-30} * 3600 ))
empty=0
echo "$(date -u +%FT%TZ) guard start pid $$ until $(date -u -d @$until +%FT%TZ)" >> $log
declare -A done_policy
while [ "$(date +%s)" -lt "$until" ]; do
  st=$(sky status 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  up=$(echo "$st" | awk '$1 ~ /^prc-ctx6[a-l]$/ && / UP / {print $1}')
  any=$(echo "$st" | awk '$1 ~ /^prc-ctx6[a-l]$/ {print $1}')
  for c in $up; do
    case "$c" in prc-ctx6[a-h]) want="stop30" ;; *) want="down30" ;; esac
    [ "${done_policy[$c]:-}" = "$want" ] && continue
    if [ "$want" = stop30 ]; then sky autostop "$c" -i 30 -y >> $log 2>&1; else sky autostop "$c" -i 30 --down -y >> $log 2>&1; fi \
      && { done_policy[$c]=$want; echo "$(date -u +%FT%TZ) $c autostop $want" >> $log; }
  done
  if [ -z "$any" ] && ! pgrep -f "[l]fpg_net_nodes.sh up" > /dev/null; then empty=$((empty + 1)); else empty=0; fi
  if [ $empty -ge 6 ]; then echo "$(date -u +%FT%TZ) no prc-ctx6 cluster and no launch retry for 6 checks: guard exits" >> $log; break; fi
  sleep 120
done
echo "$(date -u +%FT%TZ) guard end" >> $log
rm -f $pidf
