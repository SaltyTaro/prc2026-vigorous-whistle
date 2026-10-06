#!/usr/bin/env bash
# Lever 5 (fork E): billing guard, detached from any agent (nohup + setsid, pid-file singleton runs/ctx_own2/guard.pid, log runs/ctx_own2/guard.log).
# Every 2 minutes, for every prc-ctx4* cluster that sky lists as UP:
#   on-demand (prc-ctx4a-d): autostop in STOP mode after 30 idle minutes (`sky autostop -i 30`), so fits not yet mirrored survive on the disk;
#   spot (prc-ctx4e-l): AWS spot instances cannot be stopped, so autodown after 30 idle minutes (`sky autostop -i 30 --down`); idle means no
#   fit runs, and the detached dispatcher mirrors each finished fit within minutes.
# A cluster is (re)set only when its policy differs (checked against `sky status`). The guard exits at GUARD_UNTIL (default 16 hours after
# start) or after 6 consecutive checks without any prc-ctx4* cluster once no launch retry (ctx_own2_nodes.sh up) is running.
# Never touches prc-ctx3a-d (fork B) or sky-jobs-controller-*. Usage (WSL, project root): setsid nohup bash tools/ctx_own2_guard.sh &
set -u
cd "$(dirname "$0")/.."
pidf=runs/ctx_own2/guard.pid
if [ -f $pidf ] && kill -0 "$(cat $pidf)" 2>/dev/null; then echo "guard already running ($(cat $pidf))"; exit 0; fi
echo $$ > $pidf
log=runs/ctx_own2/guard.log
source ~/.venv-sky/bin/activate
until=$(( $(date +%s) + ${GUARD_HOURS:-16} * 3600 ))
empty=0
echo "$(date -u +%FT%TZ) guard start pid $$ until $(date -u -d @$until +%FT%TZ)" >> $log
declare -A done_policy
while [ "$(date +%s)" -lt "$until" ]; do
  st=$(sky status 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  up=$(echo "$st" | awk '$1 ~ /^prc-ctx4[a-l]$/ && / UP / {print $1}')
  any=$(echo "$st" | awk '$1 ~ /^prc-ctx4[a-l]$/ {print $1}')
  for c in $up; do
    case "$c" in prc-ctx4[a-d]) want="stop30" ;; *) want="down30" ;; esac
    [ "${done_policy[$c]:-}" = "$want" ] && continue
    if [ "$want" = stop30 ]; then sky autostop "$c" -i 30 -y >> $log 2>&1; else sky autostop "$c" -i 30 --down -y >> $log 2>&1; fi \
      && { done_policy[$c]=$want; echo "$(date -u +%FT%TZ) $c autostop $want" >> $log; }
  done
  if [ -z "$any" ] && ! pgrep -f "[c]tx_own2_nodes.sh up" > /dev/null; then empty=$((empty + 1)); else empty=0; fi
  if [ $empty -ge 6 ]; then echo "$(date -u +%FT%TZ) no prc-ctx4 cluster and no launch retry for 6 checks: guard exits" >> $log; break; fi
  sleep 120
done
echo "$(date -u +%FT%TZ) guard end" >> $log
rm -f $pidf
