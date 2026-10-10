#!/bin/bash
# Workstation, detached: when arcw6's real run ends (REAL_DONE or a failure), copy its outputs back to the same paths and hash-check them.
# Lane M's checks/ (validation copies) are not brought back. Log runs/febjun_tierb/vm/mirror_back.log.
cd /d/kaggle/PRC-Flight/project4
V=runs/febjun_tierb/vm; LOG=$V/mirror_back2.log; R=/home/azureuser/prc-flight-2026
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "start (pid $$)"
while true; do
  st=$(ssh -o ConnectTimeout=30 arcw6 "awk '/resume lane C/{f=1; next} f' $R/$V/logs/real_vm.log | grep -oE 'REAL_DONE|STEP_FAILED [a-zA-Z_]+' | tail -1" 2>/dev/null)
  [ -n "$st" ] && break
  sleep 300
done
say "arcw6 says: $st"
ssh arcw6 "cd $R && find runs/febjun_tierb/model runs/febjun_tierb/chain/own runs/febjun_tierb/chain/root runs/febjun_tierb/chain/work $V/logs docs/final_file -type f \
  ! -path 'runs/febjun_tierb/model/checks/*' ! -path 'docs/final_file/*.md' -newer $V/real_vm.sh -print | sort" > $V/back2_list.txt
say "$(wc -l < $V/back2_list.txt) files to copy back"
ssh arcw6 "cd $R && tar cf - -T -" < $V/back2_list.txt | tar xf - || { say "COPY_FAILED"; exit 1; }
ssh arcw6 "cd $R && tr '\n' '\0' | xargs -0 sha256sum | sort -k2" < $V/back2_list.txt > $V/back2_remote.sha
tr '\n' '\0' < $V/back2_list.txt | xargs -0 sha256sum | sed 's/ \*/  /' | sort -k2 > $V/back2_local.sha
if diff -q $V/back2_local.sha $V/back2_remote.sha > /dev/null; then say "MIRROR_BACK_OK $(wc -l < $V/back2_local.sha) files hash-equal ($st)"; else say "MIRROR_BACK_MISMATCH"; fi
