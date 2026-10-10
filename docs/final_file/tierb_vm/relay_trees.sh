#!/bin/bash
# Workstation, detached: wait for the tree copy loop's 'ALL 58 LOCAL', upload every tree run.json / predictions.parquet arcw6 lacks
# (hash-checked by upload.sh), then write TREES_READY on arcw6. Log runs/febjun_tierb/vm/relay.log.
cd /d/kaggle/PRC-Flight/project4
V=runs/febjun_tierb/vm; LOG=$V/relay.log
echo $$ > $V/relay.pid
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
say "start"
until grep -q "ALL 58 LOCAL" runs/febjun_lomo/mirror_loop.log; do sleep 120; done
say "ALL 58 LOCAL seen"
find runs/febjun_lomo/learners -name run.json -o -name predictions.parquet | sort > $V/trees_all.txt
ssh arcw6 "cd /home/azureuser/prc-flight-2026 && find runs/febjun_lomo/learners -type f -printf '%p\t%s\n'" > $V/trees_vm.tsv
say "uploading $(wc -l < $V/trees_upload.txt) of $(wc -l < $V/trees_all.txt) tree files"
rm -f $V/trees_upload.log
if [ -s $V/trees_upload.txt ]; then bash $V/upload.sh $V/trees_upload.txt $V/trees_upload.log; fi
if [ ! -s $V/trees_upload.txt ] || grep -q UPLOAD_OK $V/trees_upload.log; then
  # every tree file (all 116) hash-equal on both sides
  tr '\n' '\0' < $V/trees_all.txt | xargs -0 sha256sum | sed 's/ \*/  /' | sort -k2 > $V/trees_all.local.sha
  ssh arcw6 "cd /home/azureuser/prc-flight-2026 && tr '\n' '\0' | xargs -0 sha256sum | sort -k2" < $V/trees_all.txt > $V/trees_all.remote.sha
  if [ "$(wc -l < $V/trees_all.local.sha)" = 116 ] && diff -q $V/trees_all.local.sha $V/trees_all.remote.sha > /dev/null; then
    ssh arcw6 "touch /home/azureuser/prc-flight-2026/runs/febjun_tierb/vm/TREES_READY" && say "TREES_READY written (116 files hash-equal)"
  else say "TREE_HASH_MISMATCH"; fi
else say "TREE_UPLOAD_FAILED"; fi
rm -f $V/relay.pid
