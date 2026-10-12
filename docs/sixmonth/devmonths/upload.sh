#!/bin/bash
# workstation: upload.sh HOST -- code at HEAD (git archive) + .git + the reading's inputs (want.txt) into HOST's overlaid repo; hash-check
cd /d/kaggle/PRC-Flight/project4
echo "$(date -u +%FT%TZ) start $H" >> $LOG
git -c core.autocrlf=false archive HEAD | ssh -o ServerAliveInterval=30 $H "cd $R && tar xf -" || { echo "CODE_FAILED" >> $LOG; exit 1; }
tar cf - .git | ssh -o ServerAliveInterval=30 $H "cd $R && rm -rf .git && tar xf -" || { echo "GIT_FAILED" >> $LOG; exit 1; }
tar chf - -T $V/want.txt | ssh -o ServerAliveInterval=30 $H "cd $R && tar xf -" || { echo "DATA_FAILED" >> $LOG; exit 1; }
tr '\n' '\0' < $V/want.txt | xargs -0 sha256sum | sed 's/ \*/  /' | sort -k2 > $V/want_$H.local.sha
ssh $H "cd $R && tr '\n' '\0' | xargs -0 sha256sum | sort -k2" < $V/want.txt > $V/want_$H.remote.sha
if diff -q $V/want_$H.local.sha $V/want_$H.remote.sha > /dev/null; then echo "$(date -u +%FT%TZ) UPLOAD_OK $(wc -l < $V/want_$H.local.sha) files" >> $LOG
else echo "$(date -u +%FT%TZ) HASH_MISMATCH" >> $LOG; fi
ssh $H "cd $R && git status --short configs/sixmonth_devmonths_v1.json tools/run_sixmonth_devmonths.py tools/run_sixmonth.py | head; git log --oneline -1" >> $LOG 2>&1
tail -3 $LOG
