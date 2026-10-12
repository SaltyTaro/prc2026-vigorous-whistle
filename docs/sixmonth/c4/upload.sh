#!/bin/bash
# usage: upload.sh LIST LOG -- tar the listed repo-relative files (dereferencing junctions) into arcw0's overlaid repo, then hash-check
cd /d/kaggle/PRC-Flight/project4
echo "$(date -u +%FT%TZ) start $(wc -l < $LIST) files" >> $LOG
tar chf - -T $LIST | ssh -o ServerAliveInterval=30 arcw0 "cd $R && tar xf -" || { echo "$(date -u +%FT%TZ) UPLOAD_FAILED" >> $LOG; exit 1; }
echo "$(date -u +%FT%TZ) sent; hashing" >> $LOG
tr '\n' '\0' < $LIST | xargs -0 sha256sum | sed 's/ \*/  /' | sort -k2 > $LIST.local.sha
ssh arcw0 "cd $R && tr '\n' '\0' | xargs -0 sha256sum | sort -k2" < $LIST > $LIST.remote.sha
if diff -q $LIST.local.sha $LIST.remote.sha > /dev/null; then echo "$(date -u +%FT%TZ) UPLOAD_OK $(wc -l < $LIST.local.sha) files hash-equal" >> $LOG
else echo "$(date -u +%FT%TZ) HASH_MISMATCH $(diff $LIST.local.sha $LIST.remote.sha | grep -c '^<')" >> $LOG; fi
