#!/bin/bash
# VM: vmrun.sh NAME CMD...  -- CMD in the repo under strace (file syscalls); log runs/sixmonth/devmonths/logs/NAME.log, missing repo paths NAME.enoent
R=/home/azureuser/prc-flight-2026; cd $R
L=runs/sixmonth/devmonths/logs; mkdir -p $L /mnt/scratch/trace
T=/mnt/scratch/trace/$NAME.strace
echo "$(date -u +%FT%TZ) START $NAME: $*" >> $L/$NAME.log
strace -f -qq -e trace=openat,open,stat,newfstatat,statx,access,lstat -e signal=none -o $T "$@" >> $L/$NAME.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $NAME rc=$rc" >> $L/$NAME.log
grep -E 'ENOENT' $T | grep -oE '"[^"]*"' | tr -d '"' | sed -e "s#^$R/##" -e 's#^\./##' | grep -E '^(runs|data|scratch|docs|configs)/' | sort -u > $L/$NAME.enoent
echo "rc=$rc enoent=$(wc -l < $L/$NAME.enoent)" >> $L/$NAME.log
exit $rc
