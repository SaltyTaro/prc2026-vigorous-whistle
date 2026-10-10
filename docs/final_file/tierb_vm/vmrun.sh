#!/bin/bash
# On arcw6: vmrun.sh NAME CMD...  -- run CMD in the repo under strace (file syscalls only), log to runs/febjun_tierb/vm/logs/NAME.log,
# write the repo paths it could not find to runs/febjun_tierb/vm/logs/NAME.enoent and the directories it listed to NAME.dirs.
R=/home/azureuser/prc-flight-2026; cd $R
L=runs/febjun_tierb/vm/logs; mkdir -p $L /mnt/scratch/trace
T=/mnt/scratch/trace/$NAME.strace
echo "$(date -u +%FT%TZ) START $NAME: $*" >> $L/$NAME.log
strace -f -qq -e trace=openat,open,stat,newfstatat,statx,access,lstat -e signal=none -o $T "$@" >> $L/$NAME.log 2>&1
rc=$?
echo "$(date -u +%FT%TZ) END $NAME rc=$rc" >> $L/$NAME.log
# ENOENT paths, made repo-relative
grep -E 'ENOENT' $T | grep -oE '"[^"]*"' | tr -d '"' | sed -e "s#^$R/##" -e 's#^\./##' | grep -E '^(runs|data|scratch|docs|configs|tools|src)/' | sort -u > $L/$NAME.enoent
# directories opened for listing
grep -E 'O_DIRECTORY' $T | grep -v ENOENT | grep -oE '"[^"]*"' | tr -d '"' | sed -e "s#^$R/##" -e 's#^\./##' | grep -E '^(runs|data|scratch)/' | sort -u > $L/$NAME.dirs
echo "rc=$rc enoent=$(wc -l < $L/$NAME.enoent) dirs=$(wc -l < $L/$NAME.dirs)" >> $L/$NAME.log
exit $rc
