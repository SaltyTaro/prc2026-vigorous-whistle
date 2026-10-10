#!/bin/bash
# arcw6: clean the stand-in work dir (keep step 0's key map and matches) and relaunch validate_C.sh detached.
cd /home/azureuser/prc-flight-2026
W=runs/febjun_tierb/chain/work_standin
find $W -mindepth 1 -maxdepth 1 ! -name steps -exec rm -rf {} +
find $W/steps -mindepth 1 -maxdepth 1 ! -name rematch -exec rm -rf {} +
rm -rf $W/steps/rematch/base
n=$(ls -d runs/febjun_tierb/vm/logs/round* 2>/dev/null | wc -l); mkdir -p runs/febjun_tierb/vm/logs/round$((n + 1))
mv runs/febjun_tierb/vm/logs/vC_* runs/febjun_tierb/vm/logs/round$((n + 1))/ 2>/dev/null
setsid nohup bash runs/febjun_tierb/vm/validate_C.sh > runs/febjun_tierb/vm/logs/validate_C.out 2>&1 < /dev/null &
echo relaunched
