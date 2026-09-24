#!/usr/bin/env bash
# adsb_rnn: copy finished residual-network run directories (every <learner>/<month>/ holding a run.json) from arcw0-arcw7 into the
# workspace, then list the local count. usage: tools/rnn_mirror.sh SUBDIR   (e.g. design, or . for the leave-one-month-out learners)
set -euo pipefail
cd "$(dirname "$0")/.."
sub="${1:-design}"
for n in 0 1 2 3 4 5 6 7; do
  timeout 300 ssh -o BatchMode=yes "arcw$n" "cd /home/azureuser/prc-flight-2026/runs/adsb_rnn/$sub 2>/dev/null && find . -mindepth 3 -maxdepth 3 -name run.json -printf '%h\n' | tar cf - -T - 2>/dev/null" \
    | (mkdir -p "runs/adsb_rnn/$sub" && cd "runs/adsb_rnn/$sub" && tar xf - 2>/dev/null) || echo "arcw$n: nothing or failed"
done
find "runs/adsb_rnn/$sub" -mindepth 3 -maxdepth 3 -name run.json | wc -l
