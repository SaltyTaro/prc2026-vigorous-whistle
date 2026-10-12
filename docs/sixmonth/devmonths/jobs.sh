#!/bin/bash
# VM: jobs.sh MODE -- MODE stack_janjul | stack_novdec (four C6 fit sets at once, 4 threads each) | ss (eight C14 fit sets in turn).
# Detached; log runs/sixmonth/devmonths/logs/jobs_MODE.log; ends with ALL_DONE or FAILED.
cd /home/azureuser/prc-flight-2026
PY=".venv/bin/python -X utf8 -B tools/run_sixmonth_devmonths.py"
RUN="bash runs/sixmonth/devmonths/vm/vmrun.sh"
say() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
  stack_janjul) ms="2025-01 2025-07" ;;
  stack_novdec) ms="2025-11 2025-12" ;;
esac
  ok=1; for m in 2025-01 2025-07 2025-11 2025-12; do for v in a b; do one ss $m $v || ok=0; done; done
else
  pids=""; for m in $ms; do for v in a b; do one stack $m $v & pids="$pids $!"; done; done
  ok=1; for p in $pids; do wait $p || ok=0; done
fi
[ $ok = 1 ] && say "ALL_DONE" || say "FAILED (see above)"
