#!/usr/bin/env bash
# Mirror finished adsb_nn learner directories (all folds present) and their December leaf caches from the workers to the workstation.
# usage: tools/nn_mirror.sh FOLD [FOLD ...]   (e.g. "seasonal forward refit" or "refit12"); skips seeds already mirrored for those folds.
cd "$(dirname "$0")/.." || exit 1
folds="$*"
for n in ${NN_HOSTS:-0 1 2 3 4 5 6 7}; do
  list=$(timeout 60 ssh -o BatchMode=yes arcw$n "cd /home/azureuser/prc-flight-2026 && for d in runs/adsb_nn/nnb_s*; do ok=1; for f in $folds; do [ -f \$d/\$f/run.json ] || ok=0; done; [ \$ok -eq 1 ] && echo \$d; done" 2>/dev/null)
  for d in $list; do
    need=0; for f in $folds; do [ -f "$d/$f/run.json" ] || need=1; done
    [ $need -eq 0 ] && continue
    leaf=$(echo "$d" | sed 's#/#__#g')
    extra=""; case " $folds " in *" refit "*) extra="runs/december/learners/$leaf.parquet runs/december/learners/$leaf.json";; esac
    timeout 900 ssh arcw$n "cd /home/azureuser/prc-flight-2026 && tar -cf - $(for f in $folds; do printf "%s " "$d/$f"; done) $extra" | tar -xf - && echo "mirrored $d ($folds) from arcw$n"
  done
done
