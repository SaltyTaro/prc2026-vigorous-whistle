#!/usr/bin/env bash
# LIRF own-points design (configs/lirf_own_design_v1.json): 10 October design fits on arcw0-arcw3, CPU, as declared:
#   arcw0 lm25_32own s21 (8 threads) + ctx_lirf_own s1 (4) + ctx_lirf s1 (4, reproduction control)
#   arcw1 lm25_32own s22 (8) + lm25_32 s21 (8, reproduction control)
#   arcw2 lm25_32own s23 (8) + ctx_lirf_own s2 (4) + ctx_lirf_own s3 (4)
#   arcw3 lm25_32own s24 (8) + ctx_lirf_own s4 (4)
# ctx_lirf* use tools/run_adsb_ctx.py (round 1, the saved round-B seeds' tool); lm25_32* use tools/run_adsb_ctx3.py --device cpu.
# Own variants read --extra-features data/external/adsblol/features_v19own; controls the default features_v19.
# Usage: tools/launch_lirf_own_design.sh [worker numbers, default 0 1 2 3]
set -euo pipefail
cd "$(dirname "$0")/.."
R=/home/azureuser/prc-flight-2026
D=configs/lirf_own_design_v1.json
C1=configs/lirf_own_network_ctx1_v1.json
C3=configs/lirf_own_network_ctx3_v1.json
OWN=data/external/adsblol/features_v19own
jobs_for() {  # "tool config variant seed threads extra" lines of one worker
  case $1 in
    0) echo "run_adsb_ctx3.py $C3 lm25_32own 21 8 $OWN"; echo "run_adsb_ctx.py $C1 ctx_lirf_own 1 4 $OWN"; echo "run_adsb_ctx.py $C1 ctx_lirf 1 4 -" ;;
    1) echo "run_adsb_ctx3.py $C3 lm25_32own 22 8 $OWN"; echo "run_adsb_ctx3.py $C3 lm25_32 21 8 -" ;;
    2) echo "run_adsb_ctx3.py $C3 lm25_32own 23 8 $OWN"; echo "run_adsb_ctx.py $C1 ctx_lirf_own 2 4 $OWN"; echo "run_adsb_ctx.py $C1 ctx_lirf_own 3 4 $OWN" ;;
    3) echo "run_adsb_ctx3.py $C3 lm25_32own 24 8 $OWN"; echo "run_adsb_ctx.py $C1 ctx_lirf_own 4 4 $OWN" ;;
  esac
}
for n in ${@:-0 1 2 3}; do
  scp -q $D $C1 $C3 arcw$n:$R/configs/
  timeout 60 ssh -o BatchMode=yes arcw$n "mkdir -p $R/runs/lirf_own/logs $R/runs/lirf_own/design" < /dev/null
  while read -r tool config variant seed threads extra; do
    dev=""; [ "$tool" = run_adsb_ctx3.py ] && dev="--device cpu"
    ex=""; [ "$extra" != - ] && ex="--extra-features $extra"
    out=runs/lirf_own/design/${variant}_s$seed
    timeout 60 ssh -o BatchMode=yes arcw$n "cd $R && test ! -e $out/design/run.json && (setsid nohup .venv/bin/python -u tools/$tool --config $config --variant $variant --seed $seed --fold design --output $out $dev --threads $threads $ex > runs/lirf_own/logs/${variant}_s$seed.log 2>&1 < /dev/null &) && echo arcw$n launched $variant s$seed" < /dev/null
  done < <(jobs_for $n)
done
