#!/usr/bin/env bash
# Season-weighted network test (configs/adsb_ctx_season_v1.json) on one GPU node for seed SEED: lane 1 = sw1 (seasonal, target January)
# then sw12 (refit, target December); lane 2 = sw7 (seasonal, target July), then for seed 401 only sw0 (unweighted reproduction).
# Finished runs are skipped. Logs runs/adsb_ctx_season/logs/<run>_s<seed>.log; markers in runs/adsb_ctx_season/logs/chain.log.
set -u
cd ~/prc
seed=$1
mkdir -p runs/adsb_ctx_season/logs
echo "chain start $(date -u +%FT%TZ) seed $seed" >> runs/adsb_ctx_season/logs/chain.log
one() {
  name=$1; fold=$2; target=$3; factor=$4
  out=runs/adsb_ctx_season/${name}_s${seed}
  [ -f "$out/$fold/run.json" ] && { echo "skip ${name}_s${seed}" >> runs/adsb_ctx_season/logs/chain.log; return 0; }
  extra=""; [ "$target" != "none" ] && extra="--season-target $target --season-factor $factor"
  echo "start ${name}_s${seed} $(date -u +%FT%TZ)" >> runs/adsb_ctx_season/logs/chain.log
  if .venv/bin/python -X utf8 tools/run_adsb_ctx_season.py --config configs/adsb_ctx_season_v1.json --variant ctx_e32 --seed "$seed" --fold "$fold" \
       --output "$out" --device cuda --threads 3 $extra > runs/adsb_ctx_season/logs/${name}_s${seed}.log 2>&1; then
    echo "done ${name}_s${seed} $(date -u +%FT%TZ)" >> runs/adsb_ctx_season/logs/chain.log
  else
    echo "FAILED ${name}_s${seed} $(date -u +%FT%TZ)" >> runs/adsb_ctx_season/logs/chain.log
  fi
}
lane1() { one sw1 seasonal 1 3; one sw12 refit 12 3; }
lane2() { one sw7 seasonal 7 3; [ "$seed" = 401 ] && one sw0 seasonal none 1; }
lane1 & lane2 & wait
echo "chain done $(date -u +%FT%TZ) seed $seed" >> runs/adsb_ctx_season/logs/chain.log
