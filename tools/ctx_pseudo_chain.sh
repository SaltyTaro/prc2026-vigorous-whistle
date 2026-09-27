#!/usr/bin/env bash
# Pseudo-label adaptation test (configs/adsb_ctx_pseudo_v1.json) on one GPU node for seed SEED: pg and pgd refit-fold runs as two lanes.
set -u
cd ~/prc
seed=$1
mkdir -p runs/adsb_ctx_pseudo/logs
echo "chain start $(date -u +%FT%TZ) seed $seed" >> runs/adsb_ctx_pseudo/logs/chain.log
one() {
  name=$1; kind=$2
  out=runs/adsb_ctx_pseudo/${name}_s${seed}
  [ -f "$out/refit/run.json" ] && { echo "skip ${name}_s${seed}" >> runs/adsb_ctx_pseudo/logs/chain.log; return 0; }
  echo "start ${name}_s${seed} $(date -u +%FT%TZ)" >> runs/adsb_ctx_pseudo/logs/chain.log
  if .venv/bin/python -X utf8 tools/run_adsb_ctx_pseudo.py --config configs/adsb_ctx_pseudo_v1.json --variant ctx_e32 --seed "$seed" --fold refit \
       --output "$out" --device cuda --threads 3 --pseudo "$kind" --december-cache runs/adsb_ctx_pseudo/december_cache > runs/adsb_ctx_pseudo/logs/${name}_s${seed}.log 2>&1; then
    echo "done ${name}_s${seed} $(date -u +%FT%TZ)" >> runs/adsb_ctx_pseudo/logs/chain.log
  else
    echo "FAILED ${name}_s${seed} $(date -u +%FT%TZ)" >> runs/adsb_ctx_pseudo/logs/chain.log
  fi
}
one pg gate & one pgd gate_dark & wait
echo "chain done $(date -u +%FT%TZ) seed $seed" >> runs/adsb_ctx_pseudo/logs/chain.log
