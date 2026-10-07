#!/usr/bin/env bash
# Lever 12 (fork N): run ONE fit on this node (started by tools/lfpg_net_dispatch.py through `sky exec --gpus L40S:0.5`, two per GPU).
# Arguments: runner config variant seed fold output extra_features_dir (runner ctx3 = tools/run_adsb_ctx3.py, lomo = tools/run_adsb_ctx_lomo.py).
# A finished fit (run.json) is never re-run. Any partial output of an earlier attempt on this node is removed first, so a fit is always run
# from scratch. Markers: <output>/<fold>.started (written at start), <output>/<fold>.failed (non-zero exit). Log runs/lfpg_net/fit_logs/<tag>.log.
set -u
cd ~/prc
runner=$1; config=$2; variant=$3; seed=$4; fold=$5; output=$6; extra=$7
tag=$(basename "$output")_${fold}
[ -f "$output/$fold/run.json" ] && exit 0
rm -rf "$output/$fold" "$output/$fold.failed"
mkdir -p "$output" runs/lfpg_net/fit_logs
date -u +%FT%TZ > "$output/$fold.started"
case "$runner" in
  ctx3) script=tools/run_adsb_ctx3.py ;;
  lomo) script=tools/run_adsb_ctx_lomo.py ;;
  *) echo "bad runner $runner" > "$output/$fold.failed"; exit 1 ;;
esac
if .venv/bin/python -X utf8 $script --config "$config" --variant "$variant" --seed "$seed" --fold "$fold" --output "$output" \
     --extra-features "$extra" --device cuda --threads 3 > "runs/lfpg_net/fit_logs/${tag}.log" 2>&1; then
  rm -f "$output/$fold.started"
else
  date -u +%FT%TZ > "$output/$fold.failed"
  exit 1
fi
