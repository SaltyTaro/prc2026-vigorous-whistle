#!/usr/bin/env bash
# Code-path smoke (not a reading) for the evening GPU work, on one node: an undeclared one-epoch scratch copy of
# configs/adsb_ctx_network_v3.json through all four development folds (seasonal, forward, refit with its December leaf, refit12) and of
# configs/adsb_ctx3_design_v2.json for mask100_32 (p = 1.0), 20,000 fitted rows each, outputs and December leaves under
# runs/adsb_ctx3/smoke_dev/. Usage (on the node, from ~/prc): bash tools/ctx3_smoke_dev.sh
set -euo pipefail
cd ~/prc
s=runs/adsb_ctx3/smoke_dev
rm -rf $s && mkdir -p $s
.venv/bin/python - <<'PY'
import json
for src, dst in [("configs/adsb_ctx_network_v3.json", "runs/adsb_ctx3/smoke_dev/network.json"), ("configs/adsb_ctx3_design_v2.json", "runs/adsb_ctx3/smoke_dev/design2.json")]:
    c = json.load(open(src))
    c["defaults"]["training"]["epochs"] = 1
    c["experiment_id"] += "-SMOKE-not-a-reading"
    json.dump(c, open(dst, "w"), indent=1)
PY
for fold in seasonal forward refit refit12; do
  .venv/bin/python -X utf8 tools/run_adsb_ctx3.py --config $s/network.json --variant ctx_e32 --seed 1 --fold $fold --output $s/ctx_e32_s1 \
    --device cuda --threads 6 --limit-rows 20000 --december-cache $s/december > $s/${fold}.log 2>&1 || { echo "FAILED $fold"; tail -5 $s/${fold}.log; exit 1; }
  echo "$fold: $(tail -1 $s/${fold}.log | cut -c1-220)"
  ls $s/ctx_e32_s1/$fold | tr '\n' ' '; echo
done
ls $s/december
.venv/bin/python -X utf8 tools/run_adsb_ctx3.py --config $s/design2.json --variant mask100_32 --seed 5 --fold design --output $s/mask100_32_s5 \
  --device cuda --threads 6 --limit-rows 20000 > $s/mask100.log 2>&1 || { echo "FAILED mask100"; tail -5 $s/mask100.log; exit 1; }
grep -m1 masked_rows $s/mask100.log | cut -c1-200
