#!/usr/bin/env bash
# GPU sprint code-path smoke on one node (not a reading): an undeclared scratch copy of configs/adsb_ctx3_design_v1.json with one epoch
# (tail readout over that epoch), variant aux50 (masking + auxiliary head + readouts) and wide, 20,000 fitted rows, outputs under
# runs/adsb_ctx3/smoke/. Usage (on the node, from ~/prc): bash tools/ctx3_smoke.sh
set -euo pipefail
cd ~/prc
mkdir -p runs/adsb_ctx3/smoke
.venv/bin/python - <<'PY'
import json
c = json.load(open("configs/adsb_ctx3_design_v1.json"))
c["defaults"]["training"]["epochs"] = 1
c["defaults"]["readouts"]["tail_epochs"] = 1
c["experiment_id"] += "-SMOKE-not-a-reading"
json.dump(c, open("runs/adsb_ctx3/smoke/smoke_config.json", "w"), indent=1)
PY
for v in aux50 wide; do
  rm -rf runs/adsb_ctx3/smoke/${v}_s1
  .venv/bin/python -X utf8 tools/run_adsb_ctx3.py --config runs/adsb_ctx3/smoke/smoke_config.json --variant $v --seed 1 --fold design \
    --output runs/adsb_ctx3/smoke/${v}_s1 --device cuda --threads 6 --limit-rows 20000 > runs/adsb_ctx3/smoke/${v}_s1.log 2>&1
  tail -1 runs/adsb_ctx3/smoke/${v}_s1.log
  ls runs/adsb_ctx3/smoke/${v}_s1/design/
done
.venv/bin/python -m pytest tests/test_ctx3.py -q -p no:cacheprovider 2>&1 | tail -1
