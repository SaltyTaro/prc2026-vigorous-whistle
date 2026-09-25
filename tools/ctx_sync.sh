#!/usr/bin/env bash
# adsb_ctx: copy the context network's code and configurations to a worker, rebuild the token files there from the worker's own raw
# data and match tables (tools/build_ctx_tokens.py, ~25 s), run the context tests, and print the token content hashes so they can be
# compared with the local build (data/ctx/tokens_v1/manifest.json). Usage: tools/ctx_sync.sh HOST
set -euo pipefail
host="$1"
cd "$(dirname "$0")/.."
files="src/prc2026_ctx/__init__.py src/prc2026_ctx/tokens.py src/prc2026_ctx/model.py src/prc2026_ctx/train.py tools/run_adsb_ctx.py tools/build_ctx_tokens.py tools/ctx_chain.sh tests/test_ctx.py configs/adsb_ctx_design_v1.json"
tar cf - $files | ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar xf - && chmod +x tools/ctx_chain.sh \
  && .venv/bin/python -X utf8 -B tools/build_ctx_tokens.py > runs/ctx_tokens_build.log 2>&1 \
  && .venv/bin/python -m pytest tests/test_ctx.py -q -p no:cacheprovider 2>&1 | tail -1 \
  && .venv/bin/python -c 'import json;m=json.load(open(\"data/ctx/tokens_v1/manifest.json\"));print(json.dumps({k:v[\"content_sha256\"] for k,v in m[\"files\"].items()}))' \
  && ls runs/adsb_nn/features/*.parquet | wc -l && free -g | head -2 | tail -1 && df -h / | tail -1"
