#!/usr/bin/env bash
# adsb_rnn: copy the residual-network tool, its configurations and the source files it imports to a worker, then print the worker's
# hashes of those files and a presence check of the data it reads. Usage: tools/rnn_sync.sh HOST
set -euo pipefail
host="$1"
cd "$(dirname "$0")/.."
files="tools/run_adsb_rnn.py tools/rnn_chain.sh configs/adsb_rnn_network_v1.json configs/adsb_rnn_design_v1.json src/prc2026_nn/__init__.py src/prc2026_nn/preprocess.py src/prc2026_nn/model.py src/prc2026_nn/train.py tools/run_adsb_v24.py tools/run_adsb_v11.py src/prc2026_adsb/features.py tools/run_adsb_learner.py configs/adsb_v25_v1.json"
tar cf - $files | ssh -o BatchMode=yes "$host" "cd /home/azureuser/prc-flight-2026 && tar xf - && sha256sum $files src/prc2026_timeline/features.py src/prc2026_timeline/bundle.py \
  && ls scratch/v24/*.parquet | wc -l && ls runs/adsb_v24/features/matches_v11_*.parquet | wc -l && ls data/external/adsblol/features_v19/*.parquet | wc -l \
  && ls \$(python3 -c 'import json;print(json.load(open(\"data/timeline/latest.json\"))[\"bundle\"])')/features | wc -l && ls data/december/d198dea10625b0a75977/features && free -g | head -2 && df -h / | tail -1"
