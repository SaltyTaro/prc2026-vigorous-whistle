#!/usr/bin/env bash
# Rerun every round-2 design job without a run.json (the g_sa2_w256 seeds OOM'd in the per-epoch monitor prediction while sharing
# the L40S with a second lane) alone on the GPU. Same code, config and seeds; only the lane count and the CUDA allocator setting
# (expandable segments, no numerical effect) differ. Failed attempts' logs are preserved in runs/adsb_ctx2/logs/failed_2lanes/.
set -u
cd ~/prc
mkdir -p runs/adsb_ctx2/logs/failed_2lanes
for f in runs/adsb_ctx2/logs/g_*.log; do
  v=$(basename "$f" .log)
  [ -f "runs/adsb_ctx2/design/$v/design/run.json" ] || mv "$f" runs/adsb_ctx2/logs/failed_2lanes/
done
echo "rerun 1 lane $(date -u +%FT%TZ): $(ls runs/adsb_ctx2/logs/failed_2lanes | tr '\n' ' ')" >> runs/adsb_ctx2/logs/chain.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash tools/ctx2_design_chain.sh 1
