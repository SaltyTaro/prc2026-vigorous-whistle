#!/usr/bin/env bash
# adsb_ctx3 design (GPU sprint) on one of the four GPU nodes: the declared (variant, seed) runs of configs/adsb_ctx3_design_v1.json whose
# position in the declared order (variants alphabetically, seeds ascending) is NODE modulo NODES, LANES at a time on this node's GPU,
# via tools/run_adsb_ctx3.py --fold design. Finished runs (run.json present) are skipped. Logs runs/adsb_ctx3/logs/<variant>_s<seed>.log;
# marker lines in runs/adsb_ctx3/logs/chain.log. Usage: tools/ctx3_design_chain.sh NODE NODES [LANES]
set -u
cd ~/prc
node=$1; nodes=$2; lanes=${3:-2}
mkdir -p runs/adsb_ctx3/logs
.venv/bin/python -c 'import json,sys;c=json.load(open("configs/adsb_ctx3_design_v1.json"));jobs=[(v,s) for v in sorted(c["runs"]) for s in c["runs"][v]];[print(v,s) for i,(v,s) in enumerate(jobs) if i%int(sys.argv[2])==int(sys.argv[1])]' "$node" "$nodes" > runs/adsb_ctx3/logs/jobs.txt
echo "chain start $(date -u +%FT%TZ) node $node/$nodes lanes $lanes jobs $(wc -l < runs/adsb_ctx3/logs/jobs.txt)" >> runs/adsb_ctx3/logs/chain.log
run_one() {
  v=$1; s=$2
  out=runs/adsb_ctx3/design/${v}_s${s}
  [ -f "$out/design/run.json" ] && { echo "skip ${v}_s${s}" >> runs/adsb_ctx3/logs/chain.log; return 0; }
  echo "start ${v}_s${s} $(date -u +%FT%TZ)" >> runs/adsb_ctx3/logs/chain.log
  if .venv/bin/python -X utf8 tools/run_adsb_ctx3.py --config configs/adsb_ctx3_design_v1.json --variant "$v" --seed "$s" --fold design \
       --output "$out" --device cuda --threads 3 > runs/adsb_ctx3/logs/${v}_s${s}.log 2>&1; then
    echo "done ${v}_s${s} $(date -u +%FT%TZ)" >> runs/adsb_ctx3/logs/chain.log
  else
    echo "FAILED ${v}_s${s} $(date -u +%FT%TZ)" >> runs/adsb_ctx3/logs/chain.log
  fi
}
export -f run_one
xargs -P "$lanes" -L 1 bash -c 'run_one "$0" "$1"' < runs/adsb_ctx3/logs/jobs.txt
echo "chain done $(date -u +%FT%TZ) node $node" >> runs/adsb_ctx3/logs/chain.log
