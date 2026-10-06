"""Lever 5 (fork E): write the per-node job lists of a ctx_own2 stage for tools/ctx_own2_queue.sh.

design: every run of configs/ctx_own2_design_v1.json ("runs": variant -> seeds) on the October design fold through tools/run_adsb_ctx3.py,
        lines "ctx3 <config> <variant> <seed> design runs/ctx_own2/design/<variant>_s<seed> data/external/adsblol/features_v19own".
dev:    every (variant, seed, fold) of configs/ctx_own2_dev_v1.json through tools/run_adsb_ctx_lomo.py (written only once that declaration exists).
Jobs are dealt round-robin to the nodes in an order that interleaves variants (longest folds first in dev), so each node carries a mix.
Usage: tools/ctx_own2_jobs.py --stage design --nodes 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA = "data/external/adsblol/features_v19own"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["design", "dev"], required=True)
    parser.add_argument("--nodes", type=int, required=True)
    args = parser.parse_args()
    lines = []
    if args.stage == "design":
        config = "configs/ctx_own2_design_v1.json"
        decl = json.loads((ROOT / config).read_text(encoding="utf-8"))
        variants = list(decl["runs"])
        seeds = sorted({s for v in variants for s in decl["runs"][v]})
        for j, seed in enumerate(seeds):
            for v in variants[j % len(variants):] + variants[:j % len(variants)]:  # rotate, so every node gets every variant
                if seed in decl["runs"][v]:
                    lines.append(f"ctx3 {config} {v} {seed} design runs/ctx_own2/design/{v}_s{seed} {EXTRA}")
    else:
        config = "configs/ctx_own2_dev_v1.json"
        decl = json.loads((ROOT / config).read_text(encoding="utf-8"))
        fits = decl["fits"]
        order = ["refit12", "refit", "lomo_2025-01", "lomo_2025-07", "forward"]  # longest fits first
        folds = sorted((x["fold"] for x in fits["folds"].values()), key=order.index)
        for fold in folds:
            for seed in fits["seeds"]:
                for v in fits["variants"]:
                    lines.append(f"lomo {fits['network_config']} {v} {seed} {fold} runs/ctx_own2/nets/{v}_s{seed} {EXTRA}")
    out = ROOT / "runs/ctx_own2" / f"jobs_{args.stage}"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("node*.txt"):
        old.unlink()
    (out / "all.txt").write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")  # for tools/ctx_own2_dispatch.py
    for i in range(args.nodes):
        mine = lines[i::args.nodes]
        (out / f"node{i}.txt").write_text("".join(x + "\n" for x in mine), encoding="utf-8", newline="\n")
        print(f"node{i}: {len(mine)} jobs")
    print(f"{len(lines)} jobs in {out}")


if __name__ == "__main__":
    main()
