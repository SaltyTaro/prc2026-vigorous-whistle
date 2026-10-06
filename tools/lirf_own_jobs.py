"""Per-node job lists of the LIRF own-points development round (configs/lirf_own_dev_v1.json): the declared variant(s) x 16 seeds x the
declared folds, the development folds first and the twelve-month refit last; dealt round-robin to NODES nodes. Lines: config variant seed
fold output (tools/combo_queue.sh with ROUND=runs/lirf_own and EXTRA=data/external/adsblol/features_v19own).
Writes runs/lirf_own/jobs/node<i>.txt. Usage: tools/lirf_own_jobs.py [NODES]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["lomo_2025-07", "refit", "forward", "refit12"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nodes", nargs="?", type=int, default=4)
    parser.add_argument("--declaration", default="configs/lirf_own_dev_v1.json")
    args = parser.parse_args()
    decl = json.loads((ROOT / args.declaration).read_text(encoding="utf-8"))
    fits = decl["fits"]
    round_dir = fits["outputs"].split("/nets/")[0]
    folds = sorted(f["fold"] for f in fits["folds"].values())
    if folds != sorted(ORDER):
        raise ValueError(f"The declared folds changed: {folds}")
    jobs = [f"{fits['network_config']} {variant} {seed} {fold} {round_dir}/nets/{variant}_s{seed}"
            for fold in ORDER for seed in fits["seeds"] for variant in fits["variants"]]
    if len(jobs) != fits["count"]:
        raise ValueError("The job count differs from the declaration.")
    out = ROOT / round_dir / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("node*.txt"):
        old.unlink()
    for i in range(args.nodes):
        (out / f"node{i}.txt").write_bytes(("\n".join(jobs[i::args.nodes]) + "\n").encode("utf-8"))
    print(len(jobs), "jobs on", args.nodes, "nodes")


if __name__ == "__main__":
    main()
