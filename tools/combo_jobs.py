"""Per-node job lists for the combination's development round (configs/adsb_ctx3_combo_dev_v1.json): two variants x 16 seeds x five folds.

Order: the four development folds first (July and January leave-one-month-out, the January-November refit, the forward fold), the
twelve-month refit last, so that the development scores can be read before the ranking fits end. Within a fold the jobs alternate
between the two variants and are dealt round-robin to NODES nodes. Lines: config variant seed fold output.
Writes <round>/jobs/node<i>.txt, where <round> is the part of the declaration's fits.outputs before /nets/ (runs/adsb_ctx3_combo for the
combination). Usage: tools/combo_jobs.py [NODES] [--declaration PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["lomo_2025-07", "lomo_2025-01", "refit", "forward", "refit12"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nodes", nargs="?", type=int, default=4)
    parser.add_argument("--declaration", default="configs/adsb_ctx3_combo_dev_v1.json")
    args = parser.parse_args()
    nodes = args.nodes
    decl = json.loads((ROOT / args.declaration).read_text(encoding="utf-8"))
    fits = decl["fits"]
    round_dir = fits["outputs"].split("/nets/")[0]
    if sorted(f["fold"] for f in fits["folds"].values()) != sorted(ORDER):
        raise ValueError("The declared folds changed.")
    jobs = [f"{fits['network_config']} {variant} {seed} {fold} {round_dir}/nets/{variant}_s{seed}"
            for fold in ORDER for seed in fits["seeds"] for variant in fits["variants"]]
    if len(jobs) != fits["count"]:
        raise ValueError("The job count differs from the declaration.")
    out = ROOT / round_dir / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("node*.txt"):
        old.unlink()
    for i in range(nodes):
        (out / f"node{i}.txt").write_bytes(("\n".join(jobs[i::nodes]) + "\n").encode("utf-8"))  # bytes: no CRLF on Windows
    print(len(jobs), "jobs on", nodes, "nodes")


if __name__ == "__main__":
    main()
