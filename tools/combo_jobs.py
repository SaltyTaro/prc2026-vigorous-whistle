"""Per-node job lists for the combination's development round (configs/adsb_ctx3_combo_dev_v1.json): two variants x 16 seeds x five folds.

Order: the four development folds first (July and January leave-one-month-out, the January-November refit, the forward fold), the
twelve-month refit last, so that the development scores can be read before the ranking fits end. Within a fold the jobs alternate
between the two variants and are dealt round-robin to NODES nodes. Lines: config variant seed fold output.
Writes runs/adsb_ctx3_combo/jobs/node<i>.txt. Usage: tools/combo_jobs.py [NODES]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["lomo_2025-07", "lomo_2025-01", "refit", "forward", "refit12"]


def main():
    nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    decl = json.loads((ROOT / "configs/adsb_ctx3_combo_dev_v1.json").read_text(encoding="utf-8"))
    fits = decl["fits"]
    if sorted(f["fold"] for f in fits["folds"].values()) != sorted(ORDER):
        raise ValueError("The declared folds changed.")
    jobs = [f"{fits['network_config']} {variant} {seed} {fold} runs/adsb_ctx3_combo/nets/{variant}_s{seed}"
            for fold in ORDER for seed in fits["seeds"] for variant in fits["variants"]]
    if len(jobs) != fits["count"]:
        raise ValueError("The job count differs from the declaration.")
    out = ROOT / "runs/adsb_ctx3_combo/jobs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("node*.txt"):
        old.unlink()
    for i in range(nodes):
        (out / f"node{i}.txt").write_bytes(("\n".join(jobs[i::nodes]) + "\n").encode("utf-8"))  # bytes: no CRLF on Windows
    print(len(jobs), "jobs on", nodes, "nodes")


if __name__ == "__main__":
    main()
