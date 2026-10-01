"""Per-node job lists of the LIRF masked-member round (configs/adsb_ctx3_lirf_mask_dev_v1.json): one variant x 16 seeds x six folds.

tools/combo_jobs.py's format (lines: config variant seed fold output) and dealing (round-robin over NODES), with the fold order taken
from the declaration's fits.order (the seasonal fold first: its thinned July re-prediction is the round's main reading; the twelve-month
refit last). Writes <round>/jobs/node<i>.txt. Usage: tools/lirf_mask_jobs.py [NODES] [--declaration PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nodes", nargs="?", type=int, default=4)
    parser.add_argument("--declaration", default="configs/adsb_ctx3_lirf_mask_dev_v1.json")
    parser.add_argument("--exclude-folds", nargs="*", default=[], help="folds run elsewhere (declared in the amendment)")
    args = parser.parse_args()
    decl = json.loads((ROOT / args.declaration).read_text(encoding="utf-8"))
    if not decl.get("declared_at_utc"):
        raise ValueError("The declaration is a draft.")
    fits = decl["fits"]
    round_dir = fits["outputs"].split("/nets/")[0]
    if sorted(fits["order"]) != sorted(f["fold"] for f in fits["folds"].values()):
        raise ValueError("fits.order and fits.folds disagree.")
    jobs = [f"{fits['network_config']} {variant} {seed} {fold} {round_dir}/nets/{variant}_s{seed}"
            for fold in fits["order"] if fold not in args.exclude_folds for seed in fits["seeds"] for variant in fits["variants"]]
    expected = fits["count"] * (len(fits["order"]) - len(set(args.exclude_folds) & set(fits["order"]))) // len(fits["order"])
    if len(jobs) != expected:
        raise ValueError("The job count differs from the declaration.")
    out = ROOT / round_dir / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("node*.txt"):
        old.unlink()
    for i in range(args.nodes):
        (out / f"node{i}.txt").write_bytes(("\n".join(jobs[i::args.nodes]) + "\n").encode("utf-8"))  # bytes: no CRLF on Windows
    print(len(jobs), "jobs on", args.nodes, "nodes")


if __name__ == "__main__":
    main()
