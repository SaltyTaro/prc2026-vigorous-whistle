"""Per-node job lists for Tier B's network half (configs/febjun_tierb_v1.json, base.networks_cheap): every declared ensemble x its four seeds x
the held months lomo_2025-06 / lomo_2025-02, 80 fits. Months alternate within each (ensemble, seed) pair so both months finish together; the
list is dealt to NODES nodes with a shift, so every node gets both months. Lines: config variant seed fold output extra ('-' = no --extra-features), read by
tools/febjun_net_queue.sh. Outputs runs/febjun_nets/nets/<variant>_s<seed>/<fold>/ (e24 seeds 117-120 = ens_ctx_e24, 133-136 = ens_ctx_e24b).
Writes runs/febjun_nets/jobs/node<i>.txt. Usage: tools/febjun_net_jobs.py [NODES]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ["lomo_2025-06", "lomo_2025-02"]
OUT = "runs/febjun_nets"


def jobs() -> list[str]:
    decl = json.loads((ROOT / "configs/febjun_tierb_v1.json").read_text(encoding="utf-8"))
    nets = decl["base"]["networks_cheap"]
    out = []
    for ens in nets["ensembles"].values():
        variants = ens.get("variants") or [ens["variant"]]
        extra = ens.get("extra_features") or "-"
        for seed in ens["seeds"]:
            for variant in variants:
                for fold in FOLDS:
                    out.append(f"{ens['config']} {variant} {seed} {fold} {OUT}/nets/{variant}_s{seed} {extra}")
    if len(out) != 80 or len(set(out)) != 80:
        raise ValueError(f"Expected 80 distinct jobs, got {len(out)}.")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nodes", nargs="?", type=int, default=4)
    args = parser.parse_args()
    todo = jobs()
    out = ROOT / OUT / "jobs"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("node*.txt"):
        old.unlink()
    lists = [[] for _ in range(args.nodes)]
    for k, job in enumerate(todo):  # a shifted deal, so every node gets both months and a mix of ensembles
        lists[(k + k // args.nodes) % args.nodes].append(job)
    for i, part in enumerate(lists):
        (out / f"node{i}.txt").write_bytes(("\n".join(part) + "\n").encode("utf-8"))  # bytes: no CRLF on Windows
    print(len(todo), "jobs on", args.nodes, "nodes")


if __name__ == "__main__":
    main()
