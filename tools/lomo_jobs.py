"""Per-node job lists for the leave-one-month-out network folds (configs/lomo_dev_networks_v1.json): the 64 shipped network seeds of v34's
fold side (ctx_e24 117-148 = ens_ctx_e24 + ens_ctx_e24b, ctx_e32 401-416, ctx_lirf 301-316) x held months 2025-07 and 2025-01, spread over
the four nodes (prc-ctx3a..d = node 0..3) round-robin, July first. Lines: config variant seed fold output. Writes runs/lomo_dev/jobs/node<i>.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    decl = json.loads((ROOT / "configs/lomo_dev_networks_v1.json").read_text(encoding="utf-8"))
    jobs = []
    for month in decl["held_months"]:
        for family in decl["families"]:
            for seed in family["seeds"]:
                jobs.append(f"{decl['network_config']} {family['variant']} {seed} lomo_{month} runs/lomo_dev/nets/{family['variant']}_s{seed}")
    out = ROOT / "runs/lomo_dev/jobs"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        (out / f"node{i}.txt").write_text("\n".join(jobs[i::4]) + "\n", encoding="utf-8")
    print(len(jobs), "jobs")


if __name__ == "__main__":
    main()
