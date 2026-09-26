"""Per-node job lists for the GPU work declared on 26 September evening (four nodes, prc-ctx3a..d = node 0..3):
the 32-epoch development round (configs/adsb_ctx_v4.json: ens_ctx_e32, seeds 401-416, folds seasonal/forward/refit/refit12 by
configs/adsb_ctx_network_v3.json) and the sprint's second design batch (configs/adsb_ctx3_design_v2.json, seeds 5-8).

Node i gets development seeds 401+i, 405+i, 409+i, 413+i and design seed 5+i of every batch-2 variant, in this priority order: the
seasonal, forward and refit folds of its development seeds (so development scoring can start early), then its batch-2 design runs, then
its refit12 folds. Lines: config variant seed fold output. Writes runs/adsb_ctx3/jobs/node<i>.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    dev = json.loads((ROOT / "configs/adsb_ctx_v4.json").read_text(encoding="utf-8"))["networks"]["ens_ctx_e32"]
    design = json.loads((ROOT / "configs/adsb_ctx3_design_v2.json").read_text(encoding="utf-8"))
    out = ROOT / "runs/adsb_ctx3/jobs"
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for i in range(4):
        seeds = [s for k, s in enumerate(dev["seeds"]) if k % 4 == i]
        lines = [f"{dev['config']} {dev['variant']} {s} {fold} runs/adsb_ctx_gpu/ctx_e32_s{s}" for fold in ("seasonal", "forward", "refit") for s in seeds]
        lines += [f"configs/adsb_ctx3_design_v2.json {v} {5 + i} design runs/adsb_ctx3/design2/{v}_s{5 + i}" for v in sorted(design["runs"])]
        lines += [f"{dev['config']} {dev['variant']} {s} refit12 runs/adsb_ctx_gpu/ctx_e32_s{s}" for s in seeds]
        (out / f"node{i}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        total += len(lines)
    print(total, "jobs")


if __name__ == "__main__":
    main()
