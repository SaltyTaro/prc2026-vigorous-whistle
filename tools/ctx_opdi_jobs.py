"""Lever 9 (fork J): write the job list of a ctx_opdi stage for tools/ctx_opdi_dispatch.py (a copy of tools/ctx_own2_jobs.py's logic).

design: every run of configs/ctx_opdi_design_v1.json ("runs": variant -> seeds; the control runs are lever 5's existing GPU fits and are not
        listed) on the October design fold through tools/run_adsb_ctx3.py, lines
        "ctx3 <config> <variant> <seed> design runs/ctx_opdi/design/<variant>_s<seed> data/external/opdi/features_v19ownopdi".
dev:    every (variant, seed, fold) of configs/ctx_opdi_dev_v1.json through tools/run_adsb_ctx_lomo.py (only once that declaration exists).
Usage: tools/ctx_opdi_jobs.py --stage design
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA = "data/external/opdi/features_v19ownopdi"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["design", "dev"], required=True)
    args = parser.parse_args()
    lines = []
    if args.stage == "design":
        config = "configs/ctx_opdi_design_v1.json"
        decl = json.loads((ROOT / config).read_text(encoding="utf-8"))
        variants = list(decl["runs"])
        seeds = sorted({s for v in variants for s in decl["runs"][v]})
        for seed in seeds:
            for v in variants:
                if seed in decl["runs"][v]:
                    lines.append(f"ctx3 {config} {v} {seed} design runs/ctx_opdi/design/{v}_s{seed} {EXTRA}")
    else:
        config = "configs/ctx_opdi_dev_v1.json"
        decl = json.loads((ROOT / config).read_text(encoding="utf-8"))
        fits = decl["fits"]
        order = ["refit12", "refit", "lomo_2025-01", "lomo_2025-07", "forward"]  # longest fits first
        folds = sorted((x["fold"] for x in fits["folds"].values()), key=order.index)
        for fold in folds:
            for seed in fits["seeds"]:
                for v in fits["variants"]:
                    lines.append(f"lomo {fits['network_config']} {v} {seed} {fold} runs/ctx_opdi/nets/{v}_s{seed} {EXTRA}")
    out = ROOT / "runs/ctx_opdi" / f"jobs_{args.stage}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "all.txt").write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    print(f"{len(lines)} jobs in {out}")


if __name__ == "__main__":
    main()
