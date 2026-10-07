"""Lever 12 (fork N): write the job list of the LFPG network design for tools/lfpg_net_dispatch.py (tools/ctx_opdi_jobs.py's design branch).

design: every run of configs/lfpg_net_design_v1.json ("runs": variant -> seeds; the control runs are lever 5's existing GPU fits and are not
        listed) on the October design fold through tools/run_adsb_ctx3.py, lines
        "ctx3 <config> <variant> <seed> design runs/lfpg_net/design/<variant>_s<seed> data/external/adsblol/features_v19own".
Usage: tools/lfpg_net_jobs.py --stage design
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRA = "data/external/adsblol/features_v19own"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["design"], required=True)
    args = parser.parse_args()
    config = "configs/lfpg_net_design_v1.json"
    decl = json.loads((ROOT / config).read_text(encoding="utf-8"))
    variants = list(decl["runs"])
    seeds = sorted({s for v in variants for s in decl["runs"][v]})
    lines = [f"ctx3 {config} {v} {seed} design runs/lfpg_net/design/{v}_s{seed} {EXTRA}"
             for seed in seeds for v in variants if seed in decl["runs"][v]]
    out = ROOT / "runs/lfpg_net" / f"jobs_{args.stage}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "all.txt").write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    print(f"{len(lines)} jobs in {out}")


if __name__ == "__main__":
    main()
