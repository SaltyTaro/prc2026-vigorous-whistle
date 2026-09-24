"""Declared development of adsb_nn3: a LIRF-inclusive network on the NM-clock residual target (nnlr) in the LIRF blends, with v25 as the control.

No model is fitted here. adsb_v30 showed that v25's LIRF network (nnl) cannot take more LIRF weight because it fails on long-clock
LIRF rows. This round adds a new ensemble (ens_nnlr: 16 seeds of nnl on the NM-clock residual target) and re-reads the LIRF blends with
it, alone or averaged with ens_nnl, using tools/run_adsb_nn2_dev.py's scoring code (imported unchanged; every table, ensemble check and
the rule are adsb_nn2's).

The control is adsb_nn2's k32_m30_u25 and must reproduce its recorded development RMSE (runs/adsb_nn2/dev/selection.json) on every
month to 1e-9 s.

  --score    every candidate on the four months (all rows), with paired-day intervals against the control
  --select   the pre-registered rule
  --smoke    with --score: every candidate replaced by the control (all collapse onto it) and the new ensemble dropped, into a
             scratch output -- tests the code path and the control reproduction without producing any reading (possible before any fit)
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json  # noqa: E402
from run_adsb_nn2_dev import score, select  # noqa: E402
from run_adsb_v11 import check_parents  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_nn3/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_nn3/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    nn2 = read_json(ROOT / "configs/adsb_nn2_v1.json")
    control = config["candidates"][config["control_candidate"]]
    if control != nn2["candidates"]["k32_m30_u25"]:
        raise ValueError("The control must be the shipped v25 recipe (adsb_nn2 k32_m30_u25).")
    if args.smoke:
        if args.select or not args.score or (ROOT / args.output).resolve() == (ROOT / "runs/adsb_nn3/dev").resolve():
            raise ValueError("--smoke scores only, into a scratch output directory.")
        # every candidate collapses onto the control; the new ensemble (not yet fitted) is dropped from the tables and checks
        config = {**config, "candidates": {name: control for name in config["candidates"]},
                  "extra_feature_tables": {k: v for k, v in config["extra_feature_tables"].items() if k != "ens_nnlr"},
                  "ensembles": {k: v for k, v in config["ensembles"].items() if k != "ens_nnlr"}}
    if args.score:
        score(config, ROOT / args.output, ROOT / args.cache)
    if args.select:
        select(config, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
