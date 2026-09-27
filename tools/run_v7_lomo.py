"""Leave-one-month-out v7 for the development months (configs/lomo_dev_v1.json): tools/run_v7_dec.py's frozen twelve-component fit
(0.25 CatBoost global + 0.25 LightGBM global + 0.5 airport LightGBM, NM-residual targets, the frozen 276 predictors, the same specs,
settings, seed and October-selected counts) with ONE held month instead of a cross-fit fold of three.

run_v7_dec.py is not edited: this wrapper adds the single-month folds to its FOLDS table and calls its main() unchanged, so the fitting,
held-out prediction and reload checks are byte-for-byte the cross-fit code (population: the frozen January-November refit rows plus the
December build, the held month's rows removed before fitting; every departure of the held month predicted, NaN on missing-clock rows).

  --fold L01 | L07    held month 2025-01 / 2025-07  -> <output>/<fold>/{nodes, predictions.parquet, run.json} + lomo.json (wrapper receipt)
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_v7_dec  # noqa: E402

LOMO_FOLDS = {"L01": ["2025-01"], "L07": ["2025-07"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fold", required=True, choices=sorted(LOMO_FOLDS))
    parser.add_argument("--output", type=Path, default=Path("runs/lomo_dev/v7"))
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if set(LOMO_FOLDS) & set(run_v7_dec.FOLDS):
        raise ValueError("A single-month fold name collides with run_v7_dec's folds.")
    run_v7_dec.FOLDS.update(LOMO_FOLDS)
    sys.argv = ["run_v7_dec.py", "--fold", args.fold, "--output", args.output.as_posix(), *(["--threads", str(args.threads)] if args.threads else [])]
    run_v7_dec.main()
    run = ROOT / args.output / args.fold
    record = read_json(run / "run.json")
    if record["held_out_months"] != LOMO_FOLDS[args.fold] or LOMO_FOLDS[args.fold][0] in record["fit_months"]:
        raise ValueError("The held month is in the fit.")
    write_json(run / "lomo.json", {"created_at_utc": utc_now(), "declaration": "configs/lomo_dev_v1.json", "fold": args.fold, "held_month": LOMO_FOLDS[args.fold][0],
                                   "fit_months": record["fit_months"], "wrapper_sha256": sha256_file(ROOT / "tools/run_v7_lomo.py"),
                                   "run_v7_dec_sha256": sha256_file(ROOT / "tools/run_v7_dec.py"), "run_sha256": sha256_file(run / "run.json")})


if __name__ == "__main__":
    main()
