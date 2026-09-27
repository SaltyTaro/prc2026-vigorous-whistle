"""Build the leave-one-month-out tree tables (configs/lomo_dev_v1.json): ens_v20_bag_q_all and ens_v22_lirf_q for January and July 2025
from the LOMO leaves (runs/lomo_dev/learners/<leaf name>/lomo_<month>/predictions.parquet) with the tables' own member rule
(prc2026_december.scoring.leaf_members + ensemble_prediction, nanmean with multiplicities).

Guards: the same rule applied to the leaves' SEASONAL predictions must reproduce each stored seasonal table (1e-9 s); every LOMO leaf
must predict exactly the seasonal rows of its month, with the same finite pattern as its seasonal predictions. Output in the seasonal
predictions' row order: runs/lomo_dev/ens/<table>/lomo/predictions.parquet (MVT_ID_mvt, adsb_full) and docs/lomo_dev/tables.json
(label-free: LOMO minus seasonal per table and month). Nothing is fitted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TIME  # noqa: E402
from prc2026_december.scoring import ensemble_prediction, leaf_members  # noqa: E402

TABLES = {"ens_v20_bag_q_all": "runs/adsb_v20/ens_v20_bag_q_all", "ens_v22_lirf_q": "runs/adsb_v22/ens_v22_lirf_q"}
MONTHS = ["2025-01", "2025-07"]
OUT = ROOT / "runs/lomo_dev/ens"


def name_of(leaf: str) -> str:
    return leaf.replace("runs/", "").replace("/", "__")


def main() -> None:
    frame = pd.read_parquet(ROOT / "scratch/seasonal.parquet", columns=[ID, TIME])  # the seasonal fold's January + July rows
    month_of = frame.set_index(ID)[TIME].dt.strftime("%Y-%m")
    report = {"created_at_utc": utc_now(), "declaration": "configs/lomo_dev_v1.json", "declaration_sha256": sha256_file(ROOT / "configs/lomo_dev_v1.json"), "tables": {}}
    for table, run_dir in TABLES.items():
        stored = pd.read_parquet(ROOT / run_dir / "seasonal" / "predictions.parquet")
        ids = stored[ID]
        month = month_of.reindex(ids).to_numpy()
        if pd.isna(month).any():
            raise ValueError(f"{table}: a seasonal row is not in the seasonal fold frame.")
        members = leaf_members(ROOT, run_dir)
        seasonal, lomo, inputs = {}, {}, {}
        for leaf in sorted(set(members)):
            seasonal[leaf] = pd.read_parquet(ROOT / leaf / "seasonal" / "predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
            parts = []
            for m in MONTHS:
                path = ROOT / "runs/lomo_dev/learners" / name_of(leaf) / f"lomo_{m}" / "predictions.parquet"
                part = pd.read_parquet(path)
                if set(part[ID]) != set(ids[month == m]):
                    raise ValueError(f"{leaf} {m}: the LOMO rows differ from the seasonal rows of the month.")
                parts.append(part)
                inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
            values = pd.concat(parts, ignore_index=True).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
            if not (np.isfinite(values) == np.isfinite(seasonal[leaf])).all():
                raise ValueError(f"{leaf}: the LOMO finite pattern differs from the seasonal one.")
            lomo[leaf] = values
        rebuilt = ensemble_prediction(members, seasonal)
        ref = stored["adsb_full"].to_numpy(dtype=float)
        check = float(np.nanmax(np.abs(rebuilt - ref)))
        if not check <= 1e-9 or not (np.isfinite(rebuilt) == np.isfinite(ref)).all():
            raise ValueError(f"{table}: the member rule does not reproduce the stored seasonal table (max {check}).")
        values = ensemble_prediction(members, lomo)
        target = OUT / table / "lomo" / "predictions.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: ids.to_numpy(), "adsb_full": values}).to_parquet(target, index=False)
        diff = values - ref
        report["tables"][table] = {"run_dir": run_dir, "members": len(members), "distinct_leaves": len(set(members)), "seasonal_rebuild_max_abs": check,
                                   "output": target.relative_to(ROOT).as_posix(), "sha256": sha256_file(target),
                                   "lomo_minus_seasonal": {m: {"rows": int((month == m).sum()), "finite": int(np.isfinite(values[month == m]).sum()),
                                                               "mean_s": float(np.nanmean(diff[month == m])), "rms_s": float(np.sqrt(np.nanmean(diff[month == m] ** 2))),
                                                               "p1_s": float(np.nanquantile(diff[month == m], 0.01)), "p99_s": float(np.nanquantile(diff[month == m], 0.99))} for m in MONTHS},
                                   "inputs": inputs}
        print(table, {m: {k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()} for m, e in report["tables"][table]["lomo_minus_seasonal"].items()})
    write_json(ROOT / "docs/lomo_dev/tables.json", report)


if __name__ == "__main__":
    main()
