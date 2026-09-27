"""LOMO v9 for January and July 2025 (configs/lomo_dev_v1.json, v9 side): the held month's rows of the seasonal fold frame
(scratch/seasonal.parquet) with v9 = max(LOMO v7, floor) on clock-present rows (runs/lomo_dev/v7/L01|L07/predictions.parquet, from
tools/run_v7_lomo.py); missing-clock rows (v6 fallback) and the LIRF fill population keep the seasonal fold v9 -- the construction of
tools/build_v24_scratch.py for the development months. Writes runs/lomo_dev/v9/<month>.parquet in the scratch layout ([ID, TIME, airport,
label, v9, floor, proxy_s, schedule_interval_s, clock_available, population] + v9_fold, v9_source) and docs/lomo_dev/v9.json, which also
reports the clock-present RMSE of the fold, cross-fit (scratch/v24) and LOMO v9 on the month (development labels). Nothing is fitted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402

FOLD = {"2025-01": "L01", "2025-07": "L07"}
COLUMNS = [ID, TIME, "airport", TARGET, "v9", "floor", "proxy_s", "schedule_interval_s", "clock_available", "population"]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    frame = pd.read_parquet(ROOT / "scratch/seasonal.parquet")
    report = {"created_at_utc": utc_now(), "declaration": "configs/lomo_dev_v1.json", "declaration_sha256": sha256_file(ROOT / "configs/lomo_dev_v1.json"), "months": {}}
    out = ROOT / "runs/lomo_dev/v9"
    out.mkdir(parents=True, exist_ok=True)
    for month, fold in FOLD.items():
        run = ROOT / "runs/lomo_dev/v7" / fold
        record = read_json(run / "run.json")
        if record["held_out_months"] != [month] or month in record["fit_months"]:
            raise ValueError(f"{fold}: the v7 run does not hold out {month}.")
        v7 = pd.read_parquet(run / "predictions.parquet").set_index(ID)["prediction"]
        base = frame[frame[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
        floor = float(base["floor"].iloc[0])
        clock = base["clock_available"].to_numpy(dtype=bool)
        lomo = v7.reindex(base[ID]).to_numpy(dtype=float)
        if np.isnan(lomo[clock]).any() or np.isfinite(lomo[~clock]).any():
            raise ValueError(f"LOMO v7 does not cover exactly the clock-present rows of {month}.")
        table = base.copy()
        table["v9_fold"] = base["v9"].to_numpy(dtype=float)
        table["v9"] = np.where(clock, np.maximum(lomo, floor), base["v9"].to_numpy(dtype=float))
        table["v9_source"] = np.where(clock, "lomo_v7", "fold")
        table = table[COLUMNS + ["v9_fold", "v9_source"]]
        table.to_parquet(out / f"{month}.parquet", index=False)
        cross = pd.read_parquet(ROOT / "scratch/v24" / f"{month}.parquet").set_index(ID)["v9"].reindex(base[ID]).to_numpy(dtype=float)
        label = table[TARGET].to_numpy(dtype=float)
        report["months"][month] = {"rows": int(len(table)), "clock_rows": int(clock.sum()), "population_rows": int(table["population"].sum()), "floor_seconds": floor,
                                   "v7_run_sha256": sha256_file(run / "run.json"), "v7_predictions_sha256": sha256_file(run / "predictions.parquet"),
                                   "clock_rmse": {"fold_v9": rmse(label[clock], table["v9_fold"].to_numpy()[clock]), "cross_fit_v9": rmse(label[clock], cross[clock]),
                                                  "lomo_v9": rmse(label[clock], table["v9"].to_numpy()[clock])},
                                   "ids_sha256": identity_hash(table[ID]), "sha256": sha256_file(out / f"{month}.parquet")}
        print(month, report["months"][month]["clock_rmse"])
    write_json(ROOT / "docs/lomo_dev/v9.json", report)


if __name__ == "__main__":
    main()
