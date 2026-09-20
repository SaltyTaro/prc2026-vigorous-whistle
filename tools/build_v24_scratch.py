"""Twelve-month scratch tables for the ADS-B residual stage from the cross-fitted v7 (adsb_v24).

For every 2025 month, the cross-fit fold that held the month out (tools/run_v7_dec.py --fold A|B|C|D) provides an out-of-sample v7
for each clock-present departure. This tool writes scratch/v24/<month>.parquet in the fold-table layout that the residual tools read
([ID, TIME, airport, label, v9, floor, proxy_s, schedule_interval_s, clock_available, population] plus v9_source):

- the eight months without a development fold (02-06, 08-10): clock-present rows only, v9 = max(cross-fit v7, floor); the LIRF fill
  population is missing-clock by definition and never appears here;
- the four development months (01, 07, 11, 12): every row of the existing scratch table (fold v9, kept as v9_fold) with v9 replaced on
  clock-present rows by max(cross-fit v7, floor); missing-clock rows (v6 fallback) and the population keep the fold value.

Labels come from the raw movement files (development labels of 2025, all already used by shipped selections). Nothing is fitted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_postprocess.rules import row_context
from run_lirf_mixture import MONTH_FILES, MOVEMENT_COLUMNS
from run_v7_dec import FOLDS

DEVELOPMENT = {"2025-01": "seasonal", "2025-07": "seasonal", "2025-11": "forward", "2025-12": "december"}
COLUMNS = [ID, TIME, "airport", TARGET, "v9", "floor", "proxy_s", "schedule_interval_s", "clock_available", "population"]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7", type=Path, default=Path("runs/adsb_v24/v7"))
    parser.add_argument("--scratch", type=Path, default=Path("scratch"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v24/scratch_build.json"))
    args = parser.parse_args()
    out = ROOT / args.scratch / "v24"
    out.mkdir(parents=True, exist_ok=True)
    fold_of = {month: fold for fold, months in FOLDS.items() for month in months}
    predictions, fold_records = {}, {}
    for fold in ["A", "B", "C", "D"]:
        run = ROOT / args.v7 / fold
        record = read_json(run / "run.json")
        if record["held_out_months"] != FOLDS[fold] or record["stage"] != "cross_fit":
            raise ValueError(f"Fold {fold} run does not match the declared months.")
        frame = pd.read_parquet(run / "predictions.parquet")
        for month, part in frame.groupby("month"):
            predictions[month] = part.set_index(ID)["prediction"]
        fold_records[fold] = {"run_sha256": sha256_file(run / "run.json"), "predictions_sha256": sha256_file(run / "predictions.parquet"), "clock_fit_rows": record["clock_fit_rows"], "code_sha256": record["code"]["sha256"]}
    floors = {name: float(pd.read_parquet(ROOT / args.scratch / f"{name}.parquet", columns=["floor"])["floor"].iloc[0]) for name in ["seasonal", "forward", "december"]}
    if len(set(floors.values())) != 1:
        raise ValueError("Scratch floors disagree.")
    floor = next(iter(floors.values()))
    report = {"created_at_utc": utc_now(), "floor_seconds": floor, "folds": fold_records, "months": {}}
    for month in [f"2025-{m:02}" for m in range(1, 13)]:
        pred = predictions[month]
        if month in DEVELOPMENT:
            base = pd.read_parquet(ROOT / args.scratch / f"{DEVELOPMENT[month]}.parquet")
            base = base[base[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
            clock = base["clock_available"].to_numpy(dtype=bool)
            cross = pred.reindex(base[ID]).to_numpy(dtype=float)
            if np.isnan(cross[clock]).any() or np.isfinite(cross[~clock]).any():
                raise ValueError(f"Cross-fit predictions do not cover exactly the clock-present rows of {month}.")
            table = base.copy()
            table["v9_fold"] = base["v9"].to_numpy(dtype=float)
            table["v9"] = np.where(clock, np.maximum(cross, floor), base["v9"].to_numpy(dtype=float))
            table["v9_source"] = np.where(clock, "cross_fit_v7", "fold")
            label = table[TARGET].to_numpy(dtype=float)
            entry = {"rows": int(len(table)), "clock_rows": int(clock.sum()), "population_rows": int(table["population"].sum()), "source_frame": DEVELOPMENT[month],
                     "clock_rmse_fold_v9": rmse(label[clock], table["v9_fold"].to_numpy()[clock]), "clock_rmse_cross_fit_v9": rmse(label[clock], table["v9"].to_numpy()[clock])}
            entry["clock_change_pct"] = 100 * (entry["clock_rmse_cross_fit_v9"] / entry["clock_rmse_fold_v9"] - 1)
        else:
            raw = pd.read_parquet(MONTH_FILES[month], columns=MOVEMENT_COLUMNS)
            raw = raw[raw["PHASE_mvt"].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
            context = row_context(raw).set_index(ID).reindex(raw[ID])
            clock = context["clock_available"].to_numpy(dtype=bool)
            rows = raw.loc[clock].reset_index(drop=True)
            ctx = context.loc[clock]
            cross = pred.reindex(rows[ID]).to_numpy(dtype=float)
            finite = pred[np.isfinite(pred.to_numpy(dtype=float))]
            if np.isnan(cross).any() or set(finite.index) != set(rows[ID]):  # the fold predicts the catalog's rows of the month; one catalog row can sit in another month's raw file
                raise ValueError(f"Cross-fit predictions do not cover exactly the clock-present departures of {month}.")
            if ctx["hedge_population"].any():
                raise ValueError("A clock-present row cannot belong to the fill population.")
            table = pd.DataFrame({ID: rows[ID].to_numpy(), TIME: rows[TIME].to_numpy(), "airport": rows["ADEP_mvt"].to_numpy(), TARGET: rows[TARGET].to_numpy(dtype=float),
                                  "v9": np.maximum(cross, floor), "floor": floor, "proxy_s": ctx["proxy_s"].to_numpy(), "schedule_interval_s": ctx["schedule_interval_s"].to_numpy(),
                                  "clock_available": True, "population": False, "v9_fold": np.nan, "v9_source": "cross_fit_v7"})
            if not np.isfinite(table[TARGET]).all():
                raise ValueError(f"Missing labels in {month}.")
            entry = {"rows": int(len(table)), "clock_rows": int(len(table)), "departures_in_month": int(len(raw)), "missing_clock_rows_excluded": int((~clock).sum()),
                     "clock_rmse_cross_fit_v9": rmse(table[TARGET].to_numpy(), table["v9"].to_numpy())}
        table = table[COLUMNS + ["v9_fold", "v9_source"]]
        table.to_parquet(out / f"{month}.parquet", index=False)
        entry.update({"fold": fold_of[month], "ids_sha256": identity_hash(table[ID]), "sha256": sha256_file(out / f"{month}.parquet")})
        report["months"][month] = entry
        print(month, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in entry.items() if k not in ("ids_sha256", "sha256")}, flush=True)
    write_json(ROOT / args.report, report)


if __name__ == "__main__":
    main()
