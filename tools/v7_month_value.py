"""Month value of v7's twelve timeline components on December 2025 (read-only diagnostic; the learners' analog is docs/december/month_value.json).

Applies the seasonal-fold and forward-fold v7 components (the frozen development runs: cat_combined / lgb_combined globals and the ten
airport LightGBMs, all early-stopped on October) to December's clock-present departures and compares them with the shipped
January-November refit (scratch/december.parquet, column v7). The forward fold was fitted on January-September; the refit on
January-November: the difference is what two adjacent months bought v7 on December, out-of-sample for every model involved.
Rows: clock-present December departures; LIRF population rows are missing-clock and outside this read. Nothing is fitted or uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET
from prc2026_nmclock.features import route_mask
from prc2026_surface_airports.inference import airport_prediction
from prc2026_surface_refit.inference import WEIGHTS
from prc2026_timeline.features import FEATURE_NAMES
from prc2026_timeline.inference import node_prediction

TIMELINE = ROOT / "runs/azure_arcw0_timeline"
AIRPORTS = ROOT / "runs/azure_arcw0_surface_airports"
DECEMBER_FILE = "training_2025-12-01_2026-01-01.parquet"


def fold_runs(fold: str) -> dict:
    runs = {}
    for variant in ["cat_combined", "lgb_combined"]:
        matches = sorted(d for d in TIMELINE.iterdir() if d.name.endswith(f"_{fold}_{variant}") and (d / "node.json").exists())
        if len(matches) != 1:
            raise ValueError(f"Expected one {fold} {variant} run, found {len(matches)}.")
        runs[variant] = matches[0]
    matches = sorted(d for d in AIRPORTS.iterdir() if d.name.endswith(f"_{fold}_surface_airports") and (d / "nodes.json").exists())
    if len(matches) != 1:
        raise ValueError(f"Expected one {fold} airport run, found {len(matches)}.")
    runs["airports"] = matches[0]
    return runs


def fold_prediction(fold: str, features: pd.DataFrame, threads: int) -> tuple[np.ndarray, dict]:
    runs = fold_runs(fold)
    values = np.zeros(len(features), dtype=float)
    receipt = {}
    for name, weight in WEIGHTS.items():
        if name == "surface_airports":
            nodes = read_json(runs["airports"] / "nodes.json")
            component = airport_prediction(runs["airports"], nodes, features, threads)
            receipt[name] = {"run": runs["airports"].relative_to(ROOT).as_posix(), "run_sha256": sha256_file(runs["airports"] / "run.json"), "iterations": {a: n["iterations"] for a, n in nodes.items()}}
        else:
            node = read_json(runs[name] / "node.json")
            component = node_prediction(runs[name], node, features, threads)
            receipt[name] = {"run": runs[name].relative_to(ROOT).as_posix(), "run_sha256": sha256_file(runs[name] / "run.json"), "iterations": node["iterations"]}
        values += weight * component
    return values, receipt


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--december", type=Path, default=Path("data/december/d198dea10625b0a75977"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v24/v7_month_value.json"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    features = pd.read_parquet(ROOT / args.december / "features" / DECEMBER_FILE, columns=[ID, *FEATURE_NAMES])
    scratch = pd.read_parquet(ROOT / "scratch/december.parquet").set_index(ID).reindex(features[ID])
    if scratch["v7"].isna().any() or scratch[TARGET].isna().any():
        raise ValueError("scratch/december.parquet does not cover the December build.")
    clock = route_mask(features)
    rows = features.loc[clock].reset_index(drop=True)
    label = scratch[TARGET].to_numpy(dtype=float)[clock]
    airport = scratch["airport"].to_numpy()[clock]
    floor = float(scratch["floor"].iloc[0])
    refit = np.maximum(scratch["v7"].to_numpy(dtype=float)[clock], floor)
    report = {"created_at_utc": utc_now(), "december_build_id": read_json(ROOT / args.december / "manifest.json")["build_id"], "clock_rows": int(clock.sum()), "floor_seconds": floor,
              "refit": {"source": "scratch/december.parquet v7 (runs/azure_arcw0_surface_refit, January-November)", "rmse": rmse(label, refit),
                        "by_airport": {a: rmse(label[airport == a], refit[airport == a]) for a in sorted(set(airport))}}, "folds": {}}
    for fold, months in [("forward", "January-September, October stopping"), ("seasonal", "February-June and August-September, October stopping")]:
        values, receipt = fold_prediction(fold, rows, args.threads)
        values = np.maximum(values, floor)
        entry = {"fit_months": months, "rmse": rmse(label, values), "refit_vs_fold_pct": 100 * (report["refit"]["rmse"] / rmse(label, values) - 1),
                 "by_airport": {a: {"fold": rmse(label[airport == a], values[airport == a]), "refit": report["refit"]["by_airport"][a]} for a in sorted(set(airport))}, "components": receipt}
        report["folds"][fold] = entry
        print(fold, {"fold_rmse": round(entry["rmse"], 4), "refit_rmse": round(report["refit"]["rmse"], 4), "refit_vs_fold_pct": round(entry["refit_vs_fold_pct"], 4)}, flush=True)
        write_json(ROOT / args.report, report)
    write_json(ROOT / args.report, report)


if __name__ == "__main__":
    main()
