"""Measure the ADS-B surface features against v9 on the two development folds' scoring months.

Trains a residual LightGBM (label minus v9) on the scoring months that hold
out-of-sample v9 predictions (Jan/Jul 2025 seasonal, Nov 2025 forward), with
leave-one-month-out and day-grouped evaluation, and reports RMSE of v9 + residual
against v9 per month and per airport. Labels are read only for these scoring
months; December is never read. Read-only diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, FEATURE_NAMES, departure_features

MATCHES = ROOT / "data/external/adsblol/matches"
EXTRA = ["airport", "v9", "proxy_s", "schedule_interval_s", "hour", "stand", "runway"]


def month_frame(month: str, fold_frame: pd.DataFrame, stands: pd.DataFrame) -> pd.DataFrame:
    y, m = int(month[:4]), int(month[5:7])
    raw = pd.read_parquet(ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet",
                          columns=DEPARTURE_COLUMNS + ["RUNWAY_mvt"])
    dep = raw[raw[PHASE].eq("DEP")].reset_index(drop=True)
    matches = pd.read_parquet(MATCHES / f"matches2_{month}.parquet")
    feats = departure_features(dep, matches, stands).merge(dep[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID)
    frame = fold_frame.merge(feats, on=ID, how="inner")
    frame["month"] = month
    frame["hour"] = frame[TIME].dt.hour + frame[TIME].dt.minute / 60
    frame = frame.rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("scratch"))
    parser.add_argument("--report", type=Path, default=Path("docs/opensky/adsb_residual_folds.json"))
    parser.add_argument("--threads", type=int, default=14)
    args = parser.parse_args()
    stands = pd.read_parquet(MATCHES / "stands.parquet")
    seasonal = pd.read_parquet(ROOT / args.scratch / "seasonal.parquet")
    forward = pd.read_parquet(ROOT / args.scratch / "forward.parquet")
    frames = []
    for month, fold in [("2025-01", seasonal), ("2025-07", seasonal), ("2025-11", forward)]:
        if not (MATCHES / f"matches2_{month}.parquet").exists():
            print("missing matches for", month)
            continue
        part = fold[fold[TIME].dt.strftime("%Y-%m").eq(month)]
        frames.append(month_frame(month, part, stands))
    data = pd.concat(frames, ignore_index=True)
    for c in ["airport", "stand", "runway"]:
        data[c] = data[c].astype("category")
    cols = EXTRA + FEATURE_NAMES
    y = (data[TARGET] - data["v9"]).to_numpy(dtype=float)
    label = data[TARGET].to_numpy(dtype=float); v9 = data["v9"].to_numpy(dtype=float)
    params = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=10.0, verbose=-1, num_threads=args.threads, seed=20260915)
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    report = {"created_at_utc": utc_now(), "rows": int(len(data)), "months": sorted(data["month"].unique().tolist()), "schemes": {}}
    for scheme in ["leave_one_month_out", "day_grouped_5fold"]:
        pred = np.zeros(len(data))
        if scheme == "leave_one_month_out":
            splits = [(np.flatnonzero(data["month"].ne(mo)), np.flatnonzero(data["month"].eq(mo))) for mo in data["month"].unique()]
        else:
            splits = list(GroupKFold(5).split(data, y, groups=data[TIME].dt.strftime("%Y-%m-%d")))
        for train, valid in splits:
            booster = lgb.train(params, lgb.Dataset(data.iloc[train][cols], y[train]), 4000,
                                valid_sets=[lgb.Dataset(data.iloc[valid][cols], y[valid])], callbacks=[lgb.early_stopping(200, verbose=False)])
            pred[valid] = booster.predict(data.iloc[valid][cols], num_iteration=booster.best_iteration)
        result = {"overall": {"v9": rmse(label, v9), "v9_plus_residual": rmse(label, v9 + pred), "change_pct": 100 * (rmse(label, v9 + pred) / rmse(label, v9) - 1)}, "months": {}, "airports": {}}
        for mo in sorted(data["month"].unique()):
            k = data["month"].eq(mo).to_numpy()
            result["months"][mo] = {"rows": int(k.sum()), "v9": rmse(label[k], v9[k]), "v9_plus_residual": rmse(label[k], v9[k] + pred[k]),
                                    "change_pct": 100 * (rmse(label[k], v9[k] + pred[k]) / rmse(label[k], v9[k]) - 1)}
        for ap in sorted(data["airport"].unique()):
            k = data["airport"].eq(ap).to_numpy(); mk = k & data["adsb_matched"].eq(1).to_numpy()
            result["airports"][ap] = {"rows": int(k.sum()), "matched_share": float(mk.sum() / k.sum()), "gate_share": float(np.nanmean(data.loc[mk, "adsb_gate"])) if mk.sum() else None,
                                      "v9": rmse(label[k], v9[k]), "v9_plus_residual": rmse(label[k], v9[k] + pred[k]),
                                      "change_pct": 100 * (rmse(label[k], v9[k] + pred[k]) / rmse(label[k], v9[k]) - 1)}
        report["schemes"][scheme] = result
        print(scheme, json.dumps({k: round(v, 3) for k, v in result["overall"].items()}))
        for mo, r in result["months"].items():
            print(f"   {mo}: {r['v9']:.3f} -> {r['v9_plus_residual']:.3f} ({r['change_pct']:+.2f}%)")
        for ap, r in result["airports"].items():
            print(f"   {ap}: matched {r['matched_share']:.2f} gate {r['gate_share'] if r['gate_share'] is None else round(r['gate_share'], 2)} | {r['v9']:.2f} -> {r['v9_plus_residual']:.2f} ({r['change_pct']:+.1f}%)")
        data[f"residual_{scheme}"] = pred
    data[[ID, "month", "airport", TARGET, "v9", "residual_leave_one_month_out", "residual_day_grouped_5fold", "adsb_matched", "adsb_gate"]].to_parquet(ROOT / args.scratch / "adsb_residual_predictions.parquet", index=False)
    write_json(ROOT / args.report, report)


if __name__ == "__main__":
    main()
