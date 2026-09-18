"""The adsb_v19 screen (docs/adsb_v19_screen/README.md): does a second-stage model on the incumbent's
development residual gain from the new surface features?

Rows: the development months' departures outside LIRF. Target: label minus the incumbent's
development prediction. Leave-one-month-out over Jan/Jul/Nov 2025 with fixed LightGBM
settings; two feature sets (base, new). Fitting rows are restricted to labels of at most
43,200 s (the garbage one-day rows would otherwise dominate an L2 fit); every row is
evaluated. Writes a JSON report; nothing else.

    python tools/screen_adsb_v19.py --report docs/adsb_v19_screen/screen_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from prc2026.features import ID, TIME  # noqa: E402
from prc2026_adsb.features import FEATURE_NAMES, NEIGHBOUR_NAMES  # noqa: E402
from prc2026_adsb.features_v19 import FEATURE_NAMES_V19  # noqa: E402
from audit_export import DEV_MONTHS, development_reference  # noqa: E402
from run_adsb_v11 import month_table  # noqa: E402

MATCHES_V11 = ROOT / "data/external/adsblol/matches_v11"  # the incumbent's frozen match tables (configs/adsb_v17_v1.json)

CONTEXT = ["hour", "weekday", "schedule_interval_s", "proxy_s", "clock_available"]
CATEGORICAL = ["airport", "runway"]
PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8, "lambda_l2": 10.0,
          "bagging_fraction": 0.9, "bagging_freq": 1, "seed": 20260918, "verbose": -1, "num_threads": 12}
ROUNDS = 400
FIT_LABEL_CAP = 43200.0


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def build_table(learner: str, residual_run: str, features_dir: Path) -> pd.DataFrame:
    dev = development_reference(learner, residual_run)
    stands = pd.read_parquet(MATCHES_V11 / "stands.parquet")
    parts = []
    for month in DEV_MONTHS:
        fold = dev[dev["month"].eq(month)][[ID, TIME, "airport", "label", "pred", "v9", "gate", "adsb_taxi", "proxy_s", "schedule_interval_s", "clock_available", "population", "day"]]
        table = month_table(month, fold, stands, matches_root=MATCHES_V11, neighbours=True)
        extra = pd.read_parquet(features_dir / f"features_v19_{month}.parquet")
        table = table.merge(extra, on=ID, how="left", validate="one_to_one")
        parts.append(table)
    t = pd.concat(parts, ignore_index=True)
    t["weekday"] = t[TIME].dt.weekday
    t["clock_available"] = t["clock_available"].astype(float)
    t["kind"] = np.where(t["gate"].isna(), "unmatched", np.where(t["gate"].fillna(False).astype(bool), "gate", "dark"))
    t["residual"] = t["label"] - t["pred"]
    for c in CATEGORICAL:
        t[c] = t[c].astype("category")
    return t[t["airport"].ne("LIRF")].reset_index(drop=True)


def lomo(table: pd.DataFrame, features: list[str]) -> np.ndarray:
    out = np.full(len(table), np.nan)
    for month in DEV_MONTHS:
        test = table["month"].eq(month).to_numpy()
        fit = ~test & (table["label"] <= FIT_LABEL_CAP).to_numpy()
        ds = lgb.Dataset(table.loc[fit, features], table.loc[fit, "residual"], categorical_feature=[c for c in CATEGORICAL if c in features], free_raw_data=False)
        booster = lgb.train(PARAMS, ds, num_boost_round=ROUNDS)
        out[test] = booster.predict(table.loc[test, features])
    return out


def summarise(table: pd.DataFrame, columns: dict[str, np.ndarray]) -> dict:
    report = {}
    for month in DEV_MONTHS + ["all"]:
        m = np.ones(len(table), dtype=bool) if month == "all" else table["month"].eq(month).to_numpy()
        report[month] = {name: rmse(table["label"].to_numpy()[m] - values[m]) for name, values in columns.items()}
        for kind in ["gate", "dark", "unmatched"]:
            k = m & table["kind"].eq(kind).to_numpy()
            report[month][kind] = {name: rmse(table["label"].to_numpy()[k] - values[k]) for name, values in columns.items()} | {"rows": int(k.sum())}
        report[month]["airports"] = {}
        for airport, a in table.groupby("airport", observed=True).indices.items():
            am = np.zeros(len(table), dtype=bool); am[a] = True; am &= m
            report[month]["airports"][airport] = {name: round(rmse(table["label"].to_numpy()[am] - values[am]), 2) for name, values in columns.items()}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", default="runs/adsb_v17/ens_v17_bag_ctr_c2")
    parser.add_argument("--residual-run", default="runs/adsb_v17_sel")
    parser.add_argument("--features-dir", default="data/external/adsblol/features_v19_screen", help="tables built from matches_v11 (the residual stage's inputs); the VM round uses features_v19/ built from matches/")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    table = build_table(args.learner, args.residual_run, ROOT / args.features_dir)
    base = list(FEATURE_NAMES) + list(NEIGHBOUR_NAMES) + CONTEXT + CATEGORICAL
    new = base + list(FEATURE_NAMES_V19)
    print("rows", len(table), "kinds", table["kind"].value_counts().to_dict(), "v19 coverage", {f: round(float(table[f].notna().mean()), 3) for f in FEATURE_NAMES_V19}, flush=True)
    pred = table["pred"].to_numpy()
    r_base = lomo(table, base)
    r_new = lomo(table, new)
    columns = {"incumbent": pred, "incumbent+base": pred + r_base, "incumbent+new": pred + r_new}
    report = summarise(table, columns)
    # a floor at the recorded value keeps the comparison on the same footing as an export
    floor = float(pd.read_parquet(ROOT / "scratch/forward.parquet")["floor"].iloc[0])
    report["floored"] = summarise(table, {k: np.maximum(v, floor) for k, v in columns.items()})
    gain = {m: (report[m]["incumbent+new"] / report[m]["incumbent+base"] - 1) * 100 for m in DEV_MONTHS}
    report["gain_new_vs_base_pct"] = gain
    report["go"] = bool(all(g <= -0.3 for g in gain.values()))
    report["settings"] = {"params": PARAMS, "rounds": ROUNDS, "fit_label_cap": FIT_LABEL_CAP, "base_features": base, "new_features": list(FEATURE_NAMES_V19)}
    # feature importance of the new set on the last fit (gain), for the record
    fit = (table["label"] <= FIT_LABEL_CAP).to_numpy()
    booster = lgb.train(PARAMS, lgb.Dataset(table.loc[fit, new], table.loc[fit, "residual"], categorical_feature=CATEGORICAL), num_boost_round=ROUNDS)
    imp = dict(zip(new, booster.feature_importance("gain")))
    total = sum(imp.values()) or 1.0
    report["importance_gain_share_all_months_fit"] = {k: round(v / total, 4) for k, v in sorted(imp.items(), key=lambda kv: -kv[1])}
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    for m in DEV_MONTHS + ["all"]:
        r = report[m]
        print(m, {k: round(r[k], 3) for k in columns}, "| gate/dark/unmatched (new vs base):",
              {kind: (round(r[kind]["incumbent+base"], 1), round(r[kind]["incumbent+new"], 1)) for kind in ["gate", "dark", "unmatched"]}, flush=True)
    print("gain new vs base %:", {k: round(v, 3) for k, v in gain.items()}, "go:", report["go"])
    print("top new-feature importances:", {k: v for k, v in report["importance_gain_share_all_months_fit"].items() if k in FEATURE_NAMES_V19})
    print("wrote", out)


if __name__ == "__main__":
    main()
