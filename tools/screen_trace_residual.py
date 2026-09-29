"""Scratch screen (lever B): do richer raw-trace features predict the stack's residual on ADS-B-matched rows, out of month?
Month-blocked: fit on three development months, read the fourth. Control = the match table's own columns + v19 features + the stack's
components; candidate = control + the new trace features. Descriptive; nothing exported."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "runs/clean_line_screens/trace"
ID = "MVT_ID_mvt"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
CAP = 20000.0
rows = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet", columns=[ID, "month", "day", "airport", "label", "matched", "v9", "base", "mix", "committee", "nets", "v33"])
parts = []
for m in MONTHS:
    r = rows[rows["month"] == m]
    mt = pd.read_parquet(ROOT / f"data/external/adsblol/matches/matches2_{m}.parquet").drop(columns=["airport", "icao", "day", "stand"])
    v19 = pd.read_parquet(ROOT / f"data/external/adsblol/features_v19/features_v19_{m}.parquet")
    tr = pd.read_parquet(SP / f"features_{m}.parquet")
    parts.append(r.merge(mt, on=ID, how="left").merge(v19, on=ID, how="left").merge(tr, on=ID, how="left"))
t = pd.concat(parts, ignore_index=True)
t["m_taxi"] = t["takeoff"] - t["motion_start"]
t["m_seen"] = t["takeoff"] - t["segment_start"]
t["m_last_ground"] = t["takeoff"] - t["last_ground"]
t["m_arr_gap"] = t["segment_start"] - t["arrival_last_ground"]
t["hour"] = (t["takeoff"] % 86400) / 3600.0
t["airport_code"] = t["airport"].astype("category").cat.codes
t["gate"] = t["gate"].astype(float)
t["at_known_stand"] = t["at_known_stand"].astype(float)
t["res"] = t["label"] - t["v33"]
scope = t["matched"].to_numpy() & t["airport"].ne("LIRF").to_numpy() & t["tr_points"].notna().to_numpy()
print("rows", len(t), "scope", int(scope.sum()), {m: int((scope & t["month"].eq(m)).sum()) for m in MONTHS})
CONTROL = ["v33", "v9", "base", "mix", "committee", "nets", "airport_code", "hour", "gate", "at_known_stand", "m_taxi", "m_seen", "m_last_ground", "m_arr_gap", "first_gs", "stop_seconds",
           "taxi_path_m", "taxi_stops", "taxi_stopped_s", "slow_start_s", "gap_before_s", "points", "first_dist_stand_m", "anchor_dist_stand_m",
           *[c for c in pd.read_parquet(ROOT / "data/external/adsblol/features_v19/features_v19_2025-12.parquet").columns if c != ID]]
NEW = [c for c in t.columns if c.startswith("tr_")]
print("control", len(CONTROL), "new", len(NEW))
params = dict(objective="regression", learning_rate=0.04, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=20.0,
              n_estimators=500, verbose=-1, n_jobs=6, random_state=20260929)
rng = np.random.default_rng(20260929)
out = []
for held in MONTHS:
    fit = scope & t["month"].ne(held).to_numpy() & (t["label"] < CAP).to_numpy()
    test = scope & t["month"].eq(held).to_numpy()
    y = t.loc[fit, "res"].clip(-3600, 3600)
    preds = {}
    for name, cols in (("control", CONTROL), ("trace", CONTROL + NEW)):
        model = lgb.LGBMRegressor(**params).fit(t.loc[fit, cols], y)
        preds[name] = model.predict(t.loc[test, cols])
        if name == "trace":
            gain = pd.Series(model.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
            top_new = [(c, round(float(g / gain.sum()), 4)) for c, g in gain.items() if c in NEW][:8]
    f = t.loc[test]
    ordinary = (f["label"] < CAP).to_numpy()
    res = f["res"].to_numpy()
    month_all = t[t["month"].eq(held)]
    month_sse = float(((month_all["label"] - month_all["v33"]) ** 2).sum())
    month_sse_ord = float(((month_all["label"] - month_all["v33"]) ** 2)[month_all["label"] < CAP].sum())
    line = {"held": held, "rows": int(test.sum()), "scope_rmse": float(np.sqrt(np.mean(res[ordinary] ** 2))), "scope_share_of_month_sse_ord": float(np.sum(res[ordinary] ** 2) / month_sse_ord)}
    days = f["day"].to_numpy()[ordinary]
    for s in (0.5, 1.0):
        d = {}
        for name in preds:
            e = (res - s * preds[name])[ordinary]
            d[name] = e ** 2
            line[f"{name}_s{s}_scope_pct"] = 100 * (float(d[name].sum()) / float(np.sum(res[ordinary] ** 2)) - 1)
            line[f"{name}_s{s}_month_rmse_pct"] = 100 * (np.sqrt((month_sse - float(np.sum(res[ordinary] ** 2)) + float(d[name].sum())) / month_sse) - 1)
        uniq, inv = np.unique(days, return_inverse=True)
        diff = np.bincount(inv, d["trace"] - d["control"])
        boot = [diff[rng.integers(0, len(uniq), len(uniq))].sum() for _ in range(2000)]
        line[f"trace_minus_control_s{s}_sse_pct_of_scope"] = 100 * float(diff.sum()) / float(np.sum(res[ordinary] ** 2))
        line[f"trace_minus_control_s{s}_interval_pct"] = [round(100 * float(np.quantile(boot, q)) / float(np.sum(res[ordinary] ** 2)), 3) for q in (0.025, 0.975)]
    line["top_new"] = top_new
    out.append(line)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in line.items()}, flush=True)
    for cls, mask in (("gate", f["gate"].to_numpy() == 1), ("dark", f["gate"].to_numpy() == 0)):
        k = ordinary & mask
        print("   ", cls, int(k.sum()), "rmse", round(float(np.sqrt(np.mean(res[k] ** 2))), 1),
              {name: round(100 * (float(np.sum((res[k] - preds[name][k]) ** 2)) / float(np.sum(res[k] ** 2)) - 1), 3) for name in preds}, flush=True)
    for a in sorted(f["airport"].unique()):
        k = ordinary & (f["airport"].to_numpy() == a)
        print("      ", a, int(k.sum()), {name: round(100 * (float(np.sum((res[k] - preds[name][k]) ** 2)) / float(np.sum(res[k] ** 2)) - 1), 3) for name in preds}, flush=True)
pd.DataFrame(out).to_json(ROOT / "docs/clean_line_screens/trace_screen.json", orient="records", indent=1)
