"""Exploration script (copied from the session scratchpad, 2 October 2026; run as: python docs/scout/explore_calibrated_shift.py stack 0.9 80): per-airport calibrated day shift, applied only where it is reliable on the fitting months.

Table: 'stack' (runs/daily_totals/rows.parquet, v46 stack, four months, v48 = runs/daily_totals/daily_totals_v1) or 'twelve'
(runs/daily_totals/rows12.parquet, v23-recipe predictor, twelve months, v48 = runs/scout/daily_ifr/v48_twelve). For each month read,
from the OTHER months: c0_a (median published minus own total), per airport corr and least-squares slope beta_a of the day's mean
residual (in-figure, rows v48 moved, |r| < 1800) on I_d. An airport is calibrated in that fold when corr >= CORR_MIN over at least
MIN_DAYS fitting days; there, on read days with dn = 0 and n_eff >= 100, the rows v48 moved that are in the figure take
v48 - v48 day shift + clip(beta_a, 0, 1) * I_d. Everything else keeps v48."""
import sys
import numpy as np
import pandas as pd

ROOT = ""
sys.path.insert(0, ROOT + "src")
sys.path.insert(0, ROOT + "tools")
from prc2026_scout.published_figure import in_figure  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402

KIND = sys.argv[1]
CORR_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.9
I_MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
MIN_DAYS = 30
if KIND == "stack":
    t = pd.read_parquet(ROOT + "runs/daily_totals/rows.parquet")
    t["extra"] = False
    src = ROOT + "runs/daily_totals/daily_totals_v1"
else:
    t = pd.read_parquet(ROOT + "runs/daily_totals/rows12.parquet")
    src = ROOT + "runs/scout/daily_ifr/v48_twelve"
MONTHS = sorted(t["month"].unique())
rules = pd.read_parquet(ROOT + "runs/scout/daily_ifr/rules.parquet").set_index("MVT_ID_mvt")["ifr"]
t["fig"] = in_figure(t["airport"].to_numpy(), rules.reindex(t["MVT_ID_mvt"]).fillna(True).to_numpy(dtype=bool))
v48 = pd.concat([pd.read_parquet(f"{src}/{m}.parquet") for m in MONTHS], ignore_index=True).rename(columns={"new": "v48", "shift": "v48_shift"})
t = t.merge(v48, on="MVT_ID_mvt", how="left")
t["moved"] = t["v48"].notna() & ((t["v48"] - t["stack"]).abs() > 1e-9)
t["v48"] = t["v48"].fillna(t["stack"])
t["pos_d"] = (t["a"] > t["stack"]).astype(float)
f = t[t["fig"]]
day = f.groupby(["month", "airport", "bday"]).agg(rows=("a", "size"), implied=("pos_hat", "sum"), n_eff=("pos_d", "sum"), pub=("pub", "first"), n_pub=("n_pub", "first"))
day = day.join(f[~f["extra"]].groupby(["month", "airport", "block_day"])["pos_true"].sum().rename_axis(["month", "airport", "bday"]).rename("true_own"))
o = f[f["moved"] & ~f["extra"] & ((f["label"] - f["stack"]).abs() < 1800)]
day = day.join(o.groupby(["month", "airport", "bday"]).apply(lambda x: (x["label"] - x["stack"]).mean()).rename("need"))
day["dn"] = day["n_pub"] - day["rows"]
delta_raw = day["pub"] - day["implied"]
day["ok"] = ((day["dn"] == 0) | ((day["dn"] > 0) & (delta_raw < 0)) | ((day["dn"] < 0) & (delta_raw > 0))) & (day["n_eff"] >= 100) & day["pub"].notna()
day["ok_fit"] = (day["dn"] == 0) & (day["n_eff"] >= 100) & day["pub"].notna()
out, chosen = [], {}
for m in MONTHS:
    fit = day[(day.index.get_level_values("month") != m) & day["ok_fit"]].dropna(subset=["need", "true_own"])
    c0 = (fit["pub"] - fit["true_own"]).groupby(level="airport").median()
    fit = fit.assign(I=-(fit["pub"] - fit.index.get_level_values("airport").map(c0).to_numpy() - fit["implied"]) / fit["n_eff"])
    stats = fit.groupby(level="airport").apply(lambda x: pd.Series({"days": len(x), "corr": float(np.corrcoef(x["need"], x["I"])[0, 1]),
                                                                    "beta": float((x["need"] * x["I"]).sum() / (x["I"] ** 2).sum())}))
    ok = stats[(stats["corr"] >= CORR_MIN) & (stats["days"] >= MIN_DAYS)]
    chosen[m] = {a: round(float(b), 3) for a, b in ok["beta"].clip(0, 1).items()}
    use = day[(day.index.get_level_values("month") == m) & day["ok"]].copy()
    ap = use.index.get_level_values("airport")
    use = use[ap.isin(ok.index)]
    ap = use.index.get_level_values("airport")
    use["I"] = -(use["pub"] - ap.map(c0).to_numpy() - use["implied"]) / use["n_eff"]
    use["shift_new"] = ap.map(ok["beta"].clip(0, 1)).to_numpy() * use["I"]
    use = use[use["I"].abs() >= I_MIN]
    out.append(use[["shift_new", "I"]])
s = pd.concat(out)
t = t.join(s, on=["month", "airport", "bday"])
apply = t["moved"] & t["fig"] & t["shift_new"].notna()
t["new"] = np.maximum(np.where(apply, t["v48"] - t["v48_shift"].fillna(0) + t["shift_new"].fillna(0), t["v48"]), -12.0)
t = t[~t["extra"]]
print(KIND, "CORR_MIN", CORR_MIN, "calibrated airports per month:", chosen)
res = []
for m, g in t.groupby("month"):
    y, c, n = g["label"].to_numpy(dtype=float), g["v48"].to_numpy(dtype=float), g["new"].to_numpy(dtype=float)
    pdays = paired_days(g["bday"].to_numpy(), y, n, c, 20260929)
    ch = 100 * (rmse(y, n) / rmse(y, c) - 1)
    res.append(ch)
    print(m, f"{rmse(y, c):.4f} -> {rmse(y, n):.4f} {ch:+.4f}%", "interval", [round(v, 3) for v in pdays["rmse_difference_95pct_interval_seconds"]],
          "days better", pdays["days_with_lower_squared_error"], "/", pdays["days"], "rows changed", int((np.abs(n - c) > 0.5).sum()))
print(f"months better {sum(r < 0 for r in res)} of {len(res)}, mean {np.mean(res):+.4f}%")
t["se_c"], t["se_n"] = (t["v48"] - t["label"]) ** 2, (t["new"] - t["label"]) ** 2
by = t.groupby(["airport", "month"]).agg(c=("se_c", "sum"), n=("se_n", "sum"))
print((100 * (by["n"] / by["c"] - 1)).unstack("month").round(3).to_string())
