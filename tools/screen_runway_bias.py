"""Runway bias inside airport-months after v48's reconciliation (2025 development months; label-reading, dev only).

Oracle upper bound for per-runway monthly published taxi-out figures: within each airport-month, shift every row of a runway by
-(mean residual of that runway's valid rows - mean residual of the airport-month's valid rows). The airport-month level itself is left
alone (the daily totals already carry it). Valid rows = labels in (0, 7200], as the published workbook's valid flights roughly.
"""
import glob
import numpy as np
import pandas as pd

from pathlib import Path
ROOT = str(Path(__file__).resolve().parents[1]) + "/"
ID = "MVT_ID_mvt"
rows = pd.read_parquet(ROOT + "runs/daily_totals/rows.parquet", columns=[ID, "month", "airport", "label", "stack", "cls"])
new = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(ROOT + "runs/daily_totals/daily_totals_v1/2025-??.parquet"))])
t = rows.merge(new[[ID, "new"]], on=ID, how="left")
t["pred"] = t["new"].fillna(t["stack"])
rw = []
for f in sorted(glob.glob(ROOT + "data/raw/training_2025-*.parquet")):
    r = pd.read_parquet(f, columns=[ID, "PHASE_mvt", "RUNWAY_mvt"])
    rw.append(r[r["PHASE_mvt"].eq("DEP")][[ID, "RUNWAY_mvt"]])
t = t.merge(pd.concat(rw).drop_duplicates(ID), on=ID, how="left")
t["RUNWAY_mvt"] = t["RUNWAY_mvt"].fillna("NA")
t["r"] = t["label"] - t["pred"]
t["valid"] = (t["label"] > 0) & (t["label"] <= 7200)
print("rows", len(t), "runway missing", int(t["RUNWAY_mvt"].eq("NA").sum()))

out = []
for (m, ap), g in t.groupby(["month", "airport"]):
    v = g[g["valid"]]
    level = v["r"].mean()
    b = v.groupby("RUNWAY_mvt")["r"].agg(["mean", "size"])
    b["bias"] = b["mean"] - level
    corr = g["RUNWAY_mvt"].map(b["bias"]).fillna(0.0).to_numpy()
    e0 = (g["r"].to_numpy()) ** 2
    e1 = (g["r"].to_numpy() - corr) ** 2
    vv = g["valid"].to_numpy()
    out.append({"month": m, "airport": ap, "rows": len(g), "runways": int((b["size"] >= 30).sum()),
                "max_abs_bias_s": float(b.loc[b["size"] >= 30, "bias"].abs().max()) if (b["size"] >= 30).any() else 0.0,
                "sse": float(e0.sum()), "dsse": float(e1.sum() - e0.sum()), "dsse_valid": float(e1[vv].sum() - e0[vv].sum())})
o = pd.DataFrame(out)
month = o.groupby("month")[["sse", "dsse"]].sum()
month["pct"] = 100 * month["dsse"] / month["sse"]
print(month.round(4).to_string())
o["pct_of_month"] = 100 * o["dsse"] / o["month"].map(month["sse"])
pd.set_option("display.width", 200)
print(o.sort_values("dsse").head(15).round(3).to_string(index=False))
# persistence: does a runway's within-airport bias carry from one dev month to another? (if yes, the model part has a learnable bias)
bias = []
for (m, ap), g in t[t["valid"]].groupby(["month", "airport"]):
    lv = g["r"].mean()
    for rwy, h in g.groupby("RUNWAY_mvt"):
        if len(h) >= 100:
            bias.append({"month": m, "airport": ap, "runway": rwy, "bias": h["r"].mean() - lv, "n": len(h)})
B = pd.DataFrame(bias).pivot_table(index=["airport", "runway"], columns="month", values="bias")
print(B.round(1).to_string())
c = B.corr()
print("month-to-month correlation of runway biases:\n", c.round(2).to_string())
