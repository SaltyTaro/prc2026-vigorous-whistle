"""Scratch screen: does the final stack's residual persist per flight number across months? (development months only; nothing exported)"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "MVT_ID_mvt"
rows = pd.read_parquet(ROOT / "runs/lomo_dev/rows.parquet", columns=[ID, "month", "day", "airport", "label", "matched", "v9", "v33", "provenance"])
print(rows.groupby("month").size().to_dict(), rows["provenance"].value_counts().to_dict())
files = {"2025-01": "training_2025-01-01_2025-02-01", "2025-07": "training_2025-07-01_2025-08-01", "2025-11": "training_2025-11-01_2025-12-01", "2025-12": "training_2025-12-01_2026-01-01"}
raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{f}.parquet", columns=[ID, "PHASE_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt"]) for f in files.values()])
raw = raw[raw["PHASE_mvt"].eq("DEP")]
t = rows.merge(raw, on=ID, how="left", validate="one_to_one")
t["fn"] = t["FLIGHT_mvt"].astype("string").str.strip().str.upper()
t["cs"] = t["CALLSIGN_flt"].astype("string").str.strip().str.upper()
t["hour"] = t["SCHED_TIME_UTC_mvt"].dt.hour
t["res"] = t["label"] - t["v33"]
CAP = 20000.0
ordinary = t["label"] < CAP
t["resc"] = t["res"].clip(-1800, 1800)
print("rows", len(t), "fn missing", t["fn"].isna().mean().round(4), "cs missing", t["cs"].isna().mean().round(4))
months = sorted(t["month"].unique())

def screen(key_cols, name, lam_values=(2, 5, 10, 20)):
    out = []
    for m in months:
        tgt = t[t["month"].eq(m)]
        src = t[~t["month"].eq(m) & ordinary]
        g = src.groupby(key_cols, observed=True)["resc"].agg(["sum", "count"]).reset_index()
        x = tgt.merge(g, on=key_cols, how="left")
        n = x["count"].fillna(0).to_numpy()
        mean = np.where(n > 0, x["sum"].fillna(0).to_numpy() / np.maximum(n, 1), 0.0)
        ordm = (x["label"] < CAP).to_numpy()
        res = x["res"].to_numpy()
        base = float(np.sum(res[ordm] ** 2))
        full = float(np.sum(res ** 2))
        line = {"month": m, "covered": float((n > 0).mean()), "n5": float((n >= 5).mean())}
        k = ordm & (n >= 3)
        if k.sum() > 100:
            line["slope_n3"] = float(np.sum(mean[k] * res[k]) / np.sum(mean[k] ** 2))
            line["corr_n3"] = float(np.corrcoef(mean[k], res[k])[0, 1])
        for lam in lam_values:
            corr = n / (n + lam) * mean
            line[f"ord_pct_l{lam}"] = 100 * (float(np.sum((res[ordm] - corr[ordm]) ** 2)) / base - 1)
        line["full_pct_l10"] = 100 * (float(np.sum((res - n / (n + 10) * mean) ** 2)) / full - 1)
        out.append(line)
    print(f"\n== {name}")
    print(pd.DataFrame(out).round(4).to_string(index=False))

screen(["airport", "fn"], "airport + flight number")
screen(["airport", "cs"], "airport + ATC callsign")
screen(["airport", "fn", "RUNWAY_mvt"], "airport + flight number + runway")
screen(["airport", "AIRCRAFT_OPERATOR_flt", "ADES_mvt"], "airport + operator + destination")
screen(["airport", "AIRCRAFT_OPERATOR_flt", "STAND_mvt"], "airport + operator + stand")
screen(["airport", "STAND_mvt", "RUNWAY_mvt"], "airport + stand + runway (control: known to the models)")
