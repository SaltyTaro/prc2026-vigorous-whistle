import glob, sys
import numpy as np, pandas as pd
fs = sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/raw/ranking.parquet"]
drop = {"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"}
cols = [c for c in pd.read_parquet(fs[-1]).columns if c not in drop]
frames = []
for p in fs:
    r = pd.read_parquet(p, columns=cols)
    r = r[r.PHASE_mvt == "DEP"]
    frames.append(r)
d = pd.concat(frames).drop_duplicates("MVT_ID_mvt")
d["m"] = d.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
feat = {}
for c in cols:
    if c in ("MVT_ID_mvt", "PHASE_mvt", "m"): continue
    feat[f"na:{c}"] = d[c].isna().astype(float)
# derived equalities
feat["flight_eq_callsign"] = (d.FLIGHT_mvt.astype(str) == d.CALLSIGN_flt.astype(str)).astype(float)
feat["type_eq_flttype"] = (d.AIRCRAFT_TYPE_mvt.astype(str) == d.AIRCRAFT_TYPE_flt.astype(str)).astype(float)
feat["ades_eq_flt"] = (d.ADES_mvt.astype(str) == d.ADES_flt.astype(str)).astype(float)
feat["stand_numeric"] = d.STAND_mvt.astype(str).str.fullmatch(r"\d+").astype(float)
feat["callsign_alnum"] = d.CALLSIGN_flt.astype(str).str.fullmatch(r"[A-Z]{3}\d{1,4}").astype(float).where(d.CALLSIGN_flt.notna())
feat["flight_iata"] = d.FLIGHT_mvt.astype(str).str.fullmatch(r"[A-Z0-9]{2}\d{1,4}[A-Z]?").astype(float).where(d.FLIGHT_mvt.notna())
feat["sched_sec0"] = (d.SCHED_TIME_UTC_mvt.dt.second == 0).astype(float).where(d.SCHED_TIME_UTC_mvt.notna())
feat["mvt_sec0"] = (d.MVT_TIME_UTC_mvt.dt.second == 0).astype(float)
feat["ifr"] = (d.FLIGHT_RULE_mvt.astype(str) == "I").astype(float)
F = pd.DataFrame(feat)
F["a"] = d.ADEP_mvt.values; F["m"] = d.m.values
g = F.groupby(["a", "m"]).mean()
res = []
for (a), ga in g.groupby(level=0):
    ga = ga.droplevel(0)
    base = ga.loc[[m for m in ga.index if m.startswith("2025")]]
    for m in ("2026-01", "2026-07"):
        if m not in ga.index: continue
        row = ga.loc[m]
        lo, hi = base.min(), base.max()
        sd = base.std().replace(0, np.nan)
        for c in ga.columns:
            v = row[c]
            if pd.isna(v): continue
            out = v - hi[c] if v > hi[c] else (v - lo[c] if v < lo[c] else 0.0)
            if abs(out) >= 0.02:
                res.append((a, m, c, round(v, 3), round(lo[c], 3), round(hi[c], 3), round(out, 3)))
res = pd.DataFrame(res, columns=["airport", "month", "feature", "value_2026", "min_2025", "max_2025", "outside_by"])
pd.set_option("display.width", 220)
print(res.sort_values("outside_by", key=abs, ascending=False).to_string(index=False))
