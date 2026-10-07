import glob, sys
import numpy as np, pandas as pd
raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]) for p in
                 sorted(glob.glob("data/raw/training_2025-*.parquet")) + ["data/raw/ranking.parquet"]])
raw = raw[raw.PHASE_mvt == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
src = sys.argv[1]
fs = sorted(glob.glob(src))
res = []
G = []
for p in fs:
    f = pd.read_parquet(p)
    f = f.set_index("MVT_ID_mvt")
    f = f.join(raw[["ADEP_mvt", "MVT_TIME_UTC_mvt"]], how="inner")
    f["m"] = f.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    num = f.select_dtypes(include=[np.number, "bool"]).astype(float)
    na = num.isna().add_prefix("na:")
    med = num.add_prefix("mean:")
    x = pd.concat([na.astype(float), med], axis=1)
    x["a"] = f.ADEP_mvt.values; x["m"] = f.m.values
    G.append(x.groupby(["a", "m"]).mean())
g = pd.concat(G)
g = g[~g.index.duplicated()]
for a, ga in g.groupby(level=0):
    ga = ga.droplevel(0)
    base = ga.loc[[m for m in ga.index if m.startswith("2025") and m >= "2025-05"]]  # LIRF ADS-B starts May
    for m in ("2026-01", "2026-07"):
        if m not in ga.index: continue
        row = ga.loc[m]; lo, hi = base.min(), base.max(); span = (hi - lo).replace(0, np.nan)
        for c in ga.columns:
            v = row[c]
            if pd.isna(v) or pd.isna(lo[c]): continue
            out = v - hi[c] if v > hi[c] else (v - lo[c] if v < lo[c] else 0.0)
            rel = out / (abs(hi[c]) + 1e-9) if c.startswith("mean:") else out
            if (c.startswith("na:") and abs(out) >= 0.05) or (c.startswith("mean:") and abs(rel) >= 0.25):
                res.append((a, m, c, round(v, 3), round(lo[c], 3), round(hi[c], 3), round(out, 3)))
res = pd.DataFrame(res, columns=["airport", "month", "feature", "v2026", "min25", "max25", "outside"])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
print(res.sort_values(["airport", "month", "feature"]).to_string(index=False))
