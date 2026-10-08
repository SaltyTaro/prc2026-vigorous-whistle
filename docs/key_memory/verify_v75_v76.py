"""Independent check of the v75 (season_pass_v2 w25b_5) and v76 (key_memory_v1 km_9) dry runs. Run from the repo root."""
import hashlib

import numpy as np
import pandas as pd

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


v74p = "runs/submissions/vigorous-whistle_v74.parquet"
v75p = "runs/season_pass2/rank/dry_v75/vigorous-whistle_v75.parquet"
v76p = "runs/key_memory/rank/dry_v76/vigorous-whistle_v76.parquet"
assert sha(v74p) == "85e57d05322009d886d7423ba818ec6435f023bb8fd054463dd8adca6983f761"
print("v75", sha(v75p)[:16], "v76", sha(v76p)[:16])
a, b, c = pd.read_parquet(v74p), pd.read_parquet(v75p), pd.read_parquet(v76p)
tmpl = pd.read_parquet("data/raw/submitting.parquet")
for nm, f in [("v75", b), ("v76", c)]:
    print(nm, "rows", len(f), "dtype", f[T].dtype, "cols", list(f.columns), "order==template", bool((tmpl[ID].values == f[ID].values).all()), "min", f[T].min())
raw = pd.read_parquet("data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt"]).set_index(ID).reindex(a[ID])
mon = raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()
ap = raw["ADEP_mvt"].to_numpy()
pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
guard = a[ID].isin([213142471, 213142469]).to_numpy()
K = pd.read_parquet("runs/season_pass2/rank_sp2.parquet", columns=[ID, "kind"]).set_index(ID).reindex(a[ID])
fixed = (K["kind"] == 3).to_numpy()
print("kind-3 rows", fixed.sum(), "population rows", pop.sum(), "population within kind 3", bool((~pop | fixed).all()), "guard within kind 3", bool(fixed[guard].all()))
for nm, x, y in [("v75 vs v74", a, b), ("v76 vs v75", b, c)]:
    ch = x[T].values != y[T].values
    d = y[T].values.astype(float) - x[T].values
    print(nm, "changed", int(ch.sum()), "airports", sorted(set(ap[ch])), "months", sorted(set(mon[ch])), "pop", int((ch & pop).sum()), "guard", int((ch & guard).sum()),
          "fixed", int((ch & fixed).sum()), "max|d|", np.abs(d).max(), "S", (d ** 2).sum())

# v75 recomputed from the bags
P = pd.read_parquet("runs/season_pass2/preds_rank.parquet").set_index(ID).reindex(a[ID])
v = a[T].values.astype(float).copy()
for apt in ["EDDF", "EDDM", "EHAM", "LSZH", "LEMD"]:
    p = P[f"sp_{apt}"].values
    m = ~np.isnan(p)
    assert (ap[m] == apt).all() and (mon[m] == "2026-01").all() and not fixed[m].any(), apt
    v[m] = np.maximum(v[m] + np.clip(0.5 * p[m], -120, 120), -12)
print("v75 recomputed equal", bool((np.rint(v).astype(b[T].dtype) == b[T].values).all()))

# v76 recomputed: residual means per airport + flight number on v75's development values (Nov + Dec 2025), written without the tool
D = pd.read_parquet("runs/season_pass2/dev_sp2.parquet", columns=[ID, "month", "airport", "label", "ship", "kind"])
PD = pd.read_parquet("runs/season_pass2/preds_dev.parquet").set_index(ID).reindex(D[ID])
dv = D["ship"].to_numpy(dtype=float).copy()
for apt in ["EDDF", "EDDM", "EHAM", "LSZH", "LEMD"]:
    p = PD[f"sp_{apt}"].values
    m = ~np.isnan(p)
    dv[m] = np.maximum(dv[m] + np.clip(0.5 * p[m], -120, 120), -12)
files = ["data/raw/training_2025-11-01_2025-12-01.parquet", "data/raw/training_2025-12-01_2026-01-01.parquet"]
fr = pd.concat([pd.read_parquet(f, columns=[ID, "PHASE_mvt", "FLIGHT_mvt"]) for f in files])
fr = fr[fr["PHASE_mvt"] == "DEP"].set_index(ID)["FLIGHT_mvt"]
D["fn"] = fr.reindex(D[ID]).astype("string").str.strip().str.upper().to_numpy()
use = D["month"].isin(["2025-11", "2025-12"]).to_numpy() & (D["kind"] != 3).to_numpy() & (D["label"] < 20000).to_numpy() & D["fn"].notna().to_numpy() & (D["fn"].str.len() > 0).fillna(False).to_numpy()
r = np.clip(D["label"].to_numpy() - dv, -1800, 1800)
acc = {}
for k1, k2, rv in zip(D["airport"].to_numpy()[use], D["fn"].to_numpy()[use], r[use]):
    s = acc.setdefault((k1, k2), [0.0, 0])
    s[0] += rv
    s[1] += 1
aps9 = {"EDDF", "EDDM", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"}
fnk = raw["FLIGHT_mvt"].astype("string").str.strip().str.upper().to_numpy()
w = b[T].values.astype(float).copy()
for i in np.flatnonzero((mon == "2026-01") & ~fixed & np.isin(ap, list(aps9))):
    key = (ap[i], fnk[i])
    if fnk[i] is pd.NA or key not in acc:
        continue
    s, n = acc[key]
    w[i] = max(w[i] + float(np.clip(0.25 * n / (n + 30) * (s / n), -120, 120)), -12)
print("v76 recomputed equal", bool((np.rint(w).astype(c[T].dtype) == c[T].values).all()), "mismatches", int((np.rint(w) != c[T].values).sum()))
