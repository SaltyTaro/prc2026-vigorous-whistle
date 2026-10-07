"""Recording-drift screen, steps 1-5 (docs/rec_drift/): small tables only, local.

Inputs
  runs/rec_drift/arr_arcw*.parquet   per-arrival ADS-B touchdown / stop times (tools/rec_drift_arrivals.py on arcw1-3)
  data/external/adsblol/matches/matches2_<month>.parquet   shipped departure matcher (gate rows: motion_start = first point beyond
                                                          40 m of the parked anchor after the aircraft last left 25 m of it)
  data/raw (departure off-block of 2025 = take-off - label, AOBT_3_flt both years)
  runs/submissions/vigorous-whistle_v70.parquet, runs/stack_weight/dev/w65_bag10/proxy.parquet (v69 development proxy)

Quantities (seconds; positive = the recorded clock is later than the ADS-B event)
  arr_A    recorded in-block - first stop (gs <= 1 kt) within 40 m of the learned stand position (previous point <= 30 s earlier)
  arr_E    recorded in-block - first point within 40 m of the stand (entering)
  td       ADS-B touchdown - recorded landing (matching check)
  dep      recorded off-block - ADS-B motion start (2025 only: the off-block is masked in the ranking file)
  aobt     AOBT_3_flt - ADS-B motion start
  imp      implied off-block of a prediction - motion start = (take-off - prediction) - motion start  (v70 in 2026, v69 proxy in 2025)
  to       recorded take-off - ADS-B last ground point (coverage check)
Offsets beyond +-900 s are treated as mismatches. Two readings per quantity and airport-month:
  raw   median over matched rows (departures: gate rows);
  fe    median of the row minus its stand's 2025 median (stands with >= 20 rows in 2025; departures: gate rows whose segment starts
        within 40 m of the known stand), so that a change in WHICH stands the receivers see does not read as a recording change;
  fe_dense (departures) as fe on tracks with > 0.05 points / s: sparse tracks see the motion start late (EDDF 2026).
Writes runs/rec_drift/airport_month.csv and docs/rec_drift/results.json.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402

ID = "MVT_ID_mvt"
EPOCH = pd.Timestamp(0, tz="UTC")
MONTHS_2025 = [f"2025-{m:02}" for m in range(1, 13)]
MONTHS = MONTHS_2025 + ["2026-01", "2026-07"]
DEV = ["2025-01", "2025-07", "2025-11", "2025-12"]
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
GAP_MAX, OFF_MAX, MIN_N, MIN_STAND, DENSE = 30.0, 900.0, 100, 20, 0.05
BAR = 5e6
V70 = "runs/submissions/vigorous-whistle_v70.parquet"
PROXY = "runs/stack_weight/dev/w65_bag10/proxy.parquet"
OUT_CSV = ROOT / "runs/rec_drift/airport_month.csv"
OUT_JSON = ROOT / "docs/rec_drift/results.json"


def stats(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    if n == 0:
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.quantile(x, [0.1, 0.25, 0.5, 0.75, 0.9])
    iqr = q75 - q25
    return {"n": int(n), "median": float(q50), "tmean": float(x[(x >= q10) & (x <= q90)].mean()), "q25": float(q25), "q75": float(q75),
            "iqr": float(iqr), "se_median": float(1.2533 * (iqr / 1.349) / np.sqrt(n))}


def training_file(month: str) -> Path:
    y, m = int(month[:4]), int(month[5:7])
    return ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def clip(x: pd.Series) -> pd.Series:
    return x.where(x.abs() <= OFF_MAX)


def stand_fe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Row minus its (airport, stand) 2025 median, for stands with >= MIN_STAND 2025 rows of the first column."""
    base = df[df["month"] < "2026"].groupby(["airport", "stand"])[cols].median().add_prefix("m_")
    base["nst"] = df[df["month"] < "2026"].groupby(["airport", "stand"])[cols[0]].count()
    out = df.join(base, on=["airport", "stand"])
    out = out[out["nst"] >= MIN_STAND].copy()
    for c in cols:
        out[f"fe_{c}"] = out[c] - out[f"m_{c}"]
    return out


def arrivals() -> tuple[pd.DataFrame, list[str]]:
    files = sorted(glob.glob(str(ROOT / "runs/rec_drift/arr_arcw*.parquet")))
    a = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates(ID)
    a["month"] = a["day"].str[:4] + "-" + a["day"].str[4:6]
    a["arr_A"] = clip(pd.Series(np.where(a["stopA_gap"] <= GAP_MAX, a["inblk"] - a["stopA"], np.nan), index=a.index))
    a["arr_E"] = clip(a["inblk"] - a["enterA"])
    a["td"] = clip(a["td"] - a["L"])
    return a, files


def departures() -> pd.DataFrame:
    parts = []
    for month in MONTHS:
        m = pd.read_parquet(ROOT / f"data/external/adsblol/matches/matches2_{month}.parquet",
                            columns=[ID, "airport", "stand", "gate", "segment_start", "motion_start", "takeoff", "last_ground", "at_known_stand", "points"])
        src = training_file(month) if month < "2026" else ROOT / "data/raw/ranking.parquet"
        r = pd.read_parquet(src, columns=[ID, "PHASE_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"])
        r = r[r["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
        m = m.merge(r, on=ID, how="inner")
        m["month"] = month
        m["blk"] = (m["BLOCK_TIME_UTC_mvt"] - EPOCH).dt.total_seconds() if month < "2026" else np.nan
        m["aobt3"] = (m["AOBT_3_flt"] - EPOCH).dt.total_seconds()
        parts.append(m.drop(columns=["PHASE_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"]))
    d = pd.concat(parts, ignore_index=True)
    d = d[d["gate"].fillna(False).astype(bool)].copy()
    d["dep"] = clip(d["blk"] - d["motion_start"])
    d["aobt"] = clip(d["aobt3"] - d["motion_start"])
    d["to"] = clip(d["takeoff"] - d["last_ground"])
    d["dens"] = d["points"] / (d["last_ground"] - d["segment_start"]).clip(lower=1)
    v70 = pd.read_parquet(ROOT / V70).rename(columns={"TAXITIME_SEC_mvt": "v70"})
    prox = pd.read_parquet(ROOT / PROXY, columns=[ID, "label", "value"])
    d = d.merge(v70, on=ID, how="left").merge(prox, on=ID, how="left")
    pred = np.where(d["month"].str.startswith("2026"), d["v70"], d["value"])
    d["imp"] = clip(d["takeoff"] - pred - d["motion_start"])
    return d


def table(a: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    afe = stand_fe(a, ["arr_A", "arr_E"])
    known = d[d["at_known_stand"].fillna(False).astype(bool)].copy()
    dfe = stand_fe(known, ["dep", "aobt"])
    dfe["fe_imp"] = dfe["imp"] - dfe["m_dep"]
    dense = dfe[dfe["dens"] > DENSE]
    readings = [("raw", a, ["arr_A", "arr_E", "td"], ""), ("fe", afe, ["arr_A", "arr_E"], "fe_"),
                ("raw", d, ["dep", "aobt", "imp", "to"], ""), ("fe", dfe, ["dep", "aobt", "imp"], "fe_"), ("fe_dense", dense, ["dep", "aobt", "imp"], "fe_")]
    for reading, df, cols, pre in readings:
        for (airport, month), g in df.groupby(["airport", "month"]):
            for q in cols:
                rows.append({"airport": airport, "month": month, "reading": reading, "quantity": q, **stats(g[pre + q])})
    return pd.DataFrame(rows)


def series(t: pd.DataFrame, reading: str, q: str, stat: str = "median") -> pd.DataFrame:
    s = t[(t["reading"] == reading) & (t["quantity"] == q) & (t["n"] >= MIN_N)].pivot(index="month", columns="airport", values=stat)
    return s.reindex(index=MONTHS)


def shifts(t: pd.DataFrame, reading: str, q: str) -> dict:
    s, se, n = series(t, reading, q), series(t, reading, q, "se_median"), series(t, reading, q, "n")
    out = {}
    for ap in s.columns:
        h = s[ap].loc[MONTHS_2025].dropna()
        if len(h) < 3:
            continue
        rec = {"months_2025": int(len(h)), "mean_2025": round(float(h.mean()), 2), "min_2025": round(float(h.min()), 2),
               "max_2025": round(float(h.max()), 2), "sd_2025": round(float(h.std()), 2), "by_month_2025": {m: round(float(v), 1) for m, v in h.items()}}
        for m26 in ("2026-01", "2026-07"):
            v = s[ap].get(m26)
            if pd.notna(v):
                same = s[ap].get(m26.replace("2026", "2025"))
                rec[m26] = {"median": round(float(v), 2), "n": int(n[ap][m26]), "se": round(float(se[ap][m26]), 2),
                            "vs_mean_2025": round(float(v - h.mean()), 2),
                            "vs_same_month_2025": round(float(v - same), 2) if pd.notna(same) else None,
                            "outside_2025_range": bool(v < h.min() or v > h.max())}
        out[ap] = rec
    return out


def comovement(t: pd.DataFrame, x: tuple, y: tuple) -> dict:
    """Correlation of 2025 month-to-month deviations (from each airport's 2025 mean) of two airport-month series."""
    sx, sy = series(t, *x).loc[MONTHS_2025], series(t, *y).loc[MONTHS_2025]
    out, px, py = {}, [], []
    for ap in AIRPORTS:
        if ap not in sx or ap not in sy:
            continue
        both = pd.concat([sx[ap], sy[ap]], axis=1).dropna()
        if len(both) < 6:
            continue
        dx, dy = both.iloc[:, 0] - both.iloc[:, 0].mean(), both.iloc[:, 1] - both.iloc[:, 1].mean()
        out[ap] = {"months": int(len(both)), "r": round(float(np.corrcoef(dx, dy)[0, 1]), 3) if dx.std() > 0 and dy.std() > 0 else None,
                   "slope_y_on_x": round(float((dx * dy).sum() / (dx * dx).sum()), 3) if (dx * dx).sum() > 0 else None,
                   "sd_x": round(float(dx.std()), 2), "sd_y": round(float(dy.std()), 2)}
        px += list(dx); py += list(dy)
    if len(px) >= 6:
        out["pooled"] = {"points": len(px), "r": round(float(np.corrcoef(px, py)[0, 1]), 3)}
    return out


def v70_response(t: pd.DataFrame, n_dep: pd.DataFrame) -> dict:
    """Per airport and 2026 month: arrival shift d (fe), v70's implied-offset change against the 2025 proxy (same month), the proxy's
    2025 bias pattern on the same rows (imp - dep = label - prediction), and the order of magnitude b^2 x departures."""
    out = {}
    arr = series(t, "fe", "arr_A")
    for reading in ("fe", "fe_dense"):
        imp, dep, aob = series(t, reading, "imp"), series(t, reading, "dep"), series(t, reading, "aobt")
        res = {}
        for ap in AIRPORTS:
            if ap not in imp:
                continue
            rec = {"proxy_2025": {m: {"imp": round(float(imp[ap][m]), 1), "dep": round(float(dep[ap][m]), 1),
                                      "label_minus_pred": round(float(imp[ap][m] - dep[ap][m]), 1)}
                                  for m in DEV if pd.notna(imp[ap].get(m)) and ap in dep and pd.notna(dep[ap].get(m))}}
            for m26 in ("2026-01", "2026-07"):
                v = imp[ap].get(m26)
                if pd.isna(v):
                    continue
                m25 = m26.replace("2026", "2025")
                d_arr = float(arr[ap][m26] - arr[ap].loc[MONTHS_2025].mean()) if ap in arr and pd.notna(arr[ap].get(m26)) else None
                d_aobt = float(aob[ap][m26] - aob[ap].loc[MONTHS_2025].mean()) if ap in aob and pd.notna(aob[ap].get(m26)) else None
                resp = float(v - imp[ap][m25]) if pd.notna(imp[ap].get(m25)) else None
                nd = int(n_dep.get((ap, m26), 0))
                r = {"imp_v70": round(float(v), 1), "v70_change_vs_proxy_same_month": None if resp is None else round(resp, 1),
                     "arrival_shift_d": None if d_arr is None else round(d_arr, 1), "aobt_shift": None if d_aobt is None else round(d_aobt, 1),
                     "departures": nd}
                if d_arr is not None:
                    r["b2n_if_v70_ignores_d"] = round(d_arr ** 2 * nd, 0)
                    if resp is not None:
                        r["b_if_recording_moved_by_d"] = round(resp - d_arr, 1)
                        r["b2n_if_recording_moved_by_d"] = round((resp - d_arr) ** 2 * nd, 0)
                rec[m26] = r
            res[ap] = rec
        out[reading] = res
    return out


def main():
    a, files = arrivals()
    d = departures()
    t = table(a, d)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_CSV, index=False)
    rk = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    n_dep = rk.groupby([rk["ADEP_mvt"], rk["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")]).size()
    res = {"computed_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
           "inputs": {"arrival_tables": {Path(f).name: sha256_file(Path(f)) for f in files},
                      "arrival_code_sha256": sha256_file(ROOT / "tools/rec_drift_arrivals.py"),
                      "v70_sha256": sha256_file(ROOT / V70), "proxy_sha256": sha256_file(ROOT / PROXY)},
           "filters": {"stop_gap_max_s": GAP_MAX, "offset_abs_max_s": OFF_MAX, "min_rows_per_airport_month": MIN_N,
                       "min_2025_rows_per_stand": MIN_STAND, "dense_points_per_s": DENSE, "size_bar_s2": BAR},
           "arrivals_matched": {ap: int(v) for ap, v in a.groupby("airport").size().items()},
           "arrivals_with_stand_stop": {ap: int(v) for ap, v in a[a["arr_A"].notna()].groupby("airport").size().items()},
           "shifts": {f"{r}:{q}": shifts(t, r, q) for r, q in [("fe", "arr_A"), ("fe", "arr_E"), ("raw", "arr_A"), ("raw", "td"),
                                                               ("fe", "dep"), ("fe", "aobt"), ("fe_dense", "aobt"), ("raw", "aobt"),
                                                               ("raw", "dep"), ("raw", "to"), ("fe", "imp"), ("fe_dense", "imp")]},
           "comovement_2025": {f"{x[0]}:{x[1]}~{y[0]}:{y[1]}": comovement(t, x, y) for x, y in [
               (("fe", "arr_A"), ("fe", "dep")), (("fe", "arr_E"), ("fe", "dep")), (("fe", "aobt"), ("fe", "dep")),
               (("fe_dense", "aobt"), ("fe_dense", "dep")), (("fe", "arr_A"), ("fe", "aobt")), (("raw", "arr_A"), ("raw", "dep"))]},
           "v70_response": v70_response(t, n_dep)}
    write_json(OUT_JSON, res)
    print("written", OUT_JSON)


if __name__ == "__main__":
    main()
