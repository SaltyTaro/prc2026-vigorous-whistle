"""Governing run of configs/nbr_state_v1.json: matched neighbours' ADS-B ground state as a layer on unmatched departures.

x_i = mean over matched rows j != i at the same airport within 900 s of (value_j - v9_j), minus the airport-day mean of that
difference over matched rows; new = max(value + clip(w * b_airport * x, +-120), -12) on unmatched, non-fixed rows.
Development (v70 proxy, month-blocked slopes), fresh August-October 2025 (rows12 'stack', slopes from the four development
months), ranking (uploaded v70; transfer guard per airport; dry run of the selected candidate). Light; runs locally.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import run_intruder_bound as R  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DECL = ROOT / "configs/nbr_state_v1.json"
DEV = (ROOT / "runs/oooi_lhr/dev_v70_proxy.parquet", "51aaa6a62f64897290199f6c6c9e7d3d43245a3862f36b2bdbc35be8012eabbe")
V70 = (ROOT / "runs/submissions/vigorous-whistle_v70.parquet", "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f")
LOMO = ROOT / "runs/lomo_dev/rows.parquet"
ROWS12 = ROOT / "runs/daily_totals/rows12.parquet"
RECIPE = ROOT / "runs/final_dryrun/run/recipe/detail.parquet"
RAW = {"2025-01": "training_2025-01-01_2025-02-01", "2025-07": "training_2025-07-01_2025-08-01", "2025-08": "training_2025-08-01_2025-09-01",
       "2025-09": "training_2025-09-01_2025-10-01", "2025-10": "training_2025-10-01_2025-11-01", "2025-11": "training_2025-11-01_2025-12-01",
       "2025-12": "training_2025-12-01_2026-01-01"}
DEV_MONTHS, EQUAL, FRESH = ["2025-01", "2025-07", "2025-11", "2025-12"], ["2025-01", "2025-07", "2025-12"], ["2025-08", "2025-09", "2025-10"]
GUARD = (213142471.0, 213142469.0)
WINDOW, MIN_N, CLIP_Y, CAP, FLOOR, MIN_FIT = 900.0, 3, 1800.0, 120.0, -12.0, 200
SEED = 20261004
OUT = ROOT / "docs/nbr_state"
RUNS = ROOT / "runs/nbr_state"
DRY = RUNS / "rank/dry_v71/vigorous-whistle_v71.parquet"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def takeoffs(months: list[str]) -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{RAW[m]}.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for m in months])
    return raw[raw["PHASE_mvt"] == "DEP"][[ID, "MVT_TIME_UTC_mvt"]]


def state(t: pd.DataFrame) -> np.ndarray:
    """x per row (NaN where fewer than MIN_N matched neighbours); t needs airport, ts (seconds), date, matched, value, v9."""
    ex = (t["value"] - t["v9"]).to_numpy(dtype=float)
    x = np.full(len(t), np.nan)
    pos = np.arange(len(t))
    for _, g in t.groupby("airport", sort=False):
        gi = pos[t.index.get_indexer(g.index)]
        mt = g["matched"].to_numpy() & np.isfinite(ex[gi])   # neighbours need v9 (fresh: 6,708 rows12 rows have no cross-fit v9)
        gm = g[mt].sort_values("ts")
        tg, eg = gm["ts"].to_numpy(dtype=float), ex[pos[t.index.get_indexer(gm.index)]]
        cs, cn = np.concatenate([[0.0], np.cumsum(eg)]), np.arange(len(tg) + 1)
        ts = g["ts"].to_numpy(dtype=float)
        lo, hi = np.searchsorted(tg, ts - WINDOW, side="left"), np.searchsorted(tg, ts + WINDOW, side="right")
        s, k = cs[hi] - cs[lo], (cn[hi] - cn[lo]).astype(float)
        s, k = s - np.where(mt, ex[gi], 0.0), k - mt
        m = np.where(k >= MIN_N, s / np.maximum(k, 1.0), np.nan)
        dmean = pd.Series(ex[gi][mt]).groupby(g["date"].to_numpy()[mt]).mean()
        x[gi] = m - g["date"].map(dmean).to_numpy(dtype=float)
    return x


def slope(x: np.ndarray, y: np.ndarray) -> float:
    yy = np.clip(y, -CLIP_Y, CLIP_Y)
    return float((x * yy).sum() / (x * x).sum())


def apply(value: np.ndarray, x: np.ndarray, b: np.ndarray, w: float, scope: np.ndarray) -> np.ndarray:
    new = value.copy()
    new[scope] = np.maximum(value[scope] + np.clip(w * b[scope] * x[scope], -CAP, CAP), FLOOR)
    return new


def month_row(s: pd.DataFrame, new: np.ndarray, rng: np.random.Generator) -> dict:
    pct, lo, hi = R.interval(s, "value", new, rng)
    return {"pct": pct, "lo": lo, "hi": hi, "rows_changed": int((new != s["value"].to_numpy()).sum()),
            "squared_change": float(((new - s["value"].to_numpy()) ** 2).sum()),
            "control_rmse": float(np.sqrt(((s["label"] - s["value"]) ** 2).mean())), "rmse": float(np.sqrt(((s["label"].to_numpy() - new) ** 2).mean()))}


# ---------------------------------------------------------------- tables

def dev_table() -> pd.DataFrame:
    if sha(DEV[0]) != DEV[1]:
        raise ValueError("v70 development proxy changed")
    d = pd.read_parquet(DEV[0]).merge(pd.read_parquet(LOMO, columns=[ID, "v9"]), on=ID, how="left")
    d = d.merge(takeoffs(DEV_MONTHS), on=ID, how="left")
    if d["v9"].isna().any() or d["MVT_TIME_UTC_mvt"].isna().any():
        raise ValueError("development join incomplete")
    d["ts"] = (d["MVT_TIME_UTC_mvt"] - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds()
    d["date"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    if not (d["date"] == d["day"]).all():
        raise ValueError("proxy day is not the UTC take-off date")
    d["matched"] = d["cls"].str.endswith("_m").to_numpy()
    d["fixed"] = d["cls"].eq("LIRF_pop").to_numpy() | d[ID].isin(GUARD).to_numpy()
    return d.reset_index(drop=True)


def fresh_table() -> pd.DataFrame:
    t = pd.read_parquet(ROWS12, columns=[ID, "month", "airport", "label", "matched", "stack"])
    t = t[t["month"].isin(FRESH)].rename(columns={"stack": "value"})
    v9 = pd.concat([pd.read_parquet(ROOT / f"scratch/v24/{m}.parquet", columns=[ID, "v9"]) for m in FRESH])
    # rows12 and scratch/v24 flag the LIRF population only in the four development months; its raw rule (LIRF, no NM origin, no
    # airport aircraft type) reproduces the development flag on every December row and on July's 337 (plus 18 rows the
    # development flag leaves out), so the fresh months hold that (slightly larger) set fixed
    raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{RAW[m]}.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
                     for m in FRESH])
    raw = raw[raw["PHASE_mvt"] == "DEP"]
    raw["population"] = (raw["ADEP_mvt"] == "LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()
    t = t.merge(v9, on=ID, how="left").merge(takeoffs(FRESH), on=ID, how="left").merge(raw[[ID, "population"]], on=ID, how="left")
    if t["MVT_TIME_UTC_mvt"].isna().any() or t["v9"].isna().sum() > 7000:
        raise ValueError("fresh join incomplete")
    t["ts"] = (t["MVT_TIME_UTC_mvt"] - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds()
    t["date"] = t["day"] = t["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    t["matched"] = t["matched"].fillna(False).astype(bool)
    t["fixed"] = t["population"].fillna(False).astype(bool).to_numpy() | t[ID].isin(GUARD).to_numpy()
    return t.reset_index(drop=True)


def ranking_table() -> pd.DataFrame:
    if sha(V70[0]) != V70[1]:
        raise ValueError("v70 file changed")
    sub = pd.read_parquet(V70[0])
    rec = pd.read_parquet(RECIPE, columns=[ID, "airport", "month", "scope", "v9"])
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"] == "DEP"][[ID, "MVT_TIME_UTC_mvt"]]
    k = sub.rename(columns={TARGET: "value"}).merge(rec, on=ID, how="left").merge(raw, on=ID, how="left")
    if len(k) != len(sub) or k[["v9", "MVT_TIME_UTC_mvt", "scope"]].isna().any().any():
        raise ValueError("ranking join incomplete")
    k["value"] = k["value"].astype(float)
    k["ts"] = (k["MVT_TIME_UTC_mvt"] - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds()
    k["date"] = k["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    k["matched"] = k["scope"].isin(["matched_no_lirf", "lirf_nonpop_matched"]).to_numpy()
    k["fixed"] = k["scope"].eq("fixed").to_numpy() | k[ID].isin(GUARD).to_numpy()
    return k


# ---------------------------------------------------------------- run

def main() -> None:
    decl = json.loads(DECL.read_text(encoding="utf-8"))
    cands = {c["name"]: float(c["weight"]) for c in decl["candidates"]}
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    d, f, k = dev_table(), fresh_table(), ranking_table()
    for t in (d, f, k):
        t["x"] = state(t)
        t["scope_rows"] = ~t["matched"].to_numpy() & ~t["fixed"].to_numpy() & ~np.isnan(t["x"].to_numpy())
    y = (d["label"] - d["value"]).to_numpy()
    airports = sorted(d["airport"].unique())
    # development slopes: month-blocked; final slopes: all four months
    b_dev = np.zeros(len(d))
    final, blocked = {}, {}
    for ap in airports:
        a = d["scope_rows"].to_numpy() & (d["airport"] == ap).to_numpy()
        if a.sum() >= MIN_FIT:
            final[ap] = slope(d["x"].to_numpy()[a], y[a])
        for m in DEV_MONTHS:
            fit, tgt = a & (d["month"] != m).to_numpy(), a & (d["month"] == m).to_numpy()
            if fit.sum() >= MIN_FIT and tgt.any():
                blocked[f"{ap} {m}"] = slope(d["x"].to_numpy()[fit], y[fit])
                b_dev[tgt] = blocked[f"{ap} {m}"]
    b_f = f["airport"].map(final).fillna(0.0).to_numpy(dtype=float)
    b_k = k["airport"].map(final).fillna(0.0).to_numpy(dtype=float)
    out = {"experiment_id": decl["experiment_id"], "declaration_sha256": sha(DECL), "audit_used": False, "official_scores_used": "none",
           "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "inputs": {
               "dev_proxy": DEV[1], "v70": V70[1], "lomo_rows": sha(LOMO), "rows12": sha(ROWS12), "recipe": sha(RECIPE)},
           "slopes_final": final, "slopes_blocked": blocked,
           "scope_rows": {"dev": int(d["scope_rows"].sum()), "fresh": int(f["scope_rows"].sum()), "ranking": int(k["scope_rows"].sum())},
           "candidates": {}}
    for name, w in cands.items():
        c = {"weight": w, "dev": {}, "fresh": {}}
        nd = apply(d["value"].to_numpy(dtype=float), d["x"].to_numpy(), b_dev, w, d["scope_rows"].to_numpy())
        rng = np.random.default_rng(SEED)
        for m in DEV_MONTHS:
            s = (d["month"] == m).to_numpy()
            c["dev"][m] = month_row(d[s], nd[s], rng)
        nf = apply(f["value"].to_numpy(dtype=float), f["x"].to_numpy(), b_f, w, f["scope_rows"].to_numpy())
        rng = np.random.default_rng(SEED)
        for m in FRESH:
            s = (f["month"] == m).to_numpy()
            c["fresh"][m] = month_row(f[s], nf[s], rng)
        nk = apply(k["value"].to_numpy(dtype=float), k["x"].to_numpy(), b_k, w, k["scope_rows"].to_numpy())
        guard = {}
        for ap in airports:
            sd = d["scope_rows"].to_numpy() & (d["airport"] == ap).to_numpy() & (nd != d["value"].to_numpy())
            sk = k["scope_rows"].to_numpy() & (k["airport"] == ap).to_numpy()
            if ap not in final or not sk.any():
                continue
            rd = float(np.sqrt(((nd - d["value"].to_numpy())[d["scope_rows"].to_numpy() & (d["airport"] == ap).to_numpy()] ** 2).mean()))
            rk = float(np.sqrt(((nk - k["value"].to_numpy())[sk] ** 2).mean()))
            guard[ap] = {"rms_dev": rd, "rms_rank": rk, "ratio": rk / rd if rd > 0 else None, "rows_rank": int(sk.sum()),
                         "ok": bool(rd > 0 and 0.5 <= rk / rd <= 2.0), "rows_dev_changed": int(sd.sum())}
        c["transfer"] = guard
        c1 = all(c["dev"][m]["pct"] <= 0.005 for m in DEV_MONTHS)
        eq = float(np.mean([c["dev"][m]["pct"] for m in EQUAL]))
        c3 = all(c["dev"][m]["hi"] < 0 for m in EQUAL)
        c4 = all(c["fresh"][m]["pct"] < 0 for m in FRESH) and sum(c["fresh"][m]["hi"] < 0 for m in FRESH) >= 2
        c.update({"clause_1": c1, "eq_mean_pct": eq, "clause_2": eq <= -0.02, "clause_3": c3, "clause_4": c4,
                  "passes": bool(c1 and eq <= -0.02 and c3 and c4), "excluded_airports": sorted(a for a, g in guard.items() if not g["ok"])})
        keep = k["airport"].isin(c["excluded_airports"]).to_numpy()
        nk2 = np.where(keep, k["value"].to_numpy(dtype=float), nk)
        c["ranking"] = {"rows_changed": int((np.rint(nk2) != k["value"].to_numpy()).sum()),
                        "squared_change": float(((np.rint(nk2) - k["value"].to_numpy()) ** 2).sum()),
                        "by_airport_rows": {a: int(((np.rint(nk2) != k["value"].to_numpy()) & (k["airport"] == a).to_numpy()).sum()) for a in airports}}
        c["_nk"] = nk2
        out["candidates"][name] = c
        print(name, {m: round(v["pct"], 4) for m, v in c["dev"].items()}, "eq", round(eq, 4), "fresh",
              {m: (round(v["pct"], 4), round(v["lo"], 4), round(v["hi"], 4)) for m, v in c["fresh"].items()},
              "passes", c["passes"], "excluded", c["excluded_airports"], flush=True)
    passing = [n for n, c in out["candidates"].items() if c["passes"]]
    sel = min(passing, key=lambda n: (out["candidates"][n]["eq_mean_pct"], cands[n])) if passing else None
    out["selected"] = sel
    if sel:
        sub = pd.read_parquet(V70[0])
        nk2 = out["candidates"][sel]["_nk"]
        vals = sub.set_index(ID)[TARGET]
        newv = pd.Series(np.rint(nk2), index=k[ID].to_numpy()).reindex(sub[ID].to_numpy()).to_numpy()
        res = sub.copy()
        res[TARGET] = newv.astype(sub[TARGET].dtype)
        fixed_ids = set(k.loc[k["fixed"], ID])
        if (res.loc[res[ID].isin(fixed_ids), TARGET].to_numpy() != vals.reindex(res.loc[res[ID].isin(fixed_ids), ID]).to_numpy()).any():
            raise ValueError("a fixed row moved")
        DRY.parent.mkdir(parents=True, exist_ok=True)
        res.to_parquet(DRY, index=False)
        back = pd.read_parquet(DRY)
        changed = int((back[TARGET].to_numpy() != sub[TARGET].to_numpy()).sum())
        out["dry_run"] = {"file": DRY.relative_to(ROOT).as_posix(), "sha256": sha(DRY), "rows_changed_vs_v70": changed,
                          "squared_change_vs_v70": float(((back[TARGET].to_numpy(dtype=float) - sub[TARGET].to_numpy(dtype=float)) ** 2).sum()),
                          "guard_rows_unchanged": bool((back.loc[back[ID].isin(GUARD), TARGET].to_numpy() == sub.loc[sub[ID].isin(GUARD), TARGET].to_numpy()).all()),
                          "template_order_same": bool((back[ID].to_numpy() == sub[ID].to_numpy()).all()), "dtype": str(back[TARGET].dtype)}
        print("dry run", out["dry_run"], flush=True)
    for c in out["candidates"].values():
        c.pop("_nk", None)
    (OUT / "results_v1.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print("selected", sel)


if __name__ == "__main__":
    main()
