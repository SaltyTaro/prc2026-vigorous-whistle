"""Governing run of the declared round oooi_pin_v1 (configs/oooi_pin_v1.json): data-link OUT pin at LFPG.

Steps (nothing is uploaded, nothing is written under runs/submissions):
  1. joins with tools/oooi_pin_join.py (declared logic; only the run directory and month list are parameters):
     development months July / November / December 2025 from runs/oooi_pin/messages_unique.parquet, fresh months
     August / September / October 2025 from runs/oooi_pin/fresh/messages_unique.parquet;
  2. scope = LFPG departures matched to a 2N or 8D report and passing the source guard (verbatim from the screen);
     LIRF / LTFM matches and tools/run_stack_resid.fixed_rows keep the chain;
  3. clauses 1-3 on the v69 development proxy (offsets month-blocked over July / November / December; January 2025
     unchanged), clause 4 on August-October 2025 against runs/daily_totals/rows12.parquet 'stack' (offsets from
     July / November / December), intervals with tools/run_intruder_bound.interval (2,000 replicates over UTC take-off
     days, seed 20261004), clause 5 ranking-vs-development change RMS;
  4. selection as declared; for an accepted candidate a ranking dry run from the uploaded v69 into
     runs/oooi_pin/rank/dry_v70/vigorous-whistle_v70.parquet.

Usage: .venv\\Scripts\\python.exe -X utf8 -B tools/run_oooi_pin.py
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
import oooi_pin_join as J  # noqa: E402
import run_intruder_bound as R  # noqa: E402
import run_stack_resid as S  # noqa: E402

ID = "MVT_ID_mvt"
CONFIG = ROOT / "configs" / "oooi_pin_v1.json"
RUN = ROOT / "runs" / "oooi_pin"
FRESH = RUN / "fresh"
PROXY = ROOT / "runs" / "stack_weight" / "dev" / "w65_bag10" / "proxy.parquet"
PROXY_SHA = "c088e409de153d81b006bcd26b5b2a28e0e306043a5172c2158d0ecc6e6f5cb8"
ROWS12 = ROOT / "runs" / "daily_totals" / "rows12.parquet"
V69 = ROOT / "runs" / "submissions" / "vigorous-whistle_v69.parquet"
V69_SHA = "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665"
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
CENSUS = ROOT / "runs" / "hunt" / "census_rank.parquet"
OUT_DIR = RUN / "rank" / "dry_v70"
RESULTS = ROOT / "docs" / "oooi_pin" / "results_v1.json"
DEV_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
PIN_DEV_MONTHS = ["2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
CLAUSE3 = ["2025-07", "2025-12"]
FRESH_MONTHS = ["2025-08", "2025-09", "2025-10"]
SEED = 20261004
FLOOR = -12.0
CODE = ["tools/run_oooi_pin.py", "tools/oooi_pin_join.py", "tools/oooi_pin_extract.py", "tools/oooi_pin_pack.py",
        "tools/run_intruder_bound.py", "tools/run_stack_resid.py", "configs/oooi_pin_v1.json"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def source_guard(x: pd.DataFrame) -> pd.Series:
    """Label-free, per source: 2N needs its OFF time to confirm the match (|take-off - t2| <= 120 s) and a raw
    estimate >= 300 s; 8D needs a raw estimate in 300..3600 s and, where the NM clock exists, |AOBT_3 - t1| <= 1800 s."""
    raw = (x["MVT_TIME_UTC_mvt"] - x["t1"]).dt.total_seconds()
    d2 = (x["MVT_TIME_UTC_mvt"] - x["t2"]).dt.total_seconds().abs()
    aob = (x["AOBT_3_flt"] - x["t1"]).dt.total_seconds().abs()
    g2n = x["source"].eq("2N") & d2.le(120) & raw.ge(300)
    g8d = x["source"].eq("8D") & raw.between(300, 3600) & (aob.le(1800) | aob.isna())
    return g2n | g8d


def scope(joined: pd.DataFrame) -> pd.DataFrame:
    """In-scope pins: LFPG, 2N or 8D, source guard passed; raw = take-off - OUT (s)."""
    j = joined.rename(columns={"label": "acars_label", "month": "msg_month"})
    j = j[j["ADEP_mvt"].eq("LFPG") & j["source"].isin(["2N", "8D"])].copy()
    j = j[source_guard(j).to_numpy()].copy()
    j["raw"] = (j["MVT_TIME_UTC_mvt"] - j["t1"]).dt.total_seconds()
    if j[ID].duplicated().any():
        raise ValueError("A departure carries two pins.")
    return j[[ID, "source", "raw", "month_dep", "MVT_TIME_UTC_mvt"]].reset_index(drop=True)


def takeoff_days(months: list[str]) -> pd.DataFrame:
    raw = ROOT / "data" / "raw"
    parts = []
    for m in months:
        d = pd.read_parquet(next(raw.glob(f"training_{m}-01_*.parquet")), columns=[ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"])
        parts.append(d[d["PHASE_mvt"].eq("DEP")][[ID, "MVT_TIME_UTC_mvt"]])
    d = pd.concat(parts, ignore_index=True).drop_duplicates(ID)
    d["tday"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m-%d")
    return d[[ID, "tday"]]


def offsets(fit: pd.DataFrame) -> pd.Series:
    """offset(source) = mean over guarded fitting rows of label - (take-off - OUT)."""
    return (fit["label"] - fit["raw"]).groupby(fit["source"]).mean()


def cand_value(w: float, pin: np.ndarray, chain: np.ndarray) -> np.ndarray:
    return np.maximum(w * pin + (1.0 - w) * chain, FLOOR)


def month_stats(s: pd.DataFrame, rng: np.random.Generator) -> dict:
    pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
    ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
    lab = s["label"].to_numpy()
    return {"rows": int(len(s)), "control_rmse": float(np.sqrt(np.mean((lab - s["ctl"].to_numpy()) ** 2))),
            "rmse": float(np.sqrt(np.mean((lab - s["cand"].to_numpy()) ** 2))), "pct": pct, "lo": lo, "hi": hi,
            "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
            "sse_gain_s2": float(((lab - s["ctl"].to_numpy()) ** 2).sum() - ((lab - s["cand"].to_numpy()) ** 2).sum()),
            "max_abs_change_s": float(np.abs(ch).max())}


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    if sha(PROXY) != PROXY_SHA or sha(V69) != V69_SHA:
        raise ValueError("A control file's SHA-256 differs from the declared one.")
    res = {"experiment_id": cfg["experiment_id"], "computed_at_utc": utc_now(), "config_sha256": sha(CONFIG),
           "official_scores_used": "none", "audit_used": False,
           "code_sha256": {p: sha(ROOT / p) for p in CODE},
           "inputs_sha256": {"proxy": PROXY_SHA, "rows12": sha(ROWS12), "v69": V69_SHA,
                             "messages_dev": sha(RUN / "messages_unique.parquet"),
                             "messages_fresh": sha(FRESH / "messages_unique.parquet")}}

    # 1. joins (declared logic, month list as parameter)
    J.main(RUN, PIN_DEV_MONTHS)
    J.main(FRESH, FRESH_MONTHS)
    pin_dev = scope(pd.read_parquet(RUN / "joined_dev.parquet"))
    pin_fr = scope(pd.read_parquet(FRESH / "joined_dev.parquet"))
    pin_rk = scope(pd.read_parquet(RUN / "joined_rank.parquet"))
    res["joined_sha256"] = {"dev": sha(RUN / "joined_dev.parquet"), "fresh": sha(FRESH / "joined_dev.parquet"),
                            "rank": sha(RUN / "joined_rank.parquet")}
    res["scope_rows"] = {"dev": {k: {s_: int(v) for s_, v in g["source"].value_counts().items()}
                                 for k, g in pin_dev.groupby("month_dep")},
                         "fresh": {k: {s_: int(v) for s_, v in g["source"].value_counts().items()}
                                   for k, g in pin_fr.groupby("month_dep")},
                         "rank": {k: {s_: int(v) for s_, v in g["source"].value_counts().items()}
                                  for k, g in pin_rk.groupby("month_dep")}}

    # 2. development table (v69 proxy, every row of the four months)
    dev = pd.read_parquet(PROXY, columns=[ID, "month", "airport", "label", "cls", "day", "value"])
    dev = dev.merge(takeoff_days(DEV_MONTHS), on=ID, how="left", validate="one_to_one")
    if dev["tday"].isna().any():
        raise ValueError("Take-off days do not cover the development rows.")
    res["dev_proxy_day_equals_takeoff_day_share"] = float((dev["day"].astype(str) == dev["tday"]).mean())
    dev["day"] = dev["tday"]
    dev = dev.merge(pin_dev[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    fixed = S.fixed_rows(dev)
    dev["in_scope"] = dev["raw"].notna().to_numpy() & ~fixed & dev["airport"].eq("LFPG").to_numpy()
    if pin_dev[ID].isin(dev[ID]).sum() != len(pin_dev):
        raise ValueError("A pinned development row is missing from the proxy.")
    res["dev_fixed_rows_in_scope"] = int((dev["raw"].notna().to_numpy() & fixed).sum())
    dev["pin"] = np.nan
    for m in PIN_DEV_MONTHS:
        fit = dev[dev["in_scope"] & dev["month"].ne(m) & dev["month"].isin(PIN_DEV_MONTHS)]
        off = offsets(fit)
        res.setdefault("dev_offsets_month_blocked", {})[m] = {k: float(v) for k, v in off.items()}
        sel = dev["in_scope"] & dev["month"].eq(m)
        dev.loc[sel, "pin"] = dev.loc[sel, "raw"] + dev.loc[sel, "source"].map(off)

    # 3. fresh table (rows12 'stack', every row of August-October 2025)
    fr = pd.read_parquet(ROWS12, columns=[ID, "month", "airport", "label", "stack", "cls", "population"])
    fr = fr[fr["month"].isin(FRESH_MONTHS)].copy()
    fr = fr.merge(takeoff_days(FRESH_MONTHS), on=ID, how="left", validate="one_to_one")
    if fr["tday"].isna().any():
        raise ValueError("Take-off days do not cover the fresh rows.")
    fr["day"] = fr["tday"]
    fr = fr.merge(pin_fr[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    ffixed = S.fixed_rows(fr) | fr["population"].fillna(False).to_numpy(dtype=bool)
    fr["in_scope"] = fr["raw"].notna().to_numpy() & ~ffixed & fr["airport"].eq("LFPG").to_numpy()
    res["fresh_pins_missing_from_rows12"] = int((~pin_fr[ID].isin(fr[ID])).sum())
    off_all = offsets(dev[dev["in_scope"] & dev["month"].isin(PIN_DEV_MONTHS)])
    res["offsets_jul_nov_dec"] = {k: float(v) for k, v in off_all.items()}
    fr["pin"] = np.where(fr["in_scope"], fr["raw"] + fr["source"].map(off_all), np.nan)

    # 4. ranking
    v69 = pd.read_parquet(V69)
    rk = pd.read_parquet(RANKING, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    if list(v69.columns) != [ID, "TAXITIME_SEC_mvt"] or len(v69) != len(rk):
        raise ValueError("v69 is not a full submission.")
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    rfixed_ids = set(rk.loc[pop, ID]) | set(R.GUARD) | set(rk.loc[rk["ADEP_mvt"].eq("LIRF"), ID])
    rkt = v69.merge(rk[[ID, "ADEP_mvt"]], on=ID, how="left", validate="one_to_one")
    rkt = rkt.merge(pin_rk[[ID, "source", "raw", "month_dep"]], on=ID, how="left", validate="one_to_one")
    rkt["in_scope"] = rkt["raw"].notna() & rkt["ADEP_mvt"].eq("LFPG") & ~rkt[ID].isin(rfixed_ids)
    rkt["pin"] = np.where(rkt["in_scope"], rkt["raw"] + rkt["source"].map(off_all), np.nan)
    rkt["chain"] = rkt["TAXITIME_SEC_mvt"].astype(float)

    # 5. candidates
    res["candidates"] = {}
    for c in cfg["candidates"]:
        name, w = c["name"], float(c["weight"])
        out = {"weight": w}
        # development
        t = dev[[ID, "month", "label", "day"]].copy()
        t["ctl"] = dev["value"].to_numpy(dtype=float)
        t["cand"] = t["ctl"]
        sc = dev["in_scope"].to_numpy()
        t.loc[sc, "cand"] = cand_value(w, dev.loc[sc, "pin"].to_numpy(), t.loc[sc, "ctl"].to_numpy())
        rng = np.random.default_rng(SEED)
        months = {m: month_stats(t[t["month"] == m], rng) for m in DEV_MONTHS}
        eq = float(np.mean([months[m]["pct"] for m in EQUAL]))
        dch = (t["cand"] - t["ctl"]).to_numpy()[sc]
        out["development"] = months
        out["dev_change_rms_in_scope_s"] = float(np.sqrt(np.mean(dch ** 2)))
        out["dev_in_scope_rows"] = int(sc.sum())
        # fresh
        f = fr[[ID, "month", "label", "day"]].copy()
        f["ctl"] = fr["stack"].to_numpy(dtype=float)
        f["cand"] = f["ctl"]
        fs = fr["in_scope"].to_numpy()
        f.loc[fs, "cand"] = cand_value(w, fr.loc[fs, "pin"].to_numpy(), f.loc[fs, "ctl"].to_numpy())
        rng4 = np.random.default_rng(SEED)
        fm = {}
        for m in FRESH_MONTHS:
            s = f[f["month"] == m]
            fm[m] = month_stats(s, rng4)
            pm = fs[(f["month"] == m).to_numpy()]
            pins = fr.loc[(fr["month"] == m) & fr["in_scope"], ["pin", "label"]]
            fm[m]["pinned_rows"] = int(pm.sum())
            fm[m]["share_pin_abs_err_gt_1800"] = float(((pins["pin"] - pins["label"]).abs() > 1800).mean()) if len(pins) else None
        out["fresh"] = fm
        out["fresh_change_rms_in_scope_s"] = float(np.sqrt(np.mean(((f["cand"] - f["ctl"]).to_numpy()[fs]) ** 2)))
        # ranking (file values, rounded with numpy.rint into the file's dtype)
        rs = rkt["in_scope"].to_numpy()
        rv = rkt["chain"].to_numpy().copy()
        rv[rs] = cand_value(w, rkt.loc[rs, "pin"].to_numpy(), rkt.loc[rs, "chain"].to_numpy())
        new_int = np.rint(rv).astype(v69["TAXITIME_SEC_mvt"].dtype)
        rch = new_int.astype(float) - rkt["chain"].to_numpy()
        out["ranking_in_scope_rows"] = int(rs.sum())
        out["ranking_change_rms_in_scope_s"] = float(np.sqrt(np.mean(rch[rs] ** 2)))
        out["ranking_change_rms_in_scope_unrounded_s"] = float(np.sqrt(np.mean((rv - rkt["chain"].to_numpy())[rs] ** 2)))
        ratio = out["ranking_change_rms_in_scope_s"] / out["dev_change_rms_in_scope_s"]
        out["clause5_ratio"] = ratio
        checks = {
            "1_no_dev_month_worse_than_0.005pct": all(months[m]["pct"] <= 0.005 for m in DEV_MONTHS),
            "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -0.02,
            "3_jul_dec_intervals_below_zero": all(months[m]["hi"] < 0 for m in CLAUSE3),
            "4_every_fresh_month_better": all(fm[m]["pct"] < 0 for m in FRESH_MONTHS),
            "4_fresh_intervals_below_zero": int(sum(fm[m]["hi"] < 0 for m in FRESH_MONTHS)),
            "4_ok": all(fm[m]["pct"] < 0 for m in FRESH_MONTHS) and sum(fm[m]["hi"] < 0 for m in FRESH_MONTHS) >= 2,
            "5_ratio_in_0.5_2": 0.5 <= ratio <= 2.0}
        checks["passes"] = bool(checks["1_no_dev_month_worse_than_0.005pct"] and checks["2_mean_ok"]
                                and checks["3_jul_dec_intervals_below_zero"] and checks["4_ok"] and checks["5_ratio_in_0.5_2"])
        out["checks"] = checks
        out["_rank_values"] = new_int
        res["candidates"][name] = out
        print(name, "dev", " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.4f},{months[m]['hi']:+.4f}]"
                                     for m in DEV_MONTHS), f"eq {eq:+.4f}%", flush=True)
        print("   fresh", " ".join(f"{m[5:]} {fm[m]['pct']:+.4f}% [{fm[m]['lo']:+.4f},{fm[m]['hi']:+.4f}] n{fm[m]['pinned_rows']} "
                                   f">1800 {fm[m]['share_pin_abs_err_gt_1800']}" for m in FRESH_MONTHS),
              f"| clause5 {ratio:.3f}", checks, flush=True)

    # 6. selection
    passing = [(res["candidates"][n]["checks"]["2_equal_footing_mean_pct"], res["candidates"][n]["weight"], n)
               for n in res["candidates"] if res["candidates"][n]["checks"]["passes"]]
    selected = sorted(passing)[0][2] if passing else None
    res["selected"] = selected

    # 7. dry run for the selected candidate
    if selected is not None:
        new_int = res["candidates"][selected]["_rank_values"]
        out_df = v69.copy()
        out_df["TAXITIME_SEC_mvt"] = pd.Series(new_int, index=v69.index).astype(v69["TAXITIME_SEC_mvt"].dtype)
        if not out_df[ID].equals(v69[ID]):
            raise ValueError("Row order changed.")
        changed = out_df["TAXITIME_SEC_mvt"].ne(v69["TAXITIME_SEC_mvt"])
        fixed_r = v69[ID].isin(rfixed_ids)
        if (changed & fixed_r).any() or (changed & ~rkt["in_scope"]).any():
            raise ValueError("A row outside the scope, a LIRF, population or guard row moved.")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / "vigorous-whistle_v70.parquet"
        out_df.to_parquet(path, index=False)
        back = pd.read_parquet(path)
        if not back.equals(out_df):
            raise ValueError("The written file does not read back identically.")
        diff = out_df["TAXITIME_SEC_mvt"].astype(float) - v69["TAXITIME_SEC_mvt"].astype(float)
        cen = pd.read_parquet(CENSUS, columns=[ID, "k", "own_n", "month"])
        cen["cls"] = np.where(cen["k"].eq("unm"), np.where(cen["own_n"].fillna(0) >= 1, "unm_seen", "unm_unseen"), cen["k"])
        ch = v69.loc[changed, [ID]].merge(cen[[ID, "cls", "month"]], on=ID, how="left")
        receipt = {"kind": "dry run, not a submission", "candidate": selected, "written_at_utc": utc_now(),
                   "base": str(V69.relative_to(ROOT)), "base_sha256": V69_SHA, "file": str(path.relative_to(ROOT)),
                   "sha256": sha(path), "bytes": path.stat().st_size, "rows_in_scope": int(rkt["in_scope"].sum()),
                   "rows_changed": int(changed.sum()), "squared_change": float((diff ** 2).sum()),
                   "max_abs_change_s": float(diff.abs().max()), "max_up_s": float(diff.max()), "max_down_s": float(diff.min()),
                   "mean_change_s": float(diff[changed].mean()) if changed.any() else 0.0,
                   "rows_changed_by_month_class": {f"{m}|{c}": int(v) for (m, c), v in ch.groupby(["month", "cls"]).size().items()},
                   "rows_changed_by_month": {m: int(v) for m, v in ch.groupby("month").size().items()},
                   "lirf_population_guard_lirf_rows_changed": int((changed & fixed_r).sum()),
                   "guard_rows_unchanged": bool(not changed[v69[ID].isin(R.GUARD)].any()),
                   "other_rows_identical_to_v69": bool(out_df.loc[~changed].equals(v69.loc[~changed])),
                   "rounding": "numpy.rint into int32, ties to even"}
        (OUT_DIR / "vigorous-whistle_v70.receipt.json").write_text(json.dumps(receipt, indent=1))
        res["dry_run"] = receipt
        print("dry run", {k: receipt[k] for k in ["sha256", "rows_changed", "squared_change", "max_abs_change_s",
                                                   "lirf_population_guard_lirf_rows_changed", "other_rows_identical_to_v69"]},
              receipt["rows_changed_by_month_class"], flush=True)
    for n in res["candidates"]:
        res["candidates"][n].pop("_rank_values")
    RESULTS.write_text(json.dumps(res, indent=1, default=float))
    print("selected:", selected)


if __name__ == "__main__":
    main()
