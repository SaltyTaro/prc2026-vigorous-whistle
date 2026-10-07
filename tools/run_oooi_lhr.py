"""Governing run of the declared round oooi_lhr_v1 (configs/oooi_lhr_v1.json): Heathrow data-link OUT pin on rows ADS-B
did not trace from the stand (dark and unmatched).

Steps (nothing is uploaded, nothing is written under runs/submissions):
  1. control: v70's development proxy = the v69 proxy with oooi_pin_v1 w50 applied through tools/run_oooi_pin.py's
     functions; it must reproduce docs/oooi_pin/results_v1.json's w50 complete-month RMSEs exactly (else stop);
     saved as runs/oooi_lhr/dev_v70_proxy.parquet;
  2. EGLL_family pins from runs/oooi_ext/scan/formats.parquet (84df504's formats; BA_MVT16, UA_OUT13, MVA_AD, QP_VS
     with the screen's membership and priority order; uniform guard); scope = EGLL, guarded family pin, ADS-B class
     dark or unmatched (matches2 tables, as tools/hunt_census.py); fixed rows keep the chain;
  3. clauses 1-3 on November / December 2025 (offsets Nov <- Dec, Dec <- Nov; pooled family fallback), clause 4 on
     October 2025 against rows12 'stack' (offsets from Nov + Dec; October labels read from rows12 for this step only;
     the formats table keeps them blanked), clause 5 ranking-vs-development change RMS;
  4. if accepted: ranking dry run from the uploaded v70 into runs/oooi_lhr/rank/dry_v71/vigorous-whistle_v71.parquet.

Usage: .venv\\Scripts\\python.exe -X utf8 -B tools/run_oooi_lhr.py
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
import run_intruder_bound as R  # noqa: E402
import run_oooi_pin as P  # noqa: E402
import run_stack_resid as S  # noqa: E402

ID = "MVT_ID_mvt"
CONFIG = ROOT / "configs" / "oooi_lhr_v1.json"
OUT = ROOT / "runs" / "oooi_lhr"
FORMATS = ROOT / "runs" / "oooi_ext" / "scan" / "formats.parquet"
FORMATS_SHA = "822b7cf5b823e53763af8184caa82e398e97909513a2bba732ffaf2702089c71"
PIN_RESULTS = ROOT / "docs" / "oooi_pin" / "results_v1.json"
MATCHES = ROOT / "data" / "external" / "adsblol" / "matches"
V70 = ROOT / "runs" / "submissions" / "vigorous-whistle_v70.parquet"
V70_SHA = "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f"
RESULTS = ROOT / "docs" / "oooi_lhr" / "results_v1.json"
DRY = OUT / "rank" / "dry_v71"
FAM_ORDER = ["UA_OUT13", "BA_MVT16", "MVA_AD", "QP_VS"]
DEV_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
SRC_MONTHS = ["2025-11", "2025-12"]
FRESH = "2025-10"
W, FLOOR, SEED = 0.5, -12.0, 20261004
CODE = ["tools/run_oooi_lhr.py", "tools/run_oooi_pin.py", "tools/oooi_ext_formats.py", "tools/oooi_ext_scan.py",
        "tools/oooi_ext_extract.py", "tools/oooi_ext_pack.py", "tools/oooi_ext_screen.py", "tools/hunt_census.py",
        "tools/run_intruder_bound.py", "tools/run_stack_resid.py", "configs/oooi_lhr_v1.json"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def klass(ids: pd.Series, months: list[str]) -> pd.Series:
    """ADS-B class exactly as tools/hunt_census.py: matches2 tables of the months, first row per ID; gate True -> gate,
    gate False -> dark, no match -> unm."""
    m = pd.concat([pd.read_parquet(MATCHES / f"matches2_{x}.parquet", columns=[ID, "gate"]) for x in months]).drop_duplicates(ID)
    g = pd.DataFrame({ID: ids.to_numpy()}).merge(m, on=ID, how="left")["gate"]
    return pd.Series(np.where(g.eq(True), "gate", np.where(g.eq(False), "dark", "unm")), index=ids.index)


def family() -> pd.DataFrame:
    """The screen's EGLL_family (tools/oooi_ext_screen.py): guarded rows of the four formats, QP_VS = Virgin's QP_ORIG,
    one pin per departure in FAM_ORDER priority."""
    f = pd.read_parquet(FORMATS).rename(columns={"label": "acars_label"})
    f = pd.concat([f, f[f["format"].eq("QP_ORIG") & f["al2"].eq("VS")].assign(format="QP_VS")], ignore_index=True)
    fam = f[f["format"].isin(FAM_ORDER) & f["guard"]].copy()
    fam["prio"] = fam["format"].map({k: i for i, k in enumerate(FAM_ORDER)})
    fam = fam.sort_values([ID, "prio"]).drop_duplicates(ID)
    fam = fam[fam["ADEP_mvt"].eq("EGLL")].copy()
    fam["raw"] = (fam["tko"] - fam["t1"]).dt.total_seconds()
    return fam[[ID, "format", "src", "dep_month", "raw", "tko"]].reset_index(drop=True)


def v70_dev_proxy(res: dict) -> pd.DataFrame:
    """v69 development proxy + oooi_pin_v1 w50 through run_oooi_pin's own functions (development part of its main)."""
    if sha(P.PROXY) != P.PROXY_SHA:
        raise ValueError("The v69 proxy differs from its declared SHA-256.")
    pin_res = json.loads(PIN_RESULTS.read_text())
    if sha(P.RUN / "joined_dev.parquet") != pin_res["joined_sha256"]["dev"]:
        raise ValueError("runs/oooi_pin/joined_dev.parquet changed since oooi_pin_v1's governing run.")
    pin_dev = P.scope(pd.read_parquet(P.RUN / "joined_dev.parquet"))
    dev = pd.read_parquet(P.PROXY, columns=[ID, "month", "airport", "label", "cls", "day", "value"])
    dev = dev.merge(P.takeoff_days(P.DEV_MONTHS), on=ID, how="left", validate="one_to_one")
    dev["day"] = dev["tday"]
    dev = dev.merge(pin_dev[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    fixed = S.fixed_rows(dev)
    dev["in_scope"] = dev["raw"].notna().to_numpy() & ~fixed & dev["airport"].eq("LFPG").to_numpy()
    dev["pin"] = np.nan
    for m in P.PIN_DEV_MONTHS:
        off = P.offsets(dev[dev["in_scope"] & dev["month"].ne(m) & dev["month"].isin(P.PIN_DEV_MONTHS)])
        sel = dev["in_scope"] & dev["month"].eq(m)
        dev.loc[sel, "pin"] = dev.loc[sel, "raw"] + dev.loc[sel, "source"].map(off)
    v = dev["value"].to_numpy(dtype=float).copy()
    sc = dev["in_scope"].to_numpy()
    v[sc] = P.cand_value(0.5, dev.loc[sc, "pin"].to_numpy(), v[sc])
    out = dev[[ID, "month", "airport", "label", "cls", "day"]].copy()
    out["value"] = v
    want = {m: d["rmse"] for m, d in pin_res["candidates"]["w50"]["development"].items()}
    got = {m: float(np.sqrt(np.mean((out.loc[out["month"].eq(m), "label"].to_numpy()
                                     - out.loc[out["month"].eq(m), "value"].to_numpy()) ** 2))) for m in DEV_MONTHS}
    res["control_rebuild"] = {"want_w50_rmse": want, "got_rmse": got, "exact": all(want[m] == got[m] for m in DEV_MONTHS),
                              "rows_moved_vs_v69_proxy": int(sc.sum())}
    if not res["control_rebuild"]["exact"]:
        raise ValueError(f"v70's development proxy does not reproduce results_v1.json: {want} vs {got}")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "dev_v70_proxy.parquet", index=False)
    res["control_rebuild"]["path"] = "runs/oooi_lhr/dev_v70_proxy.parquet"
    res["control_rebuild"]["sha256"] = sha(OUT / "dev_v70_proxy.parquet")
    return out


def fit_offsets(rows: pd.DataFrame) -> tuple[pd.Series, float]:
    r = rows["label"] - rows["raw"]
    return r.groupby(rows["format"]).mean(), float(r.mean())


def apply_pin(rows: pd.DataFrame, off: pd.Series, pooled: float) -> pd.Series:
    return rows["raw"] + rows["format"].map(off).fillna(pooled)


def month_stats(s: pd.DataFrame, rng: np.random.Generator) -> dict:
    pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
    lab, c0, c1 = s["label"].to_numpy(), s["ctl"].to_numpy(), s["cand"].to_numpy()
    ch = c1 - c0
    return {"rows": int(len(s)), "control_rmse": float(np.sqrt(np.mean((lab - c0) ** 2))), "rmse": float(np.sqrt(np.mean((lab - c1) ** 2))),
            "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
            "sse_gain_s2": float(((lab - c0) ** 2).sum() - ((lab - c1) ** 2).sum()), "max_abs_change_s": float(np.abs(ch).max())}


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    if sha(FORMATS) != FORMATS_SHA or sha(V70) != V70_SHA:
        raise ValueError("The formats table or v70 differs from the expected SHA-256.")
    res = {"experiment_id": cfg["experiment_id"], "computed_at_utc": utc_now(), "config_sha256": sha(CONFIG),
           "official_scores_used": "none", "audit_used": False, "code_sha256": {p: sha(ROOT / p) for p in CODE},
           "inputs_sha256": {"formats": FORMATS_SHA, "v69_proxy": P.PROXY_SHA, "v70": V70_SHA, "rows12": sha(P.ROWS12),
                             **{f"matches2_{m}": sha(MATCHES / f"matches2_{m}.parquet") for m in
                                ["2025-01", "2025-07", "2025-10", "2025-11", "2025-12", "2026-01", "2026-07"]}}}
    ctl = v70_dev_proxy(res)
    fam = family()

    # development scope
    dv = fam[fam["src"].eq("train") & fam["dep_month"].isin(SRC_MONTHS)].merge(
        ctl[[ID, "month", "label", "cls", "airport"]], on=ID, how="inner")
    dv["k"] = klass(dv[ID], ["2025-01", "2025-07", "2025-11", "2025-12"])
    fixed = S.fixed_rows(dv)
    dv["in_scope"] = dv["k"].isin(["dark", "unm"]).to_numpy() & ~fixed & dv["airport"].eq("EGLL").to_numpy()
    res["dev_family_rows_by_month_class"] = {f"{m}|{k}": int(v) for (m, k), v in dv.groupby(["month", "k"]).size().items()}
    ins = dv[dv["in_scope"]].copy()
    ins["pin"] = np.nan
    res["dev_offsets"] = {}
    for m, other in [("2025-11", "2025-12"), ("2025-12", "2025-11")]:
        off, pooled = fit_offsets(ins[ins["month"].eq(other)])
        res["dev_offsets"][m] = {"fitted_on": other, "by_format": {k: float(v) for k, v in off.items()}, "pooled": pooled,
                                 "formats_using_pooled": sorted(set(ins.loc[ins["month"].eq(m), "format"]) - set(off.index))}
        sel = ins["month"].eq(m)
        ins.loc[sel, "pin"] = apply_pin(ins[sel], off, pooled)
    t = ctl[[ID, "month", "label", "day"]].copy()
    t["ctl"] = ctl["value"].to_numpy(dtype=float)
    t["cand"] = t["ctl"]
    pinmap = ins.set_index(ID)["pin"]
    sc = t[ID].isin(pinmap.index).to_numpy()
    t.loc[sc, "cand"] = np.maximum(W * t.loc[sc, ID].map(pinmap).to_numpy() + (1 - W) * t.loc[sc, "ctl"].to_numpy(), FLOOR)
    rng = np.random.default_rng(SEED)
    months = {m: month_stats(t[t["month"] == m], rng) for m in DEV_MONTHS}
    dch = (t["cand"] - t["ctl"]).to_numpy()[sc]
    mean_nd = float(np.mean([months[m]["pct"] for m in SRC_MONTHS]))

    # fresh October 2025 (labels from rows12 for this scoring step only; the formats table stays blanked)
    off_nd, pooled_nd = fit_offsets(ins)
    res["offsets_nov_dec"] = {"by_format": {k: float(v) for k, v in off_nd.items()}, "pooled": pooled_nd}
    fr = pd.read_parquet(P.ROWS12, columns=[ID, "month", "airport", "label", "stack", "cls", "population"])
    fr = fr[fr["month"].eq(FRESH)].copy()
    fr = fr.merge(P.takeoff_days([FRESH]), on=ID, how="left", validate="one_to_one")
    fr["day"] = fr["tday"]
    fo = fam[fam["src"].eq("train") & fam["dep_month"].eq(FRESH)].copy()
    fo["k"] = klass(fo[ID], [FRESH])
    fo = fo.merge(fr[[ID, "airport", "cls", "population", "label"]], on=ID, how="inner")
    ffixed = S.fixed_rows(fo) | fo["population"].fillna(False).to_numpy(dtype=bool)
    fo = fo[fo["k"].isin(["dark", "unm"]).to_numpy() & ~ffixed & fo["airport"].eq("EGLL").to_numpy()].copy()
    fo["pin"] = apply_pin(fo, off_nd, pooled_nd)
    f = fr[[ID, "month", "label", "day"]].copy()
    f["ctl"] = fr["stack"].to_numpy(dtype=float)
    f["cand"] = f["ctl"]
    fmap = fo.set_index(ID)["pin"]
    fs = f[ID].isin(fmap.index).to_numpy()
    f.loc[fs, "cand"] = np.maximum(W * f.loc[fs, ID].map(fmap).to_numpy() + (1 - W) * f.loc[fs, "ctl"].to_numpy(), FLOOR)
    fstat = month_stats(f, np.random.default_rng(SEED))
    fstat["pinned_rows"] = int(fs.sum())
    fstat["pinned_by_format"] = {k: int(v) for k, v in fo["format"].value_counts().items()}
    fstat["pinned_by_class"] = {k: int(v) for k, v in fo["k"].value_counts().items()}
    fstat["share_pin_abs_err_gt_1800"] = float(((fo["pin"] - fo["label"]).abs() > 1800).mean()) if len(fo) else None
    fstat["october_labels_unblanked"] = "October 2025 labels read from runs/daily_totals/rows12.parquet for clause 4 only; runs/oooi_ext/scan/formats.parquet unchanged (block / label still blank)"

    # ranking
    v70 = pd.read_parquet(V70)
    rk = pd.read_parquet(P.RANKING, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    if list(v70.columns) != [ID, "TAXITIME_SEC_mvt"] or len(v70) != len(rk):
        raise ValueError("v70 is not a full submission.")
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    rfixed = set(rk.loc[pop, ID]) | set(R.GUARD) | set(rk.loc[rk["ADEP_mvt"].isin(["LIRF", "LFPG"]), ID])
    rf = fam[fam["src"].eq("rank")].copy()
    rf["k"] = klass(rf[ID], ["2026-01", "2026-07"])
    rf = rf[rf["k"].isin(["dark", "unm"]) & ~rf[ID].isin(rfixed)].copy()
    rf["pin"] = apply_pin(rf, off_nd, pooled_nd)
    rkt = v70.merge(rk[[ID, "ADEP_mvt"]], on=ID, how="left", validate="one_to_one")
    rkt["chain"] = rkt["TAXITIME_SEC_mvt"].astype(float)
    rmap = rf.set_index(ID)["pin"]
    rs = rkt[ID].isin(rmap.index).to_numpy() & rkt["ADEP_mvt"].eq("EGLL").to_numpy()
    rv = rkt["chain"].to_numpy().copy()
    rv[rs] = np.maximum(W * rkt.loc[rs, ID].map(rmap).to_numpy() + (1 - W) * rv[rs], FLOOR)
    new_int = np.rint(rv).astype(v70["TAXITIME_SEC_mvt"].dtype)
    rch = new_int.astype(float) - rkt["chain"].to_numpy()
    rank_rms = float(np.sqrt(np.mean(rch[rs] ** 2)))
    dev_rms = float(np.sqrt(np.mean(dch ** 2)))
    ratio = rank_rms / dev_rms
    checks = {"1_no_dev_month_worse_than_0.005pct": all(months[m]["pct"] <= 0.005 for m in DEV_MONTHS),
              "2_mean_nov_dec_pct": mean_nd, "2_ok": mean_nd <= -0.02,
              "3_nov_dec_intervals_below_zero": all(months[m]["hi"] < 0 for m in SRC_MONTHS),
              "4_october_change_below_zero": bool(fstat["pct"] < 0),
              "4_october_share_gt1800_le_2pct": (fstat["share_pin_abs_err_gt_1800"] or 0.0) <= 0.02,
              "5_ratio": ratio, "5_ok": 0.5 <= ratio <= 2.0}
    checks["4_ok"] = bool(checks["4_october_change_below_zero"] and checks["4_october_share_gt1800_le_2pct"])
    checks["passes"] = bool(checks["1_no_dev_month_worse_than_0.005pct"] and checks["2_ok"] and checks["3_nov_dec_intervals_below_zero"]
                            and checks["4_ok"] and checks["5_ok"])
    res["candidate"] = {"name": "lhr_w50", "weight": W, "development": months, "dev_in_scope_rows": int(sc.sum()),
                        "dev_in_scope_by_month_format": {f"{m}|{k}": int(v) for (m, k), v in ins.groupby(["month", "format"]).size().items()},
                        "dev_change_rms_in_scope_s": dev_rms, "october": fstat, "ranking_in_scope_rows": int(rs.sum()),
                        "ranking_change_rms_in_scope_s": rank_rms, "checks": checks}
    res["selected"] = "lhr_w50" if checks["passes"] else None
    print("control rebuild exact:", res["control_rebuild"]["exact"], flush=True)
    print("dev", " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.4f},{months[m]['hi']:+.4f}] n{months[m]['rows_changed']}"
                          for m in DEV_MONTHS), f"| mean Nov/Dec {mean_nd:+.4f}%", flush=True)
    print("october", f"{fstat['pct']:+.4f}% [{fstat['lo']:+.4f},{fstat['hi']:+.4f}] pinned {fstat['pinned_rows']} >1800 {fstat['share_pin_abs_err_gt_1800']}",
          "| ranking rows", int(rs.sum()), f"clause5 {ratio:.3f}", checks, flush=True)

    if res["selected"]:
        out_df = v70.copy()
        out_df["TAXITIME_SEC_mvt"] = pd.Series(new_int, index=v70.index).astype(v70["TAXITIME_SEC_mvt"].dtype)
        changed = out_df["TAXITIME_SEC_mvt"].ne(v70["TAXITIME_SEC_mvt"])
        if not out_df[ID].equals(v70[ID]) or (changed & ~pd.Series(rs, index=v70.index)).any():
            raise ValueError("Row order changed or a row outside the scope moved.")
        if (changed & v70[ID].isin(rfixed)).any():
            raise ValueError("A LIRF, LFPG, population or guard row moved.")
        DRY.mkdir(parents=True, exist_ok=True)
        path = DRY / "vigorous-whistle_v71.parquet"
        out_df.to_parquet(path, index=False)
        back = pd.read_parquet(path)
        if not back.equals(out_df):
            raise ValueError("The written file does not read back identically.")
        diff = out_df["TAXITIME_SEC_mvt"].astype(float) - v70["TAXITIME_SEC_mvt"].astype(float)
        cen = pd.read_parquet(P.CENSUS, columns=[ID, "k", "own_n", "month"])
        cen["cls"] = np.where(cen["k"].eq("unm"), np.where(cen["own_n"].fillna(0) >= 1, "unm_seen", "unm_unseen"), cen["k"])
        ch = v70.loc[changed, [ID]].merge(cen[[ID, "cls", "month"]], on=ID, how="left").merge(
            rf[[ID, "format"]], on=ID, how="left")
        receipt = {"kind": "dry run, not a submission", "candidate": "lhr_w50", "written_at_utc": utc_now(),
                   "base": "runs/submissions/vigorous-whistle_v70.parquet", "base_sha256": V70_SHA,
                   "file": "runs/oooi_lhr/rank/dry_v71/vigorous-whistle_v71.parquet", "sha256": sha(path), "bytes": path.stat().st_size,
                   "rows_in_scope": int(rs.sum()), "rows_changed": int(changed.sum()), "squared_change_vs_v70": float((diff ** 2).sum()),
                   "max_abs_change_s": float(diff.abs().max()), "max_up_s": float(diff.max()), "max_down_s": float(diff.min()),
                   "mean_change_s": float(diff[changed].mean()) if changed.any() else 0.0,
                   "rows_changed_by_month_class": {f"{m}|{c}": int(v) for (m, c), v in ch.groupby(["month", "cls"]).size().items()},
                   "rows_changed_by_format": {k: int(v) for k, v in ch["format"].value_counts().items()},
                   "lirf_lfpg_population_guard_rows_changed": int((changed & v70[ID].isin(rfixed)).sum()),
                   "lfpg_rows_unchanged": bool(not changed[rkt["ADEP_mvt"].eq("LFPG")].any()),
                   "guard_rows_unchanged": bool(not changed[v70[ID].isin(R.GUARD)].any()),
                   "other_rows_identical_to_v70": bool(out_df.loc[~changed].equals(v70.loc[~changed])),
                   "rounding": "numpy.rint into int32, ties to even"}
        (DRY / "vigorous-whistle_v71.receipt.json").write_text(json.dumps(receipt, indent=1))
        res["dry_run"] = receipt
        print("dry run", {k: receipt[k] for k in ["sha256", "rows_changed", "squared_change_vs_v70", "max_abs_change_s",
                                                   "lirf_lfpg_population_guard_rows_changed", "lfpg_rows_unchanged",
                                                   "other_rows_identical_to_v70"]}, receipt["rows_changed_by_month_class"], flush=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(res, indent=1, default=float))
    print("selected:", res["selected"])


if __name__ == "__main__":
    main()
