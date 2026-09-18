"""Read-only audit of an uploaded ranking export against the development months.

Checks the file itself (row set, types, floor, hash), the shape of the predictions per
airport-month and per airport-day against the incumbent's development predictions and
labels on Jan/Jul/Nov 2025, the extreme predictions, the negative-proxy rows, the LIRF
population, the ADS-B-informed rows (prediction against the ADS-B-implied taxi time),
the no-NM rows, and the differences to earlier exports. No December data, no ranking
label (there is none), nothing is written except the report.

    python tools/audit_export.py --version 16 --report docs/export_audit/v16_audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID, TIME, PHASE = "MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
DEV_MONTHS = ["2025-01", "2025-07", "2025-11"]
RANK_MONTHS = ["2026-01", "2026-07"]
EPOCH = pd.Timestamp(0, tz="UTC")


def seconds(series: pd.Series) -> np.ndarray:
    return (series - EPOCH).dt.total_seconds().to_numpy()


def quantiles(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    q = np.quantile(v, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {"n": int(len(v)), "mean": float(v.mean()), "p01": float(q[0]), "p10": float(q[1]), "p50": float(q[2]), "p90": float(q[3]), "p99": float(q[4]),
            "max": float(v.max()), "share_gt_1h": float((v > 3600).mean()), "share_gt_4h": float((v > 14400).mean()), "share_gt_12h": float((v > 43200).mean())}


def context(raw: pd.DataFrame) -> pd.DataFrame:
    """Departure rows with the clocks the audit needs; derived exactly as the mixture/export tools do."""
    dep = raw[raw[PHASE].eq("DEP")].copy()
    dep["month"] = dep[TIME].dt.strftime("%Y-%m")
    dep["day"] = dep[TIME].dt.strftime("%Y-%m-%d")
    dep["takeoff"] = seconds(dep[TIME])
    dep["schedule_interval_s"] = dep["takeoff"] - seconds(dep["SCHED_TIME_UTC_mvt"])
    dep["proxy_s"] = dep["takeoff"] - seconds(dep["AOBT_3_flt"])
    dep["clock_available"] = dep["AOBT_3_flt"].notna()
    dep["no_nm"] = dep["FLIGHT_ID_mvt"].isna()
    dep["population"] = dep["ADEP_mvt"].eq("LIRF") & dep["AOBT_3_flt"].isna() & dep["LOBT_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()
    return dep


def development_reference(learner: str, residual_run: str) -> pd.DataFrame:
    """The incumbent's development predictions (v9 + residual on matched non-LIRF rows, blended 0.7 with the learner, floored)."""
    sc = pd.concat([pd.read_parquet(ROOT / "scratch" / f"{f}.parquet") for f in ["seasonal", "forward"]], ignore_index=True)
    res = pd.read_parquet(ROOT / residual_run / "development_residuals.parquet")
    res = res.rename(columns={[c for c in res.columns if c != ID][0]: "res"})
    lrn = pd.concat([pd.read_parquet(ROOT / learner / f / "predictions.parquet") for f in ["seasonal", "forward"]], ignore_index=True)
    m = pd.concat([pd.read_parquet(ROOT / "data/external/adsblol/matches_v11" / f"matches2_{k}.parquet")[[ID, "gate", "motion_start", "takeoff"]] for k in DEV_MONTHS], ignore_index=True)
    d = sc.merge(res, on=ID, how="left").merge(lrn, on=ID, how="left").merge(m, on=ID, how="left")
    non_lirf = d["airport"].ne("LIRF").to_numpy()
    matched = d["gate"].notna().to_numpy()
    v = np.where(matched & non_lirf, d["v9"] + d["res"].fillna(0), d["v9"])
    v = np.where(non_lirf & np.isfinite(d["adsb_full"]), 0.3 * v + 0.7 * d["adsb_full"], v)
    d["pred"] = np.maximum(v, d["floor"])
    d["label"] = d["TAXITIME_SEC_mvt"]
    d["month"] = d[TIME].dt.strftime("%Y-%m")
    d["day"] = d[TIME].dt.strftime("%Y-%m-%d")
    d["adsb_taxi"] = d["takeoff"] - d["motion_start"]
    return d


def daily_outliers(frame: pd.DataFrame, value: str) -> dict:
    """Airport-days whose mean prediction sits more than 3 robust sigmas from the airport-month's daily means."""
    out = {}
    daily = frame.groupby(["ADEP_mvt", "month", "day"])[value].agg(["mean", "size"]).reset_index()
    for (airport, month), g in daily.groupby(["ADEP_mvt", "month"]):
        med = g["mean"].median()
        mad = 1.4826 * (g["mean"] - med).abs().median()
        flag = g[(g["mean"] - med).abs() > max(3 * mad, 0.15 * med)]
        for r in flag.itertuples(index=False):
            out.setdefault(f"{airport} {month}", []).append({"day": r.day, "rows": int(r.size), "mean": round(float(r.mean), 1), "month_median_of_daily_means": round(float(med), 1)})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--compare", nargs="*", type=int, default=[15, 11, 9], help="earlier export versions to diff against")
    parser.add_argument("--learner", default="runs/adsb_v17/ens_v17_bag_ctr_c2")
    parser.add_argument("--residual-run", default="runs/adsb_v17_sel")
    parser.add_argument("--features", default="runs/adsb_v17_sel/export_v16/ranking_features.parquet")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    path = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    export = pd.read_parquet(path)
    report = {"file": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": int(len(export)),
              "columns": {c: str(t) for c, t in export.dtypes.items()}}
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
    dep = context(raw)
    dep = dep[dep["month"].isin(RANK_MONTHS)]
    pred_col = [c for c in export.columns if c != ID][0]
    export = export.rename(columns={pred_col: "pred"})

    # A. file integrity
    ids_export, ids_rank = set(export[ID].astype("int64")), set(dep[ID].astype("int64"))
    report["integrity"] = {"unique_ids": bool(export[ID].is_unique), "ids_match_ranking_departures": ids_export == ids_rank,
                           "missing_from_export": len(ids_rank - ids_export), "extra_in_export": len(ids_export - ids_rank),
                           "nan_predictions": int(export["pred"].isna().sum()), "min": float(export["pred"].min()), "max": float(export["pred"].max()),
                           "negative_predictions": int((export["pred"] < 0).sum()), "predictions_below_minus_12": int((export["pred"] < -12).sum())}

    d = dep.merge(export, on=ID, how="inner")
    feats = pd.read_parquet(ROOT / args.features)
    d = d.merge(feats[[ID, "adsb_matched", "adsb_gate", "adsb_motion_s", "v9"]], on=ID, how="left")
    dev = development_reference(args.learner, args.residual_run)

    # B. distribution per airport-month, ranking versus development predictions and labels
    dist = {}
    for airport in AIRPORTS:
        dist[airport] = {}
        for month in RANK_MONTHS:
            dist[airport][month] = quantiles(d.loc[d["ADEP_mvt"].eq(airport) & d["month"].eq(month), "pred"].to_numpy())
        for month in DEV_MONTHS:
            sub = dev[dev["airport"].eq(airport) & dev["month"].eq(month)]
            dist[airport][f"dev {month} pred"] = quantiles(sub["pred"].to_numpy())
            dist[airport][f"dev {month} label"] = quantiles(sub["label"].to_numpy())
    report["distribution"] = dist
    shift = {}
    for airport in AIRPORTS:
        dev_means = [dist[airport][f"dev {m} pred"]["mean"] for m in DEV_MONTHS]
        for month in RANK_MONTHS:
            mean = dist[airport][month]["mean"]
            ratio = mean / np.mean(dev_means)
            if abs(ratio - 1) > 0.15:
                shift[f"{airport} {month}"] = {"mean": round(mean, 1), "dev_means": [round(x, 1) for x in dev_means], "ratio_to_dev_mean": round(float(ratio), 3)}
    report["airport_month_mean_shift_gt_15pct"] = shift

    # C. airport-days out of line with their month
    report["daily_outliers"] = daily_outliers(d, "pred")
    watch = d[(d["ADEP_mvt"].eq("EHAM") & d["day"].between("2026-01-02", "2026-01-07")) | (d["ADEP_mvt"].eq("LFPG") & d["day"].eq("2026-01-07"))]
    report["disruption_days"] = {f"{r.ADEP_mvt} {r.day}": {"rows": int(r.n), "mean_pred": round(float(r.mean_pred), 1), "p90_pred": round(float(r.p90), 1), "no_nm_share": round(float(r.no_nm), 3),
                                                            "adsb_matched_share": round(float(r.matched), 3)}
                                 for r in watch.groupby(["ADEP_mvt", "day"]).agg(n=("pred", "size"), mean_pred=("pred", "mean"), p90=("pred", lambda s: s.quantile(0.9)), no_nm=("no_nm", "mean"), matched=("adsb_matched", "mean")).reset_index().itertuples(index=False)}
    ref = dev[dev["airport"].eq("EHAM") & dev["month"].eq("2025-01")]
    report["disruption_days"]["EHAM 2025-01 reference (dev pred / label)"] = {"mean_pred": round(float(ref["pred"].mean()), 1), "mean_label": round(float(ref["label"].mean()), 1)}

    # D. extreme predictions
    ext = d.nlargest(40, "pred")[["ADEP_mvt", "day", "pred", "v9", "schedule_interval_s", "proxy_s", "population", "clock_available", "adsb_matched"]]
    report["top_40_predictions"] = [{k: (None if pd.isna(v) else (float(v) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))) for k, v in r.items()} for r in ext.to_dict("records")]
    counts = {}
    for th in [3600, 14400, 43200, 86400]:
        counts[f"gt_{th}"] = {"ranking_rows": int((d["pred"] > th).sum()), "ranking_share": float((d["pred"] > th).mean()),
                              "dev_pred_share": float((dev["pred"] > th).mean()), "dev_label_share": float((dev["label"] > th).mean()),
                              "ranking_by_airport": d[d["pred"] > th].groupby("ADEP_mvt").size().to_dict()}
    report["extreme_counts"] = counts

    # E. negative NM proxy rows and the LIRF population
    neg = d[d["proxy_s"] < 0]
    report["negative_proxy_rows"] = {"rows": int(len(neg)), "by_airport": neg.groupby("ADEP_mvt").size().to_dict(), "pred": quantiles(neg["pred"].to_numpy()),
                                     "dev_rows": int((dev["proxy_s"] < 0).sum()), "dev_pred": quantiles(dev.loc[dev["proxy_s"] < 0, "pred"].to_numpy())}
    pop = d[d["population"]]
    long_pop = pop[pop["schedule_interval_s"] > 14400]
    report["lirf_population"] = {"rows": int(len(pop)), "by_month": pop.groupby("month").size().to_dict(), "dev_rows_per_month": dev[dev["population"]].groupby("month").size().to_dict(),
                                 "pred": quantiles(pop["pred"].to_numpy()), "dev_pred": quantiles(dev.loc[dev["population"], "pred"].to_numpy()),
                                 "interval_gt_4h": {"rows": int(len(long_pop)), "pred_over_interval_median": float((long_pop["pred"] / long_pop["schedule_interval_s"]).median()) if len(long_pop) else None,
                                                    "pred_gt_86400": int((long_pop["pred"] > 86400).sum()),
                                                    "dev_pred_over_interval_median": float((dev.loc[dev["population"] & (dev["schedule_interval_s"] > 14400), "pred"] / dev.loc[dev["population"] & (dev["schedule_interval_s"] > 14400), "schedule_interval_s"]).median())}}

    # F. ADS-B-informed rows: coverage and prediction minus the ADS-B-implied taxi time, ranking versus development
    adsb = {}
    d["adsb_delta"] = d["pred"] - d["adsb_motion_s"]
    dev["adsb_delta"] = dev["pred"] - dev["adsb_taxi"]
    for airport in AIRPORTS:
        adsb[airport] = {}
        for month in RANK_MONTHS:
            sub = d[d["ADEP_mvt"].eq(airport) & d["month"].eq(month)]
            gate = sub[sub["adsb_gate"].eq(1)]
            adsb[airport][month] = {"matched_share": float(sub["adsb_matched"].fillna(0).mean()), "gate_share": float(sub["adsb_gate"].fillna(0).mean()),
                                    "gate_pred_minus_adsb_taxi": quantiles(gate["adsb_delta"].to_numpy())}
        for month in DEV_MONTHS:
            sub = dev[dev["airport"].eq(airport) & dev["month"].eq(month)]
            gate = sub[sub["gate"].fillna(False).astype(bool)]
            adsb[airport][f"dev {month}"] = {"matched_share": float(sub["gate"].notna().mean()), "gate_share": float(sub["gate"].fillna(False).astype(bool).mean()),
                                             "gate_pred_minus_adsb_taxi": quantiles(gate["adsb_delta"].to_numpy()),
                                             "gate_label_minus_adsb_taxi": quantiles((gate["label"] - gate["adsb_taxi"]).to_numpy())}
    report["adsb_rows"] = adsb

    # G. rows without an NM record
    report["no_nm_share"] = {f"{a} {m}": round(float(v), 4) for (a, m), v in d.groupby(["ADEP_mvt", "month"])["no_nm"].mean().items()}
    report["no_nm_share_dev"] = {f"{a} {m}": round(float(v), 4) for (a, m), v in dev.assign(no_nm=~dev["clock_available"]).groupby(["airport", "month"])["no_nm"].mean().items()}

    # H. differences to earlier exports
    diffs = {}
    for version in args.compare:
        other = pd.read_parquet(ROOT / "runs/submissions" / f"vigorous-whistle_v{version}.parquet")
        other = other.rename(columns={[c for c in other.columns if c != ID][0]: "other"})
        j = d.merge(other, on=ID, how="inner")
        delta = j["pred"] - j["other"]
        diffs[f"v{version}"] = {"rows": int(len(j)), "rms_diff": float(np.sqrt((delta ** 2).mean())), "mean_diff": float(delta.mean()),
                                "share_abs_gt_600": float((delta.abs() > 600).mean()), "rows_abs_gt_1800": int((delta.abs() > 1800).sum()),
                                "rms_by_airport": {a: round(float(np.sqrt((g ** 2).mean())), 1) for a, g in delta.groupby(j["ADEP_mvt"])},
                                "largest": [{"airport": r.ADEP_mvt, "day": r.day, "pred": float(r.pred), "other": float(r.other), "interval": float(r.schedule_interval_s)}
                                            for r in j.assign(ad=delta.abs()).nlargest(10, "ad").itertuples(index=False)]}
    report["diff_to_earlier_exports"] = diffs

    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    # console summary
    print("file", report["file"], report["sha256"][:12], "rows", report["rows"], report["columns"])
    print("integrity", report["integrity"])
    print("airport-month mean shift > 15% vs dev:", json.dumps(shift, indent=None))
    print("daily outliers:", json.dumps(report["daily_outliers"], indent=None)[:1500])
    print("disruption days:", json.dumps(report["disruption_days"], indent=None))
    print("extremes:", {k: (v["ranking_rows"], round(v["ranking_share"], 5), round(v["dev_pred_share"], 5), round(v["dev_label_share"], 5)) for k, v in counts.items()})
    print("negative proxy:", report["negative_proxy_rows"]["rows"], "pred p50/max", report["negative_proxy_rows"]["pred"].get("p50"), report["negative_proxy_rows"]["pred"].get("max"))
    print("LIRF population:", {k: v for k, v in report["lirf_population"].items() if k not in ("pred", "dev_pred")})
    for v, info in diffs.items():
        print("diff", v, {k: info[k] for k in ["rms_diff", "mean_diff", "share_abs_gt_600", "rows_abs_gt_1800"]}, info["rms_by_airport"])
    print("wrote", out)


if __name__ == "__main__":
    main()
