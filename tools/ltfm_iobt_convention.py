"""LTFM early-IOBT convention, label-free (docs/input_shift/ltfm_iobt_sensitivity.md, step 1): in every 2025 month and January 2026,
14-21% of LTFM departures have IOBT exactly 600 s before schedule; in July 2026 it is 0.6%. How consistent is that convention per
flight number (FLIGHT_mvt), callsign and operator, and how many July 2026 LTFM departures belong to flight numbers that carried it?

Reads only schedule / NM clock / identity columns of the raw files (no label, no block time). Writes --output.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import utc_now, write_json  # noqa: E402

ID = "MVT_ID_mvt"
COLS = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt",
        "IOBT_flt", "EOBT_1_flt", "LOBT_flt"]
HISTORY_FILES = sorted(p.name for p in (ROOT / "data/raw").glob("training_*.parquet")) + ["ranking.parquet"]


def ltfm() -> pd.DataFrame:
    parts = []
    for name in HISTORY_FILES:
        t = pq.read_table(ROOT / "data/raw" / name, columns=COLS).to_pandas()
        parts.append(t[t["PHASE_mvt"].eq("DEP") & t["ADEP_mvt"].eq("LTFM")])
    dep = pd.concat(parts, ignore_index=True).drop_duplicates(ID)
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    dep["sched_minus_iobt_s"] = (dep["SCHED_TIME_UTC_mvt"] - dep["IOBT_flt"]).dt.total_seconds()
    dep["early"] = dep["sched_minus_iobt_s"].eq(600)
    dep["iobt_eq_sched"] = dep["sched_minus_iobt_s"].eq(0)
    dep["eobt1_eq_iobt"] = dep["EOBT_1_flt"].eq(dep["IOBT_flt"])
    dep["lobt_eq_iobt"] = dep["LOBT_flt"].eq(dep["IOBT_flt"])
    dep["carrier"] = dep["FLIGHT_mvt"].astype("string").str.upper().str.extract(r"^([A-Z]+)", expand=False).str[:3].fillna("?")
    return dep


def row_weights(dep: pd.DataFrame, history_until: str = "2026-01", m: float = 20.0) -> tuple[pd.Series, float]:
    """The declared per-flight-number weight q (configs/ltfm_iobt_sensitivity_v1.json): over THY LTFM departures up to history_until
    with IOBT in {SCHED-600, SCHED}, q = (e + m * p0) / (e + s + m), p0 the pooled share; unseen numbers get p0."""
    h = dep[(dep["month"] <= history_until) & dep["carrier"].eq("THY") & (dep["early"] | dep["iobt_eq_sched"])]
    p0 = float(h["early"].mean())
    g = h.groupby("FLIGHT_mvt")["early"].agg(["sum", "size"])
    q = (g["sum"] + m * p0) / (g["size"] + m)
    return q, p0


def consistency(hist: pd.DataFrame, key: str, min_n: int) -> dict:
    g = hist.dropna(subset=[key]).groupby(key)["early"].agg(["size", "mean"])
    g = g[g["size"] >= min_n]
    rows = hist[hist[key].isin(g.index)]
    share = rows[key].map(g["mean"])
    bins = pd.cut(share, [-0.001, 0.0, 0.2, 0.5, 0.8, 0.999, 1.0], labels=["0", "(0,0.2]", "(0.2,0.5]", "(0.5,0.8]", "(0.8,1)", "1"])
    return {"keys": int(len(g)), "rows": int(len(rows)),
            "rows_by_key_share_bin": {str(k): int(v) for k, v in bins.value_counts(sort=False).items()},
            "early_rows_by_key_share_bin": {str(k): int(v) for k, v in bins[rows["early"].to_numpy()].value_counts(sort=False).items()},
            "keys_by_share_bin": {str(k): int(v) for k, v in pd.cut(g["mean"], [-0.001, 0.0, 0.2, 0.5, 0.8, 0.999, 1.0],
                                                                  labels=["0", "(0,0.2]", "(0.2,0.5]", "(0.5,0.8]", "(0.8,1)", "1"]).value_counts(sort=False).items()},
            "purity_early_rows_in_keys_ge_0.8": float(rows.loc[share >= 0.8, "early"].mean()) if (share >= 0.8).any() else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("docs/input_shift/ltfm_iobt_convention.json"))
    args = parser.parse_args()
    dep = ltfm()
    hist = dep[dep["month"] <= "2026-01"]
    jul = dep[dep["month"] == "2026-07"].copy()
    by_month = dep.groupby("month").agg(rows=("early", "size"), early=("early", "mean"), iobt_eq_sched=("iobt_eq_sched", "mean"),
                                        eobt1_eq_iobt=("eobt1_eq_iobt", "mean"), lobt_eq_iobt=("lobt_eq_iobt", "mean"))
    carriers = hist.groupby("carrier")["early"].agg(["size", "mean"]).sort_values("size", ascending=False).head(12)
    jul_carriers = jul.groupby("carrier")["early"].agg(["size", "mean"])
    ops = hist.groupby("AIRCRAFT_OPERATOR_flt")["early"].agg(["size", "mean"]).sort_values("size", ascending=False).head(12)
    # per flight number: history share; on July 2026 rows, which flight numbers carried it and what their IOBT looks like now
    fn = hist.dropna(subset=["FLIGHT_mvt"]).groupby("FLIGHT_mvt")["early"].agg(["size", "mean"])
    jul["hist_n"] = jul["FLIGHT_mvt"].map(fn["size"]).fillna(0).astype(int)
    jul["hist_share"] = jul["FLIGHT_mvt"].map(fn["mean"])
    # 2025 companions of the early convention: for flight numbers with share >= 0.8, what did their non-early rows look like
    strong = fn[(fn["size"] >= 5) & (fn["mean"] >= 0.8)].index
    strong_hist = hist[hist["FLIGHT_mvt"].isin(strong)]
    thresholds = {}
    for thr in [0.5, 0.8]:
        for min_n in [3, 5, 10]:
            sel = (jul["hist_n"] >= min_n) & (jul["hist_share"] >= thr)
            thresholds[f"share>={thr},n>={min_n}"] = {
                "jul_rows": int(sel.sum()), "jul_rows_iobt_eq_sched": int((sel & jul["iobt_eq_sched"]).sum()),
                "jul_rows_already_early": int((sel & jul["early"]).sum()),
                "jul_rows_iobt_eq_sched_and_eobt1_eq_iobt": int((sel & jul["iobt_eq_sched"] & jul["eobt1_eq_iobt"]).sum()),
                "jul_rows_iobt_eq_sched_and_lobt_eq_iobt": int((sel & jul["iobt_eq_sched"] & jul["lobt_eq_iobt"]).sum()),
                "flight_numbers": int(jul.loc[sel, "FLIGHT_mvt"].nunique()),
                "hist_rows_of_these_numbers": int(hist["FLIGHT_mvt"].isin(jul.loc[sel, "FLIGHT_mvt"].unique()).sum()),
                "hist_early_share_of_these_numbers": float(hist.loc[hist["FLIGHT_mvt"].isin(jul.loc[sel, "FLIGHT_mvt"].unique()), "early"].mean()) if sel.any() else None}
    # how stable is the per-number convention over time: 2025 H1 share vs 2025 H2 + Jan 2026 share, per number with >= 5 rows each
    h1 = hist[hist["month"] <= "2025-06"].groupby("FLIGHT_mvt")["early"].agg(["size", "mean"])
    h2 = hist[hist["month"] > "2025-06"].groupby("FLIGHT_mvt")["early"].agg(["size", "mean"])
    both = h1.join(h2, lsuffix="_h1", rsuffix="_h2", how="inner")
    both = both[(both["size_h1"] >= 5) & (both["size_h2"] >= 5)]
    stability = {"numbers": int(len(both)), "corr_share": float(both["mean_h1"].corr(both["mean_h2"])),
                 "h1_ge_0.8": int((both["mean_h1"] >= 0.8).sum()), "of_which_h2_ge_0.8": int(((both["mean_h1"] >= 0.8) & (both["mean_h2"] >= 0.8)).sum()),
                 "of_which_h2_lt_0.2": int(((both["mean_h1"] >= 0.8) & (both["mean_h2"] < 0.2)).sum())}
    jan26 = hist[hist["month"] == "2026-01"]
    fn25 = hist[hist["month"] <= "2025-12"].groupby("FLIGHT_mvt")["early"].agg(["size", "mean"])
    j = jan26.assign(s25=jan26["FLIGHT_mvt"].map(fn25["mean"]), n25=jan26["FLIGHT_mvt"].map(fn25["size"]).fillna(0))
    j = j[j["n25"] >= 5]
    jan_check = {f"2025_share>={thr}": {"jan26_rows": int((j["s25"] >= thr).sum()), "jan26_early_share": float(j.loc[j["s25"] >= thr, "early"].mean())}
                 for thr in [0.5, 0.8]}
    jan_check["2025_share<0.2"] = {"jan26_rows": int((j["s25"] < 0.2).sum()), "jan26_early_share": float(j.loc[j["s25"] < 0.2, "early"].mean())}
    # the declared weight q, fitted on 2025 only and checked on January 2026 THY rows with IOBT in {SCHED-600, SCHED}
    q25, p25 = row_weights(dep, "2025-12")
    jt = hist[(hist["month"] == "2026-01") & hist["carrier"].eq("THY") & (hist["early"] | hist["iobt_eq_sched"])].copy()
    jt["q"] = jt["FLIGHT_mvt"].map(q25).fillna(p25).astype(float)
    qbins = pd.cut(jt["q"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
    y = jt["early"].astype(float)
    q_check = {"p0_2025": p25, "rows": int(len(jt)), "brier_q": float(((jt["q"] - y) ** 2).mean()), "brier_p0": float(((p25 - y) ** 2).mean()),
               "calibration": {str(k): {"rows": int(len(g)), "mean_q": float(g["q"].mean()), "early_share": float(g["early"].mean())}
                               for k, g in jt.groupby(qbins, observed=True)}}
    q, p0 = row_weights(dep)
    jul_thy = jul[jul["carrier"].eq("THY") & jul["iobt_eq_sched"]]
    qj = jul_thy["FLIGHT_mvt"].map(q).fillna(p0).astype(float)
    q_july = {"p0": p0, "rows": int(len(jul_thy)), "mean_q": float(qj.mean()), "expected_old_convention_rows": float(qj.sum()),
              "unseen_numbers_rows": int(jul_thy["FLIGHT_mvt"].map(q).isna().sum()),
              "quantiles": {str(p): float(qj.quantile(p)) for p in [0.1, 0.25, 0.5, 0.75, 0.9]}}
    report = {"created_at_utc": utc_now(), "label_free": True, "columns_read": COLS, "history": "2025-01..2026-01 LTFM departures",
              "by_month": {m: {k: (float(v) if k != "rows" else int(v)) for k, v in r.items()} for m, r in by_month.iterrows()},
              "history_carriers_top12": {k: {"rows": int(r["size"]), "early_share": float(r["mean"]),
                                             "jul26_rows": int(jul_carriers["size"].get(k, 0)), "jul26_early_share": float(jul_carriers["mean"].get(k, np.nan))}
                                         for k, r in carriers.iterrows()},
              "history_operators_top12": {k: {"rows": int(r["size"]), "early_share": float(r["mean"])} for k, r in ops.iterrows()},
              "flight_number_consistency_min_n5": consistency(hist, "FLIGHT_mvt", 5),
              "callsign_consistency_min_n5": consistency(hist, "CALLSIGN_flt", 5),
              "strong_numbers_other_rows_sched_minus_iobt": {str(k): int(v) for k, v in strong_hist.loc[~strong_hist["early"], "sched_minus_iobt_s"].clip(-3600, 3600).round(-1).value_counts().head(10).items()},
              "stability_h1_vs_h2": stability, "january_2026_from_2025_shares": jan_check,
              "declared_weight_q_check_on_january_2026": q_check, "declared_weight_q_on_july_2026": q_july,
              "july_2026": {"rows": int(len(jul)), "early": int(jul["early"].sum()), "iobt_eq_sched": int(jul["iobt_eq_sched"].sum()),
                            "eobt1_eq_iobt": int(jul["eobt1_eq_iobt"].sum()), "lobt_eq_iobt": int(jul["lobt_eq_iobt"].sum()),
                            "flight_numbers_seen_before": int((jul["hist_n"] > 0).sum()), "by_threshold": thresholds}}
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    import json
    write_json(out, json.loads(json.dumps(report, default=float).replace("NaN", "null")))
    print(json.dumps({k: report[k] for k in ["stability_h1_vs_h2", "january_2026_from_2025_shares", "declared_weight_q_check_on_january_2026",
                                             "declared_weight_q_on_july_2026"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
