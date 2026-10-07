"""OOOI pin screen, step 3 (local, light): how the data-link OUT times relate to the recorded off-block, the pin rule
against the v69 development proxy on matched rows, and the label-free 2026 ranking coverage.

Inputs : runs/oooi_pin/joined_dev.parquet, runs/oooi_pin/joined_rank.parquet (tools/oooi_pin_join.py)
         runs/stack_weight/dev/w65_bag10/proxy.parquet (v69 development proxy)
         runs/hunt/census_rank.parquet (ranking classes, label-free)
Output : runs/oooi_pin/eval.json (copied into docs/oooi_pin/results.json by hand together with the receipts)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "oooi_pin"
PROXY = ROOT / "runs" / "stack_weight" / "dev" / "w65_bag10" / "proxy.parquet"
CENSUS = ROOT / "runs" / "hunt" / "census_rank.parquet"
MONTHS = ["2025-07", "2025-11", "2025-12"]
MONTH_SSE = {"2025-07": 190713 * 250.835826 ** 2, "2025-11": 162332 * 181.883484 ** 2,
             "2025-12": 165677 * 163.353697 ** 2}
MONTH_N = {"2025-07": 190713, "2025-11": 162332, "2025-12": 165677}
GUARD = {213142471, 213142469}
CAP = 300.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def q(x: pd.Series) -> dict:
    x = x.dropna().to_numpy(dtype=float)
    if len(x) == 0:
        return {"n": 0}
    pct = np.percentile(x, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    return {"n": int(len(x)), "pct_1_5_10_25_50_75_90_95_99": [round(float(v), 1) for v in pct],
            "mean": round(float(x.mean()), 2),
            "share_in_0_60": round(float(((x >= 0) & (x < 60)).mean()), 4),
            "share_abs_le_10": round(float((np.abs(x) <= 10).mean()), 4),
            "share_abs_le_60": round(float((np.abs(x) <= 60).mean()), 4),
            "share_abs_le_120": round(float((np.abs(x) <= 120).mean()), 4),
            "share_abs_le_300": round(float((np.abs(x) <= 300).mean()), 4)}


def source_guard(x: pd.DataFrame) -> pd.Series:
    """Label-free, per source: 2N needs its OFF time to confirm the match (|take-off - t2| <= 120 s) and a raw
    estimate >= 300 s; 8D needs a raw estimate in 300..3600 s and, where the NM clock exists, |AOBT_3 - t1| <= 1800 s."""
    raw = (x["MVT_TIME_UTC_mvt"] - x["t1"]).dt.total_seconds()
    d2 = (x["MVT_TIME_UTC_mvt"] - x["t2"]).dt.total_seconds().abs()
    aob = (x["AOBT_3_flt"] - x["t1"]).dt.total_seconds().abs()
    g2n = x["source"].eq("2N") & d2.le(120) & raw.ge(300)
    g8d = x["source"].eq("8D") & raw.between(300, 3600) & (aob.le(1800) | aob.isna())
    return g2n | g8d


def rmse_pct(month: str, gain: float) -> float:
    """Change of complete-month RMSE (percent) when the month SSE falls by `gain`."""
    s = MONTH_SSE[month]
    return 100.0 * (np.sqrt(max(s - gain, 0.0) / s) - 1.0)


def main() -> None:
    jd = pd.read_parquet(RUN / "joined_dev.parquet")
    jr = pd.read_parquet(RUN / "joined_rank.parquet")
    res: dict = {"inputs": {"proxy": {"path": str(PROXY.relative_to(ROOT)), "sha256": sha(PROXY)},
                            "census_rank": {"path": str(CENSUS.relative_to(ROOT)), "sha256": sha(CENSUS)}}}

    # ---------------- 1. how the times relate (all matched development rows) ----------------
    jd = jd[jd["source"].isin(["2N", "8D"])].rename(columns={"label": "acars_label", "month": "msg_month"}).copy()  # 2 QP rows left out
    jd["tko"] = jd["MVT_TIME_UTC_mvt"]
    jd["blk"] = jd["BLOCK_TIME_UTC_mvt"]
    jd["d_blk_t1"] = (jd["blk"] - jd["t1"]).dt.total_seconds()
    jd["d_tko_t2"] = (jd["tko"] - jd["t2"]).dt.total_seconds()
    jd["d_aobt_t1"] = (jd["AOBT_3_flt"] - jd["t1"]).dt.total_seconds()
    jd["raw_pin"] = (jd["tko"] - jd["t1"]).dt.total_seconds()
    jd["early"] = (jd["TAXITIME_SEC_mvt"] - (jd["tko"] - jd["AOBT_3_flt"]).dt.total_seconds()) > 300
    rel = {}
    for s, g in jd.groupby("source"):
        r = {"rows": int(len(g)), "recorded_offblock_minus_t1": q(g["d_blk_t1"]),
             "takeoff_minus_t2": q(g["d_tko_t2"]), "AOBT3_minus_t1": q(g["d_aobt_t1"]),
             "label_minus_takeoff_minus_t1": q(g["TAXITIME_SEC_mvt"] - g["raw_pin"]),
             "early_rows": int(g["early"].sum()),
             "early_recorded_offblock_minus_t1": q(g.loc[g["early"], "d_blk_t1"]),
             "early_AOBT3_minus_t1": q(g.loc[g["early"], "d_aobt_t1"])}
        r["by_month_recorded_offblock_minus_t1"] = {m: q(gg["d_blk_t1"]) for m, gg in g.groupby("month_dep")}
        rel[s] = r
    rel["both"] = {"rows": int(len(jd)), "recorded_offblock_minus_t1": q(jd["d_blk_t1"]),
                   "early_rows": int(jd["early"].sum()),
                   "early_recorded_offblock_minus_t1": q(jd.loc[jd["early"], "d_blk_t1"])}
    res["time_relations"] = rel

    # ---------------- 2. v69 proxy versus the pin rule on matched rows ----------------
    p = pd.read_parquet(PROXY)
    x = jd.merge(p[["MVT_ID_mvt", "month", "airport", "label", "cls", "day", "value"]], on="MVT_ID_mvt", how="inner")
    assert (x["label"] - x["TAXITIME_SEC_mvt"]).abs().max() < 1e-6
    x = x[x["month"].isin(MONTHS)].copy()
    x["resid_raw"] = x["label"] - x["raw_pin"]
    # label-free guards (screen variants): 'none'; 'plaus' = pin only when the raw estimate take-off - t1 is a
    # plausible taxi-out (300..3600 s); 'plaus_agree' = plaus and |raw - v69| <= 900 s. Unguarded rows keep v69.
    guards = {"none": pd.Series(True, index=x.index),
              "plaus": x["raw_pin"].between(300, 3600),
              "plaus_agree": x["raw_pin"].between(300, 3600) & (x["raw_pin"] - x["value"]).abs().le(900),
              "source": source_guard(x)}
    out = {}
    for variant in [f"{o}|{gname}" for gname in guards for o in ("mean", "median")]:
        oname, gname = variant.split("|")
        gmask = guards[gname]
        per = {}
        for m in MONTHS:
            g = x[x["month"] == m].copy()
            others = x[(x["month"] != m) & gmask]
            off = others.groupby("source")["resid_raw"].agg(oname)
            g["pin"] = np.where(gmask[g.index], g["raw_pin"] + g["source"].map(off), g["value"])
            g["blend"] = 0.5 * g["pin"] + 0.5 * g["value"]
            e69 = g["value"] - g["label"]
            row = {"rows": int(len(g)), "rows_pinned": int(gmask[g.index].sum()), "rows_by_source": {k: int(v) for k, v in g["source"].value_counts().items()},
                   "offset_by_source": {k: round(float(v), 2) for k, v in off.items()},
                   "v69_rmse": float(np.sqrt((e69 ** 2).mean()))}
            sse69 = float((e69 ** 2).sum())
            for name in ("pin", "blend"):
                e = g[name] - g["label"]
                sse = float((e ** 2).sum())
                gain = sse69 - sse
                row[f"{name}_rmse"] = float(np.sqrt((e ** 2).mean()))
                row[f"{name}_gain_s2"] = gain
                row[f"{name}_gain_pct_month_sse"] = 100.0 * gain / MONTH_SSE[m]
                row[f"{name}_month_rmse_change_pct"] = rmse_pct(m, gain)
                row[f"{name}_rows_better"] = int((e.abs() < e69.abs()).sum())
            # cap-limited oracle: ideal correction within +-300 s
            ora = np.maximum(e69.abs() - CAP, 0.0)
            ogain = sse69 - float((ora ** 2).sum())
            row["oracle300_gain_s2"] = ogain
            row["oracle300_gain_pct_month_sse"] = 100.0 * ogain / MONTH_SSE[m]
            row["oracle300_month_rmse_change_pct"] = rmse_pct(m, ogain)
            row["v69_sse"] = sse69
            row["by_source"] = {}
            for s, gs in g.groupby("source"):
                es = gs["value"] - gs["label"]
                ep = gs["pin"] - gs["label"]
                eb = gs["blend"] - gs["label"]
                row["by_source"][s] = {"rows": int(len(gs)), "v69_rmse": float(np.sqrt((es ** 2).mean())),
                                       "pin_rmse": float(np.sqrt((ep ** 2).mean())),
                                       "blend_rmse": float(np.sqrt((eb ** 2).mean())),
                                       "pin_gain_s2": float((es ** 2).sum() - (ep ** 2).sum())}
            row["by_class"] = {}
            for c, gc in g.groupby("cls"):
                ec = gc["value"] - gc["label"]
                ep = gc["pin"] - gc["label"]
                row["by_class"][c] = {"rows": int(len(gc)), "v69_rmse": float(np.sqrt((ec ** 2).mean())),
                                      "pin_rmse": float(np.sqrt((ep ** 2).mean()))}
            early = g["early"]
            if early.any():
                ee = g.loc[early, "value"] - g.loc[early, "label"]
                pe = g.loc[early, "pin"] - g.loc[early, "label"]
                row["early_rows"] = {"rows": int(early.sum()), "v69_rmse": float(np.sqrt((ee ** 2).mean())),
                                     "pin_rmse": float(np.sqrt((pe ** 2).mean()))}
            # paired day-block bootstrap of the gain (2,000 draws, fixed seed)
            rng = np.random.default_rng(20261007)
            gd = g.assign(s69=e69 ** 2, sp=(g["pin"] - g["label"]) ** 2, sb=(g["blend"] - g["label"]) ** 2)
            daysum = gd.groupby("day")[["s69", "sp", "sb"]].sum()
            gp = (daysum["s69"] - daysum["sp"]).to_numpy()
            gb = (daysum["s69"] - daysum["sb"]).to_numpy()
            idx = rng.integers(0, len(gp), size=(2000, len(gp)))
            row["pin_gain_day_bootstrap_95"] = [float(v) for v in np.percentile(gp[idx].sum(1), [2.5, 97.5])]
            row["blend_gain_day_bootstrap_95"] = [float(v) for v in np.percentile(gb[idx].sum(1), [2.5, 97.5])]
            row["days"] = int(len(gp))
            row["days_pin_better"] = int((gp > 0).sum())
            row["days_pin_worse"] = int((gp < 0).sum())
            per[m] = row
        out[variant] = per
    res["pin_vs_v69"] = out

    # ---------------- 3. ranking coverage (label-free) ----------------
    c = pd.read_parquet(CENSUS, columns=["MVT_ID_mvt", "ADEP_mvt", "k", "own_n", "month"])
    c["cls"] = np.where(c["k"].eq("unm"), np.where(c["own_n"].fillna(0) >= 1, "unm_seen", "unm_unseen"), c["k"])
    r = jr.rename(columns={"label": "acars_label", "month": "msg_month"}).merge(c[["MVT_ID_mvt", "cls"]], on="MVT_ID_mvt", how="left")
    r["cls"] = r["cls"].fillna("not_in_census")
    cov = {}
    for (m, a), g in r.groupby(["month_dep", "ADEP_mvt"]):
        cov[f"{m}|{a}"] = {"rows": int(len(g)),
                           "by_source": {k: int(v) for k, v in g["source"].value_counts().items()},
                           "by_class": {k: int(v) for k, v in g["cls"].value_counts().items()}}
    r["raw_pin"] = (r["MVT_TIME_UTC_mvt"] - r["t1"]).dt.total_seconds()
    r["plaus"] = r["raw_pin"].between(300, 3600)
    r["sguard"] = source_guard(r)
    for (m, a), g in r.groupby(["month_dep", "ADEP_mvt"]):
        gg = g[g["plaus"] & g["source"].isin(["2N", "8D"])]
        cov[f"{m}|{a}"]["rows_2N_8D_plausible"] = int(len(gg))
        cov[f"{m}|{a}"]["by_class_2N_8D_plausible"] = {k: int(v) for k, v in gg["cls"].value_counts().items()}
        gs = g[g["sguard"]]
        cov[f"{m}|{a}"]["rows_source_guard"] = int(len(gs))
        cov[f"{m}|{a}"]["by_source_source_guard"] = {k: int(v) for k, v in gs["source"].value_counts().items()}
        cov[f"{m}|{a}"]["by_class_source_guard"] = {k: int(v) for k, v in gs["cls"].value_counts().items()}
    # label-free drift check 2025 versus 2026: the same quantities that need no label
    drift = {}
    for name, d in (("dev_2025", jd.assign(month_dep=jd["month_dep"])), ("rank_2026", r)):
        for (m, s_), g in d[d["source"].isin(["2N", "8D"])].groupby(["month_dep", "source"]):
            drift[f"{name}|{m}|{s_}"] = {
                "rows": int(len(g)),
                "takeoff_minus_t2": q((g["MVT_TIME_UTC_mvt"] - g["t2"]).dt.total_seconds()),
                "AOBT3_minus_t1": q((g["AOBT_3_flt"] - g["t1"]).dt.total_seconds()),
                "takeoff_minus_t1": q((g["MVT_TIME_UTC_mvt"] - g["t1"]).dt.total_seconds())}
    res["label_free_drift"] = drift
    res["ranking_coverage"] = cov
    res["ranking_guard_rows_hit"] = int(r["MVT_ID_mvt"].isin(GUARD).sum())
    all_rank_lfpg = c[c["ADEP_mvt"].eq("LFPG")].groupby(["month", "cls"]).size()
    res["ranking_lfpg_rows_by_class"] = {f"{k[0]}|{k[1]}": int(v) for k, v in all_rank_lfpg.items()}
    (RUN / "eval.json").write_text(json.dumps(res, indent=1, default=float))

    # console summary
    for s in ("2N", "8D"):
        d = rel[s]["recorded_offblock_minus_t1"]
        print(s, rel[s]["rows"], "blk-t1 pct", d["pct_1_5_10_25_50_75_90_95_99"], "|<=60|", d["share_abs_le_60"],
              "|<=120|", d["share_abs_le_120"], "|<=300|", d["share_abs_le_300"], "early", rel[s]["early_rows"])
    for variant in out:
        for m in MONTHS:
            row = out[variant][m]
            print(variant, m, row["rows"], row["rows_pinned"], "v69 %.1f pin %.1f blend %.1f | gain pin %.3e (%.4f%% SSE, RMSE %.4f%%) "
                  "blend %.3e (%.4f%%, RMSE %.4f%%) | oracle %.3e (RMSE %.4f%%)" % (
                      row["v69_rmse"], row["pin_rmse"], row["blend_rmse"], row["pin_gain_s2"],
                      row["pin_gain_pct_month_sse"], row["pin_month_rmse_change_pct"], row["blend_gain_s2"],
                      row["blend_gain_pct_month_sse"], row["blend_month_rmse_change_pct"], row["oracle300_gain_s2"],
                      row["oracle300_month_rmse_change_pct"]))
    print(json.dumps(cov, indent=1))
    print("guard rows hit", res["ranking_guard_rows_hit"])


if __name__ == "__main__":
    main()
