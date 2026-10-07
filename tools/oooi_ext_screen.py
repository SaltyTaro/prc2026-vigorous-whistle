"""OOOI extension screen, step 4 (local, light): guarded w50 pin of each qualifying data-link format against the v69
development proxy (July / November / December 2025, offsets month-blocked), and label-free 2026 ranking coverage.

Inputs : runs/oooi_ext/formats.parquet (tools/oooi_ext_formats.py: one row per (format, departure) with the parsed OUT
         instant t1 and the label-free guard flag), runs/oooi_ext/v70_pinned_ids.parquet (oooi_pin_v1 scope),
         runs/stack_weight/dev/w65_bag10/proxy.parquet (v69 development proxy), runs/hunt/census_rank.parquet,
         data/raw/ranking.parquet (LIRF population definition).
Output : runs/oooi_ext/screen.json
Rules  : rows already in oooi_pin_v1's scope (v70's 783 LFPG ranking rows and their development counterparts), LIRF
         population rows and MVT 213142471 / 213142469 are excluded; August-October 2025 are counted, never scored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "oooi_ext"
PROXY = ROOT / "runs" / "stack_weight" / "dev" / "w65_bag10" / "proxy.parquet"
CENSUS = ROOT / "runs" / "hunt" / "census_rank.parquet"
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
GUARD_IDS = {213142471.0, 213142469.0}
MONTHS = ["2025-07", "2025-11", "2025-12"]
FRESH = ["2025-08", "2025-09", "2025-10"]
MONTH_SSE = {"2025-07": 190713 * 250.835826 ** 2, "2025-11": 162332 * 181.883484 ** 2,
             "2025-12": 165677 * 163.353697 ** 2}
W = 0.5
FLOOR = -12.0


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pct_rmse(m: str, gain: float) -> float:
    s = MONTH_SSE[m]
    return 100.0 * (np.sqrt(max(s - gain, 0.0) / s) - 1.0)


def main() -> None:
    f = pd.read_parquet(RUN / "scan" / "formats.parquet").rename(columns={"label": "acars_label"})
    # sub-format read off the token scan (disclosed): Virgin Atlantic's QP, the best-matching airline within QP_ORIG
    f = pd.concat([f, f[f["format"].eq("QP_ORIG") & f["al2"].eq("VS")].assign(format="QP_VS")], ignore_index=True)
    # EGLL family: one pin per departure from the four EGLL formats that pass the quality bar or nearly do, in the
    # order of their centred same-minute share (UA_OUT13, BA_MVT16, MVA_AD, QP_VS); each keeps its own offset
    fam_order = ["UA_OUT13", "BA_MVT16", "MVA_AD", "QP_VS"]
    fam = f[f["format"].isin(fam_order) & f["guard"]].copy()
    fam["prio"] = fam["format"].map({k: i for i, k in enumerate(fam_order)})
    fam = fam.sort_values(["MVT_ID_mvt", "prio"]).drop_duplicates("MVT_ID_mvt")
    fam["sub"] = fam["format"]
    f["sub"] = f["format"]
    f = pd.concat([f, fam.drop(columns=["prio"]).assign(format="EGLL_family")], ignore_index=True)
    pinned = pd.read_parquet(RUN / "v70_pinned_ids.parquet")
    excl = set(pinned["MVT_ID_mvt"]) | GUARD_IDS
    proxy = pd.read_parquet(PROXY, columns=["MVT_ID_mvt", "month", "label", "cls", "day", "value"])
    rk = pd.read_parquet(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    pop_ids = set(rk.loc[rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna(), "MVT_ID_mvt"])
    cen = pd.read_parquet(CENSUS, columns=["MVT_ID_mvt", "k", "own_n"])
    cen["cls"] = np.where(cen["k"].eq("unm"), np.where(cen["own_n"].fillna(0) >= 1, "unm_seen", "unm_unseen"), cen["k"])
    res = {"inputs": {"formats": sha(RUN / "scan" / "formats.parquet"), "proxy": sha(PROXY), "census": sha(CENSUS)},
           "weight": W, "formats": {}}
    f = f[~f["MVT_ID_mvt"].isin(excl)]
    f["raw"] = (f["tko"] - f["t1"]).dt.total_seconds()
    for name, g in f.groupby("format"):
        out = {"rows_by_src_month": {f"{a}|{b}": int(v) for (a, b), v in g.groupby(["src", "dep_month"]).size().items()},
               "guarded_rows_by_src_month": {f"{a}|{b}": int(v) for (a, b), v in
                                             g[g["guard"]].groupby(["src", "dep_month"]).size().items()},
               "airports": {k: int(v) for k, v in g["ADEP_mvt"].value_counts().items()}}
        # development screen
        dv = g[g["src"].eq("train") & g["dep_month"].isin(MONTHS) & g["guard"]].merge(
            proxy, on="MVT_ID_mvt", how="inner")
        dv = dv[~dv["cls"].eq("LIRF_pop")]
        dv["resid"] = dv["label"] - dv["raw"]
        per = {}
        for m in MONTHS:
            x = dv[dv["month"].eq(m)].copy()
            fit = dv[dv["month"].ne(m)]
            if len(x) == 0 or len(fit) == 0:
                per[m] = {"rows": int(len(x))}
                continue
            offs = fit.groupby("sub")["resid"].mean()
            off = {k: float(v) for k, v in offs.items()}
            x = x[x["sub"].isin(offs.index)].copy()
            x["pin"] = x["raw"] + x["sub"].map(offs)
            x["cand"] = np.maximum(W * x["pin"] + (1 - W) * x["value"], FLOOR)
            e0, e1 = x["value"] - x["label"], x["cand"] - x["label"]
            gain = float((e0 ** 2).sum() - (e1 ** 2).sum())
            orc = float((e0 ** 2).sum() - (np.maximum(e0.abs() - 300, 0) ** 2).sum())
            rng = np.random.default_rng(20261007)
            dd = x.assign(g=(e0 ** 2) - (e1 ** 2)).groupby("day")["g"].sum().to_numpy()
            bs = dd[rng.integers(0, len(dd), (2000, len(dd)))].sum(1) if len(dd) else np.array([0.0])
            per[m] = {"rows": int(len(x)), "offset_s": off, "v69_rmse": float(np.sqrt((e0 ** 2).mean())),
                      "pin_rmse": float(np.sqrt(((x["pin"] - x["label"]) ** 2).mean())),
                      "w50_rmse": float(np.sqrt((e1 ** 2).mean())), "gain_s2": gain,
                      "gain_pct_month_sse": 100 * gain / MONTH_SSE[m], "month_rmse_change_pct": pct_rmse(m, gain),
                      "gain_day_bootstrap_95": [float(v) for v in np.percentile(bs, [2.5, 97.5])],
                      "oracle300_month_rmse_change_pct": pct_rmse(m, orc),
                      "pin_abs_err_gt_1800": int(((x["pin"] - x["label"]).abs() > 1800).sum())}
        out["dev"] = per
        ok = [per[m].get("month_rmse_change_pct", 0.0) for m in MONTHS]
        out["bar_two_months_le_-0.02"] = int(sum(v <= -0.02 for v in ok))
        out["bar_none_worse"] = bool(all(v <= 0 for v in ok))
        # ranking coverage (label-free)
        rr = g[g["src"].eq("rank") & g["guard"] & ~g["MVT_ID_mvt"].isin(pop_ids)].merge(cen[["MVT_ID_mvt", "cls"]], on="MVT_ID_mvt", how="left")
        out["rank_rows"] = int(len(rr))
        out["rank_by_month_airport_class"] = {f"{a}|{b}|{c}": int(v) for (a, b, c), v in
                                              rr.groupby(["dep_month", "ADEP_mvt", "cls"]).size().items()}
        out["fresh_counts_not_scored"] = {m: int(((g["dep_month"] == m) & g["guard"] & g["src"].eq("train")).sum()) for m in FRESH}
        out["bar_rank_rows_ge_200"] = bool(len(rr) >= 200)
        res["formats"][name] = out
        print(name, {m: (per[m].get("rows"), round(per[m].get("month_rmse_change_pct", 0.0), 4)) for m in MONTHS},
              "rank", len(rr), "fresh", out["fresh_counts_not_scored"], flush=True)
    (RUN / "screen.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
