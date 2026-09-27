"""Bounds for the LTFM IOBT decision (docs/input_shift/ltfm_iobt_sensitivity.md): for the 16,618 July-2026 THY LTFM departures with
IOBT == SCHED, which of shipped v30/v31 (IOBT as given), the full restoration (runs/final_ltfm_iobt/run) or the declared soft
restoration v31 + w * q * delta is closest to 2025's expectation?

In 2025, THY LTFM rows came in two types among IOBT in {SCHED-600, SCHED}: early (IOBT = SCHED - 600) and sched (IOBT = SCHED).
Their labels are compared at the same convention-free timing MVT - SCHED (the takeoff against the schedule; AOBT_3 at LTFM is an
estimate close to MVT - 20 min, so it adds nothing). Tables: mean label per MVT - SCHED bin and type (labels <= 3 h), from the four
recipe months (dev rows, which also give v30's development residual per type), July 2025 only, and all twelve 2025 months.

Expected label per July-2026 row under three readings:
- sched: the 2026 rows are all sched-type (the old convention carried nothing, or the practice itself ended); shipped is right;
- mixture: a share q of them is early-type as in 2025 (a pure format change): T_sched + q * (T_early - T_sched);
- early: every restored row is early-type (an upper bound, not a plausible reading).
Two anchors: 'relative' keeps shipped v30 as right for the sched type and adds only the 2025 type gap at the same timing
(E = v30 + reading's gap), so a general July-2026 LTFM level drift cancels; 'absolute' is the LFPG-style E = the table's value
(drift included; its calibration on the unedited rows is reported). Bias SSE of v30 + w * q * delta and v30 + w * delta against
each E, w in {0, 0.5, 0.75, 1}, as a share of ranking SSE; row noise is common to all blends and cancels.
Descriptive; no ranking label exists or is inferred.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import utc_now, write_json  # noqa: E402
from ltfm_iobt_convention import ltfm  # noqa: E402

ID = "MVT_ID_mvt"
BINS = [-1e9, 300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1800, 2100, 2700, 3600, 1e9]
RANKING_SSE = 344841 * 229.5832 ** 2  # v31 official RMSE on all rows
DELTA = ROOT / "runs/final_ltfm_iobt/delta.parquet"
WEIGHTS = [0.0, 0.5, 0.75, 1.0]


def labelled_history() -> pd.DataFrame:
    dep = ltfm()
    dep = dep[dep["month"] <= "2025-12"]
    labels = []
    for f in sorted((ROOT / "data/raw").glob("training_2025-*.parquet")):
        t = pq.read_table(f, columns=[ID, "PHASE_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt"]).to_pandas()
        labels.append(t.loc[t["PHASE_mvt"].eq("DEP") & t["ADEP_mvt"].eq("LTFM"), [ID, "TAXITIME_SEC_mvt"]])
    dep = dep.merge(pd.concat(labels).drop_duplicates(ID), on=ID, how="left").rename(columns={"TAXITIME_SEC_mvt": "label"})
    dev = pd.read_parquet(ROOT / "runs/error_map/dev_rows.parquet", columns=[ID, "label", "v30"]).rename(columns={"label": "dev_label"})
    dep = dep.merge(dev, on=ID, how="left")
    both = dep["dev_label"].notna()
    if not np.allclose(dep.loc[both, "label"], dep.loc[both, "dev_label"]):
        raise ValueError("Raw and development labels disagree.")
    dep = dep[dep["carrier"].eq("THY") & (dep["early"] | dep["iobt_eq_sched"]) & dep["label"].le(10800)].copy()
    dep["type"] = np.where(dep["early"], "early", "sched")
    dep["ms"] = (dep["MVT_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep["bin"] = pd.cut(dep["ms"], BINS)
    return dep


def tables(rows: pd.DataFrame) -> pd.DataFrame:
    t = rows.groupby(["bin", "type"], observed=True)["label"].agg(["mean", "size"]).unstack("type")
    t.columns = [f"{a}_{b}" for a, b in t.columns]
    t["gap"] = t["mean_early"] - t["mean_sched"]
    return t


def main() -> None:
    hist = labelled_history()
    sources = {"recipe_months": hist[hist["v30"].notna()], "july_2025": hist[hist["month"] == "2025-07"], "all_2025": hist}
    report = {"created_at_utc": utc_now(), "bins_mvt_minus_sched_s": BINS, "ranking_sse_reference": RANKING_SSE, "history": {}, "readings": {}}
    for name, rows in sources.items():
        t = tables(rows)
        entry = {"rows": {k: int(v) for k, v in rows["type"].value_counts().items()},
                 "label_mean": {k: float(v) for k, v in rows.groupby("type")["label"].mean().items()},
                 "mvt_minus_sched_median": {k: float(v) for k, v in rows.groupby("type")["ms"].median().items()},
                 "by_bin": {str(b): {k: (float(v) if pd.notna(v) else None) for k, v in r.items()} for b, r in t.iterrows()}}
        if name == "recipe_months":
            res = rows["label"] - rows["v30"]
            entry["v30_residual_mean"] = {k: float(v) for k, v in res.groupby(rows["type"]).mean().items()}
            entry["v30_residual_by_bin"] = {str(b): {k: float(v) for k, v in s.items()}
                                            for b, s in res.groupby([rows["bin"], rows["type"]], observed=True).mean().unstack("type").iterrows()}
            entry["v30_type_gap_same_bin"] = {str(b): float(v) for b, v in
                                              rows.groupby(["bin", "type"], observed=True)["v30"].mean().unstack("type").eval("early - sched").items()}
        report["history"][name] = entry
    d = pd.read_parquet(DELTA)
    raw = pq.read_table(ROOT / "data/raw/ranking.parquet", columns=[ID, "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]).to_pandas()
    d = d.merge(raw, on=ID, how="left")
    d["bin"] = pd.cut(d["mvt_minus_sched_s"], BINS)
    edited = d[d["bucket"].str.startswith("edited")].copy()
    unedited = d[d["bucket"].eq("ltfm_jul_unedited")]
    for name, rows in sources.items():
        t = tables(rows)
        gap = edited["bin"].map(t["gap"]).astype(float).fillna(0.0).to_numpy()
        ts = edited["bin"].map(t["mean_sched"]).astype(float).to_numpy()
        orig, delta, q = edited["v30_orig"].to_numpy(), edited["delta_v30"].to_numpy(), edited["q"].to_numpy()
        readings = {"sched": np.zeros(len(edited)), "mixture": q * gap, "early": gap}
        out = {"mean_type_gap_at_same_timing_on_edited_rows": float(gap.mean()), "mean_q_gap": float((q * gap).mean()),
               "corr_delta_gap": float(np.corrcoef(delta, gap)[0, 1]) if gap.std() > 0 else None,
               "absolute_anchor_mean_expected_sched_minus_shipped_on_edited": float(np.nanmean(ts - orig))}
        for anchor in ["relative", "absolute"]:
            for reading, add in readings.items():
                expected = orig + add if anchor == "relative" else ts + add
                ok = np.isfinite(expected)
                sse = {}
                for kind, step in [("soft_q", q * delta), ("full", delta)]:
                    base = float(((expected - orig)[ok] ** 2).sum())
                    sse[kind] = {f"w{w}": (float(((expected - orig - w * step)[ok] ** 2).sum()) - base) / RANKING_SSE * 100 for w in WEIGHTS}
                out[f"{anchor}:{reading}"] = {"mean_expected_minus_shipped": float((expected - orig)[ok].mean()),
                                              "bias_sse_change_vs_shipped_pct_of_ranking_sse": sse}
        report["readings"][name] = out
        print(name, {k: {kk: {w: round(x, 3) for w, x in vv.items()} for kk, vv in v["bias_sse_change_vs_shipped_pct_of_ranking_sse"].items()}
                     for k, v in out.items() if isinstance(v, dict)})
        print("  gap on edited", round(out["mean_type_gap_at_same_timing_on_edited_rows"], 2), "q*gap", round(out["mean_q_gap"], 2),
              "corr(delta, gap)", out["corr_delta_gap"], "abs sched - shipped on edited", round(out["absolute_anchor_mean_expected_sched_minus_shipped_on_edited"], 2))
    # context variant (not in the declaration): the context networks also move the unedited LTFM July rows through neighbour inputs.
    # The full restoration makes ~66% of LTFM neighbours early (2025: ~17-20%); scaling by the mean q returns it to the 2025 level.
    # No 2025 label anchors this; under the 'sched' reading it is a pure cost, under 'mixture' the model's own view says it is right.
    qbar = float(edited["q"].mean())
    du = unedited["delta_v30"].to_numpy()
    report["context_variant_unedited_rows"] = {"rows": int(len(unedited)), "mean_q": qbar, "delta": {"mean": float(du.mean()), "rms": float(np.sqrt(np.mean(du ** 2)))},
                                               "cost_if_sched_pct_of_ranking_sse": {f"w{w}": float(((w * qbar * du) ** 2).sum()) / RANKING_SSE * 100 for w in WEIGHTS},
                                               "model_implied_gain_if_mixture_pct_of_ranking_sse_w1": -float(((qbar * du) ** 2).sum()) / RANKING_SSE * 100}
    print("context variant", report["context_variant_unedited_rows"])
    write_json(ROOT / "docs/input_shift/ltfm_iobt_hypotheses.json", report)


if __name__ == "__main__":
    main()
