"""Readings of the time-gap test (configs/adsb_ctx_gap_v1.json): per fold (gap = fit Jan-Jun 2025, near = fit May-Oct 2025; both
score November + December 2025) the 4-seed mean prediction of each variant (g16, g24, g32 epochs), its RMSE on non-LIRF rows (all,
by month, ADS-B matched / unmatched), the epoch effects (g32 - g16, g24 - g16) and their difference between the folds, with a paired
day bootstrap (61 days) of the difference-in-differences. Descriptive; writes docs/adsb_ctx_gap/report.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402

VARIANTS = ["g16", "g24", "g32"]
FOLDS = ["gap", "near"]


def truth() -> pd.DataFrame:
    labels = pd.concat([pd.read_parquet(ROOT / "data/phase1/58d621b245de3f5e12dd/labels/training_2025-11-01_2025-12-01.parquet"),
                        pd.read_parquet(ROOT / "data/december/d198dea10625b0a75977/labels/training_2025-12-01_2026-01-01.parquet")])
    raw = pd.concat([pd.read_parquet(ROOT / f"data/raw/{f}.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", TIME])
                     for f in ["training_2025-11-01_2025-12-01", "training_2025-12-01_2026-01-01"]])
    raw = raw[raw["PHASE_mvt"].eq("DEP")]
    adsb = pd.concat([pd.read_parquet(ROOT / f"runs/adsb_v24/features/matches_{m}.parquet", columns=[ID, "adsb_matched"]) for m in ["2025-11", "2025-12"]])
    out = labels.merge(raw, on=ID).merge(adsb, on=ID, how="left")
    out["month"] = out[TIME].dt.strftime("%Y-%m")
    out["day"] = out[TIME].dt.strftime("%Y-%m-%d")
    return out.drop_duplicates(ID).reset_index(drop=True)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx_gap"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx_gap/report.json"))
    parser.add_argument("--replicates", type=int, default=4000)
    args = parser.parse_args()
    config = read_json(ROOT / "configs/adsb_ctx_gap_v1.json")
    seeds = config["seeds"]
    t = truth()
    t = t[t["ADEP_mvt"].ne("LIRF")].reset_index(drop=True)
    y = t[TARGET].to_numpy(dtype=float)
    report = {"created_at_utc": utc_now(), "rows_non_lirf": int(len(t)), "seeds": seeds, "folds": {}, "seed_rmse": {}}
    means = {}
    for fold in FOLDS:
        report["folds"][fold] = {}
        for v in VARIANTS:
            preds = []
            for s in seeds:
                run = ROOT / args.runs / f"{v}_s{s}" / fold
                if not (run / "run.json").exists():
                    raise ValueError(f"Missing run {run}.")
                p = pd.read_parquet(run / "predictions.parquet").set_index(ID)["adsb_full"].reindex(t[ID]).to_numpy(dtype=float)
                if np.isnan(p).any():
                    raise ValueError(f"{run}: predictions do not cover every scored row.")
                preds.append(p)
                report["seed_rmse"][f"{v}_s{s}_{fold}"] = rmse(y, p)
            m = np.mean(preds, axis=0)
            means[(fold, v)] = m
            matched = t["adsb_matched"].eq(1).to_numpy()
            report["folds"][fold][v] = {"rmse": rmse(y, m), "by_month": {mo: rmse(y[t["month"].eq(mo)], m[t["month"].eq(mo).to_numpy()]) for mo in ["2025-11", "2025-12"]},
                                        "matched": rmse(y[matched], m[matched]), "unmatched": rmse(y[~matched], m[~matched]),
                                        "seed_rmse_mean": float(np.mean([rmse(y, p) for p in preds]))}
    days = t["day"].to_numpy()
    uniq = np.unique(days)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    se = {k: (v - y) ** 2 for k, v in means.items()}
    day_sse = {k: np.array([v[idx[d]].sum() for d in uniq]) for k, v in se.items()}
    n_day = np.array([len(idx[d]) for d in uniq], dtype=float)
    rng = np.random.default_rng(20260927)
    report["effects"] = {}
    for long in ["g24", "g32"]:
        entry = {}
        for fold in FOLDS:
            entry[f"{fold}_rmse_change_vs_g16_pct"] = 100 * (report["folds"][fold][long]["rmse"] / report["folds"][fold]["g16"]["rmse"] - 1)
        # difference-in-differences of RMSE effects (percentage points), day bootstrap
        draws = []
        for _ in range(args.replicates):
            pick = rng.integers(0, len(uniq), len(uniq))
            n = n_day[pick].sum()
            eff = {}
            for fold in FOLDS:
                r_long = np.sqrt(day_sse[(fold, long)][pick].sum() / n)
                r_base = np.sqrt(day_sse[(fold, "g16")][pick].sum() / n)
                eff[fold] = 100 * (r_long / r_base - 1)
            draws.append(eff["gap"] - eff["near"])
        entry["gap_minus_near_pct_points"] = entry["gap_rmse_change_vs_g16_pct"] - entry["near_rmse_change_vs_g16_pct"]
        entry["gap_minus_near_95pct_interval"] = [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]
        entry["interpretation"] = "positive = the longer training helps less (or hurts more) when the scored months are further away"
        report["effects"][long] = entry
    write_json(ROOT / args.report, report)
    for fold in FOLDS:
        print(fold, {v: round(report["folds"][fold][v]["rmse"], 3) for v in VARIANTS})
    print({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in e.items() if kk != "interpretation"} for k, e in report["effects"].items()})


if __name__ == "__main__":
    main()
