"""Readings of the season-weighted network test (configs/adsb_ctx_season_v1.json): for each read month (January from sw1, July from sw7,
December from sw12) the 4-seed mean weighted prediction against the 4-seed mean of the seed-matched controls (GPU development seeds
401-404), on non-LIRF rows: RMSE change overall / ADS-B matched / unmatched and by airport, paired-day bootstrap interval of the RMSE
difference; plus the unweighted reproduction sw0_s401 against control s401. Descriptive; writes docs/adsb_ctx_season/report.json.
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

FILES = {"2025-01": "training_2025-01-01_2025-02-01", "2025-07": "training_2025-07-01_2025-08-01", "2025-12": "training_2025-12-01_2026-01-01"}


def truth(month: str) -> pd.DataFrame:
    base = "data/december/d198dea10625b0a75977" if month == "2025-12" else "data/phase1/58d621b245de3f5e12dd"
    labels = pd.read_parquet(ROOT / base / "labels" / f"{FILES[month]}.parquet")
    raw = pd.read_parquet(ROOT / f"data/raw/{FILES[month]}.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", TIME])
    adsb = pd.read_parquet(ROOT / f"runs/adsb_v24/features/matches_{month}.parquet", columns=[ID, "adsb_matched"])
    out = labels.merge(raw[raw["PHASE_mvt"].eq("DEP")], on=ID).merge(adsb, on=ID, how="left")
    out = out[out["ADEP_mvt"].ne("LIRF")].drop_duplicates(ID).reset_index(drop=True)
    out["day"] = out[TIME].dt.strftime("%Y-%m-%d")
    return out


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def interval(days, y, a, b, rng, replicates=4000):
    uniq, inv = np.unique(days, return_inverse=True)
    sa, sb, n = np.bincount(inv, (a - y) ** 2), np.bincount(inv, (b - y) ** 2), np.bincount(inv).astype(float)
    draws = [np.sqrt(sa[p].sum() / n[p].sum()) - np.sqrt(sb[p].sum() / n[p].sum()) for p in (rng.integers(0, len(uniq), len(uniq)) for _ in range(replicates))]
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def load(path: Path, ids: pd.Series) -> np.ndarray:
    values = pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"{path} does not cover every read row.")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx_season"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx_season/report.json"))
    args = parser.parse_args()
    config = read_json(ROOT / "configs/adsb_ctx_season_v1.json")
    seeds = config["seeds"]
    rng = np.random.default_rng(20260927)
    report = {"created_at_utc": utc_now(), "seeds": seeds, "months": {}}
    for run, month, fold, file in [("sw1", "2025-01", "seasonal", "predictions.parquet"), ("sw7", "2025-07", "seasonal", "predictions.parquet"),
                                   ("sw12", "2025-12", "refit", "december_predictions.parquet")]:
        t = truth(month)
        ids, y = t[ID], t[TARGET].to_numpy(dtype=float)
        weighted = np.mean([load(ROOT / args.runs / f"{run}_s{s}" / fold / file, ids) for s in seeds], axis=0)
        control = np.mean([load(ROOT / (f"runs/adsb_ctx_gpu/ctx_e32_s{s}/seasonal/predictions.parquet" if fold == "seasonal"
                                        else f"runs/december/learners/runs__adsb_ctx_gpu__ctx_e32_s{s}.parquet"), ids) for s in seeds], axis=0)
        matched = t["adsb_matched"].eq(1).to_numpy()
        airports = t["ADEP_mvt"].to_numpy()
        entry = {"run": run, "rows": int(len(t)), "rmse_control": rmse(y, control), "rmse_weighted": rmse(y, weighted),
                 "change_pct": 100 * (rmse(y, weighted) / rmse(y, control) - 1),
                 "matched_change_pct": 100 * (rmse(y[matched], weighted[matched]) / rmse(y[matched], control[matched]) - 1),
                 "unmatched_change_pct": 100 * (rmse(y[~matched], weighted[~matched]) / rmse(y[~matched], control[~matched]) - 1),
                 "day_interval_seconds": interval(t["day"].to_numpy(), y, weighted, control, rng),
                 "airports": {a: round(100 * (rmse(y[airports == a], weighted[airports == a]) / rmse(y[airports == a], control[airports == a]) - 1), 3)
                              for a in sorted(set(airports))}}
        entry["passes_month"] = bool(entry["change_pct"] <= -0.30 and entry["day_interval_seconds"][1] < 0)
        report["months"][month] = entry
        print(month, run, round(entry["change_pct"], 3), [round(x, 2) for x in entry["day_interval_seconds"]], "unmatched", round(entry["unmatched_change_pct"], 3), flush=True)
    sw0 = ROOT / args.runs / "sw0_s401" / "seasonal" / "predictions.parquet"
    if sw0.exists():
        repro = {}
        for month in ["2025-01", "2025-07"]:
            t = truth(month)
            y = t[TARGET].to_numpy(dtype=float)
            a, b = load(sw0, t[ID]), load(ROOT / "runs/adsb_ctx_gpu/ctx_e32_s401/seasonal/predictions.parquet", t[ID])
            repro[month] = {"rmse_sw0": rmse(y, a), "rmse_control_s401": rmse(y, b), "rms_prediction_difference": float(np.sqrt(np.mean((a - b) ** 2)))}
        report["unweighted_reproduction"] = repro
        print("sw0 vs control s401", {m: (round(r["rmse_sw0"], 2), round(r["rmse_control_s401"], 2)) for m, r in repro.items()})
    report["go"] = all(e["passes_month"] for e in report["months"].values())
    write_json(ROOT / args.report, report)
    print("GO" if report["go"] else "no GO")


if __name__ == "__main__":
    main()
