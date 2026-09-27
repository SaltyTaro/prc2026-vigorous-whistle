"""Readings of the per-airport fine-tuning test (configs/adsb_ctx_ft_design_v1.json) on October 2025: for each airport and setting the
4-seed mean fine-tuned prediction against the same 4 base networks' mean (both on the airport's October rows), RMSE change overall and
on ADS-B matched / unmatched rows, with a paired-day bootstrap interval of the RMSE difference; and, per setting, all airports
fine-tuned together on October non-LIRF rows. Descriptive; writes docs/adsb_ctx_ft/design_report.json.
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


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def day_interval(days, y, a, b, rng, replicates=4000):
    """95% interval of RMSE(a) - RMSE(b) over a day bootstrap."""
    uniq, inv = np.unique(days, return_inverse=True)
    sa, sb = np.bincount(inv, (a - y) ** 2), np.bincount(inv, (b - y) ** 2)
    n = np.bincount(inv).astype(float)
    draws = []
    for _ in range(replicates):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws.append(np.sqrt(sa[pick].sum() / n[pick].sum()) - np.sqrt(sb[pick].sum() / n[pick].sum()))
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx_ft"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx_ft/design_report.json"))
    args = parser.parse_args()
    config = read_json(ROOT / "configs/adsb_ctx_ft_design_v1.json")
    seeds = config["base"]["seeds"]
    labels = pd.read_parquet(ROOT / "data/phase1/58d621b245de3f5e12dd/labels/training_2025-10-01_2025-11-01.parquet")
    raw = pd.read_parquet(ROOT / "data/raw/training_2025-10-01_2025-11-01.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", TIME])
    raw = raw[raw["PHASE_mvt"].eq("DEP")]
    adsb = pd.read_parquet(ROOT / "runs/adsb_v24/features/matches_2025-10.parquet", columns=[ID, "adsb_matched"])
    t = labels.merge(raw, on=ID).merge(adsb, on=ID, how="left").set_index(ID)
    rng = np.random.default_rng(20260927)
    report = {"created_at_utc": utc_now(), "seeds": seeds, "airports": {}, "all_airports": {}, "go": {}}
    combined = {s: [] for s in config["settings"]}
    for airport in config["airports"]:
        report["airports"][airport] = {}
        for setting in config["settings"]:
            frames = [pd.read_parquet(ROOT / args.runs / f"e32_s{s}" / f"{airport}_{setting}" / "october.parquet").set_index(ID) for s in seeds]
            ids = frames[0].index
            if any(not f.index.equals(ids) for f in frames):
                raise ValueError(f"{airport} {setting}: seeds cover different rows.")
            ft = np.mean([f["adsb_full"].to_numpy() for f in frames], axis=0)
            base = np.mean([f["base"].to_numpy() for f in frames], axis=0)
            rows = t.reindex(ids)
            y = rows[TARGET].to_numpy(dtype=float)
            days = rows[TIME].dt.strftime("%Y-%m-%d").to_numpy()
            matched = rows["adsb_matched"].eq(1).to_numpy()
            entry = {"rows": int(len(ids)), "rmse_base": rmse(y, base), "rmse_ft": rmse(y, ft), "change_pct": 100 * (rmse(y, ft) / rmse(y, base) - 1),
                     "matched_change_pct": 100 * (rmse(y[matched], ft[matched]) / rmse(y[matched], base[matched]) - 1) if matched.any() else None,
                     "unmatched_change_pct": 100 * (rmse(y[~matched], ft[~matched]) / rmse(y[~matched], base[~matched]) - 1) if (~matched).any() else None,
                     "day_interval_seconds": day_interval(days, y, ft, base, rng)}
            entry["go"] = bool(entry["change_pct"] <= -1.0 and entry["day_interval_seconds"][1] < 0)
            report["airports"][airport][setting] = entry
            combined[setting].append(pd.DataFrame({ID: ids, "ft": ft, "base": base}))
    for setting, parts in combined.items():
        c = pd.concat(parts).set_index(ID)
        rows = t.reindex(c.index)
        y = rows[TARGET].to_numpy(dtype=float)
        days = rows[TIME].dt.strftime("%Y-%m-%d").to_numpy()
        report["all_airports"][setting] = {"rows": int(len(c)), "rmse_base": rmse(y, c["base"]), "rmse_ft": rmse(y, c["ft"]),
                                           "change_pct": 100 * (rmse(y, c["ft"]) / rmse(y, c["base"]) - 1),
                                           "day_interval_seconds": day_interval(days, y, c["ft"].to_numpy(), c["base"].to_numpy(), rng)}
    report["go"] = {a: [s for s, e in v.items() if e["go"]] for a, v in report["airports"].items()}
    write_json(ROOT / args.report, report)
    for a, v in report["airports"].items():
        print(a, {s: (round(e["change_pct"], 3), [round(x, 2) for x in e["day_interval_seconds"]]) for s, e in v.items()})
    print("all", {s: round(e["change_pct"], 3) for s, e in report["all_airports"].items()})


if __name__ == "__main__":
    main()
