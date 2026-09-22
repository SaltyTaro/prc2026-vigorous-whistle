"""What does the ranking months' ADS-B coverage cost us? Read-only estimate from measured class errors.

December 2025 is fully covered and has labels; the ranking months are not (EDDF's gate share falls from
38.0% to 7.8%/3.8%, LIRF loses the surface entirely in July). This reweights December's measured per-class,
per-airport squared error by each ranking month's observed class profile, giving a first-order estimate of
the RMSE we lose to coverage alone.

The estimate assumes a row's error depends on its coverage class and airport, not on which particular rows
fall into each class. That is first order: rows that stay matched at a poorly covered airport are probably
not a random sample. It is an estimate of the penalty, not a prediction of the leaderboard.

Nothing is fitted and nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from run_adsb_v24 import load_frame  # noqa: E402

CLASSES = ["gate", "dark", "unmatched"]


def december_classes(config: dict, cache: Path) -> pd.DataFrame:
    """December rows with their coverage class and the shipped v23 squared error."""
    table = load_frame(config, "2025-12", "cross_fit")
    ids = table[ID].to_numpy()
    prediction = pd.read_parquet(ROOT / "runs/adsb_v25/dev/predictions_2025-12.parquet").set_index(ID)["mean_adj2_cat"].reindex(ids).to_numpy(dtype=float)
    features = pd.read_parquet(cache / "matches_v11_2025-12.parquet").set_index(ID).reindex(ids)
    matched = features["adsb_matched"].eq(1).to_numpy()
    gate = features["adsb_gate"].eq(1).to_numpy()
    label = table[TARGET].to_numpy(dtype=float)
    return pd.DataFrame({"airport": table["airport"].to_numpy(),
                         "cls": np.where(~matched, "unmatched", np.where(gate, "gate", "dark")),
                         "sq": np.square(label - prediction)})


def profile(census: dict, month: str) -> pd.DataFrame:
    """Observed class shares per airport in a ranking month: gate, dark, and everything else as unmatched."""
    rows = []
    for airport, counts in census["months"][month]["by_airport"].items():
        total = sum(counts.values())
        gate, dark = counts["gate"], counts["dark"]
        rows.append({"airport": airport, "rows": total, "gate": gate / total, "dark": dark / total,
                     "unmatched": (total - gate - dark) / total})
    return pd.DataFrame(rows).set_index("airport")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--output", type=Path, default=Path("docs/adsb_coverage/coverage_shift.json"))
    args = parser.parse_args()
    config = read_json(ROOT / "configs/adsb_v27_v1.json")
    december = december_classes(config, ROOT / args.cache)
    census = {**read_json(ROOT / "docs/adsb_coverage/census_probe.json")["months"],
              **read_json(ROOT / "docs/adsb_coverage/census_jul_dec.json")["months"]}
    census = {"months": census}

    mean_sq = december.groupby(["airport", "cls"])["sq"].mean().unstack()
    counts = december.groupby(["airport", "cls"])["sq"].size().unstack().fillna(0)
    observed = float(np.sqrt(december["sq"].mean()))

    report = {"created_at_utc": utc_now(), "audit_used": False,
              "december_rmse_observed": observed,
              "december_rows": int(len(december)),
              "december_class_rmse_by_airport": {a: {c: (float(np.sqrt(mean_sq.loc[a, c])) if c in mean_sq.columns and np.isfinite(mean_sq.loc[a, c]) else None) for c in CLASSES} for a in mean_sq.index},
              "december_class_rows_by_airport": {a: {c: int(counts.loc[a, c]) if c in counts.columns else 0 for c in CLASSES} for a in counts.index},
              "months": {}}

    for month in ["2025-12", "2026-01", "2026-07"]:
        shares = profile(census, month)
        rows, sse, missing = 0, 0.0, []
        per_airport = {}
        for airport in shares.index:
            n = int(shares.loc[airport, "rows"])
            if airport not in mean_sq.index:
                missing.append(airport)
                continue
            airport_sse, airport_rows = 0.0, 0
            for cls in CLASSES:
                share = float(shares.loc[airport, cls])
                if share <= 0:
                    continue
                value = mean_sq.loc[airport, cls] if cls in mean_sq.columns else np.nan
                if not np.isfinite(value):  # no December row of that class at that airport
                    value = mean_sq.loc[airport].dropna().max()  # the worst class we did observe there
                airport_sse += share * n * float(value)
                airport_rows += share * n
            per_airport[airport] = {"rows": n, "shares": {c: round(float(shares.loc[airport, c]), 4) for c in CLASSES},
                                    "implied_rmse": float(np.sqrt(airport_sse / airport_rows)) if airport_rows else None}
            sse += airport_sse
            rows += airport_rows
        report["months"][month] = {"implied_rmse_on_december_errors": float(np.sqrt(sse / rows)),
                                   "airports_without_december_reference": missing,
                                   "by_airport": per_airport}
    base = report["months"]["2025-12"]["implied_rmse_on_december_errors"]
    for month, entry in report["months"].items():
        entry["vs_december_profile_pct"] = round(100 * (entry["implied_rmse_on_december_errors"] / base - 1), 3)
    write_json(ROOT / args.output, report)
    print(f"December observed RMSE {observed:.2f} on {len(december)} rows")
    for month, entry in report["months"].items():
        print(f"  under the {month} coverage profile: {entry['implied_rmse_on_december_errors']:.2f}  ({entry['vs_december_profile_pct']:+.2f}%)")
    print()
    for airport in sorted(report["months"]["2026-07"]["by_airport"]):
        a = {m: report["months"][m]["by_airport"].get(airport) for m in ["2025-12", "2026-01", "2026-07"]}
        if all(a.values()):
            print(f"  {airport}  Dec {a['2025-12']['implied_rmse']:7.1f} | Jan {a['2026-01']['implied_rmse']:7.1f} | Jul {a['2026-07']['implied_rmse']:7.1f}"
                  f"   gate share {a['2025-12']['shares']['gate']:.2f} -> {a['2026-01']['shares']['gate']:.2f} -> {a['2026-07']['shares']['gate']:.2f}")


if __name__ == "__main__":
    main()
