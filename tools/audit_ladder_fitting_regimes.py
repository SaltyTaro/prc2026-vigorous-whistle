"""Describe source-clock residual regimes on the intersection of both fitting sets only."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_nmclock.bundle import load_bundle

NAMES = [
    "airport", "nm_missing", "aircraft_missing", "lirf_missing_record",
    "conditional_schedule_offset_s", "schedule_to_movement_s",
    "nmclock_available", "nmclock_source", "nmclock_proxy_s",
    "nmclock_both_present", "nmclock_agree_60s", "nmclock_clock_difference_s",
    "nmclock_aobt_movement_minus_s", "nmclock_lobt_movement_minus_s",
]
RESIDUAL_BINS = [-np.inf, -86400, -10800, -3600, -900, -120, 120, 900, 3600, 10800, 86400, np.inf]


def statistics(values):
    data = np.asarray(values, dtype=float)
    if not np.isfinite(data).all() or not len(data):
        raise ValueError("Complete finite fitting residuals are required.")
    quantiles = dict(zip(
        ["minimum", "p001", "p01", "p10", "median", "p90", "p99", "p999", "maximum"],
        map(float, np.quantile(data, [0, .001, .01, .1, .5, .9, .99, .999, 1])),
    ))
    total = float(np.square(data).sum())
    bins = []
    for lower, upper in zip(RESIDUAL_BINS[:-1], RESIDUAL_BINS[1:]):
        mask = (data >= lower) & (data < upper)
        sse = float(np.square(data[mask]).sum())
        bins.append({"lower_inclusive_seconds": float(lower) if np.isfinite(lower) else None,
                     "upper_exclusive_seconds": float(upper) if np.isfinite(upper) else None,
                     "rows": int(mask.sum()), "sse": sse,
                     "fraction_of_proxy_sse": sse/total if total else 0.})
    nearest_hour = np.rint(data/3600)
    hour_match = (nearest_hour != 0) & (np.abs(data-nearest_hour*3600) <= 60)
    nearest_day = np.rint(data/86400)
    day_match = (nearest_day != 0) & (np.abs(data-nearest_day*86400) <= 120)
    return {"rows": len(data), "bias_seconds": float(data.mean()),
            "rmse_seconds": float(np.sqrt(total/len(data))), "sse": total,
            "quantiles_seconds": quantiles, "bins": bins,
            "absolute_error_at_most_60_seconds_rows": int((np.abs(data) <= 60).sum()),
            "nonzero_hour_multiple_within_60_seconds_rows": int(hour_match.sum()),
            "nonzero_day_multiple_within_120_seconds_rows": int(day_match.sum()),
            "interpretation": "Error of the stated raw clock offset, not a trained-model or leaderboard score."}


def summarize(data):
    available = data["nmclock_available"].eq(1)
    output = {}
    for route, mask in [("clock_available", available), ("clock_missing", ~available)]:
        group = data.loc[mask]
        if not len(group):
            continue
        offset = group["nmclock_proxy_s"] if route == "clock_available" else group["conditional_schedule_offset_s"]
        residual = group[TARGET].to_numpy(dtype=float)-offset.to_numpy(dtype=float)
        output[route] = statistics(residual)
        output[route]["offset"] = "AOBT then LOBT" if route == "clock_available" else "unchanged conditional LIRF schedule offset"
        output[route]["target_above_3h_rows"] = int(group[TARGET].gt(10800).sum())
        output[route]["negative_target_rows"] = int(group[TARGET].lt(0).sum())
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root/"docs/ladder/fitting_clock_regimes.json"
    if destination.exists():
        raise ValueError("Preserve the original fitting-only diagnostic.")
    base = root/"data/nmclock/72e86725e23367ede5b5"
    manifest, parent, original, catalog = load_bundle(root, base)
    folds = {fold: pd.read_parquet(parent/"splits"/(fold+".parquet")) for fold in ["seasonal", "forward"]}
    fitting = set.intersection(*[set(frame.loc[frame["partition"].eq("fit"), ID]) for frame in folds.values()])
    protected = set.union(*[set(frame.loc[~frame["partition"].eq("fit"), ID]) for frame in folds.values()])
    if fitting & protected:
        raise ValueError("A held-out, stopping or reserved ID entered the diagnostic.")
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(assignment[ID].isin(fitting), "fit", "excluded")
    features, truth = load_partition(base, parent, catalog, assignment, "fit", NAMES)
    data = features.merge(truth[[ID, TIME, TARGET]], on=ID, validate="one_to_one")
    months = data[TIME].dt.strftime("%Y-%m")
    expected = {"2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09"}
    if set(months) != expected or len(data) != len(fitting) or identity_hash(data[ID]) != identity_hash(sorted(fitting)):
        raise ValueError("The protected fitting intersection changed.")
    groups = {"all": summarize(data)}
    for airport, group in data.groupby("airport", sort=True, observed=True):
        groups["airport/"+str(airport)] = summarize(group)
    for month, group in data.assign(month=months).groupby("month", sort=True, observed=True):
        groups["month/"+str(month)] = summarize(group)
    present = data.loc[data["nmclock_available"].eq(1)]
    for name, mask in [
        ("aobt_only", present["nmclock_source"].eq(1) & present["nmclock_both_present"].eq(0)),
        ("lobt_only", present["nmclock_source"].eq(2)),
        ("both_agree_60s", present["nmclock_agree_60s"].eq(1)),
        ("both_disagree_60s", present["nmclock_agree_60s"].eq(0)),
        ("proxy_negative", present["nmclock_proxy_s"].lt(0)),
        ("proxy_0_to_3h", present["nmclock_proxy_s"].between(0, 10800)),
        ("proxy_over_3h", present["nmclock_proxy_s"].gt(10800)),
    ]:
        groups["source/"+name] = summarize(present.loc[mask])
    paired = {}
    both = data.loc[data["nmclock_both_present"].eq(1)]
    for threshold in [60, 900, 3600]:
        group = both.loc[both["nmclock_clock_difference_s"].abs().gt(threshold)]
        if not len(group):
            continue
        aobt_error = group[TARGET].to_numpy(dtype=float)-group["nmclock_aobt_movement_minus_s"].to_numpy(dtype=float)
        lobt_error = group[TARGET].to_numpy(dtype=float)-group["nmclock_lobt_movement_minus_s"].to_numpy(dtype=float)
        paired[str(threshold)] = {
            "rows": len(group), "aobt_residual": statistics(aobt_error), "lobt_residual": statistics(lobt_error),
            "aobt_closer_rows": int((np.abs(aobt_error) < np.abs(lobt_error)).sum()),
            "lobt_closer_rows": int((np.abs(lobt_error) < np.abs(aobt_error)).sum()),
            "equal_absolute_error_rows": int((np.abs(lobt_error) == np.abs(aobt_error)).sum()),
            "interpretation": "Descriptive fitting-only source comparison; no target-aware choice is made at prediction time.",
        }
    report = {
        "created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)),
        "bundle_id": manifest["bundle_id"], "parent_bundle_id": original["bundle_id"],
        "fitting_rows": len(data), "fitting_ids_sha256": identity_hash(data[ID]),
        "split_file_sha256": {fold: sha256_file(parent/"splits"/(fold+".parquet")) for fold in folds},
        "actual_months": sorted(months.unique()), "feature_columns": NAMES,
        "scope": "Intersection of both original fitting partitions; no scoring, stopping, reserved December or ranking labels.",
        "audit_used": False, "models_fitted": 0, "predictions_changed": False,
        "groups": groups, "aobt_lobt_comparisons_by_disagreement_threshold": paired,
        "interpretation": "Raw residual distributions guide a possible future declaration. This audit cannot change the active ladder recipes, weights or promotion rule.",
    }
    write_json(destination, report)
    print({"fitting_rows": len(data), "routes": groups["all"],
           "clock_disagreement_counts": {key: value["rows"] for key, value in paired.items()}})


if __name__ == "__main__":
    main()
