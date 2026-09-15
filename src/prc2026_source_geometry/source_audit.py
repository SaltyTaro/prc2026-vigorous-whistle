"""Descriptive reporting relationships on the original common fitting rows."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from prc2026.artifacts import ResourceMonitor, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026_phase4.features import quality_features
from .common import CONFIG, checked_config, load_fitting


def finite_summary(values):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"rows": 0}
    return {"rows": len(array), "mean_seconds": float(array.mean()),
            "rmse_seconds": float(np.sqrt(np.mean(array * array))),
            "quantiles_seconds": dict(zip(["min", "p01", "p10", "median", "p90", "p99", "max"],
                                           map(float, np.quantile(array, [0, .01, .1, .5, .9, .99, 1]))))}


def near_nonzero_multiple(values, period, tolerance):
    values = np.asarray(values, dtype=float)
    nearest = np.rint(values / period)
    return np.isfinite(values) & (nearest != 0) & (np.abs(values - nearest * period) <= tolerance)


def prepare(frame, config):
    result = frame.copy()
    y = frame[TARGET].to_numpy(dtype=float)
    interval = (frame[TIME] - frame.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    quality = quality_features(frame)
    conditional = quality.conditional_schedule_offset_s.to_numpy(dtype=float)
    gate = frame.ADEP_mvt.eq("LIRF") & frame.ADEP_flt.isna() & frame.AIRCRAFT_TYPE_mvt.isna()
    if not np.array_equal(conditional, np.where(gate & np.isfinite(interval), interval, 0)):
        raise ValueError("The frozen schedule offset was not preserved.")
    aobt = (frame[TIME] - frame.AOBT_3_flt).dt.total_seconds().to_numpy()
    lobt = (frame[TIME] - frame.LOBT_flt).dt.total_seconds().to_numpy()
    available = np.isfinite(aobt) | np.isfinite(lobt)
    result["airport"] = frame.ADEP_mvt
    result["month"] = frame[TIME].dt.strftime("%Y-%m")
    result["nm_available"] = available
    result["aircraft_missing"] = frame.AIRCRAFT_TYPE_mvt.isna()
    result["stand"] = frame.STAND_mvt.astype("string").str.strip().str.upper().fillna("__MISSING__")
    result["schedule_midnight"] = frame.SCHED_TIME_UTC_mvt.eq(frame.SCHED_TIME_UTC_mvt.dt.floor("D"))
    edges = [-np.inf, *config["source_audit"]["schedule_interval_bins_seconds"], np.inf]
    result["schedule_band"] = pd.cut(interval, edges, right=False).astype("string").fillna("missing")
    result["schedule_minus_block_s"] = y - interval
    result["aobt_minus_block_s"] = y - aobt
    result["lobt_minus_block_s"] = y - lobt
    proxy = np.where(np.isfinite(aobt), aobt, lobt)
    result["raw_frozen_offset_residual_s"] = y - np.where(available, proxy, conditional)
    result["aobt_lobt_difference_s"] = (frame.AOBT_3_flt - frame.LOBT_flt).dt.total_seconds()
    threshold = config["source_audit"]["near_clock_seconds"]
    for name in ["schedule", "aobt", "lobt"]:
        result[f"block_near_{name}"] = result[f"{name}_minus_block_s"].abs().le(threshold)
    result["schedule_day_multiple"] = near_nonzero_multiple(result.schedule_minus_block_s, 86400,
                                        config["source_audit"]["day_multiple_tolerance_seconds"])
    result["nm_hour_multiple"] = near_nonzero_multiple(result.aobt_minus_block_s, 3600,
                                        config["source_audit"]["hour_multiple_tolerance_seconds"])
    result["target_over_3h"] = frame[TARGET].gt(10800)
    result["target_negative"] = frame[TARGET].lt(0)
    return result


def summarize(group):
    flags = ["block_near_schedule", "block_near_aobt", "block_near_lobt", "schedule_day_multiple",
             "nm_hour_multiple", "target_over_3h", "target_negative"]
    result = {"rows": len(group), "active_months": int(group.month.nunique()),
              "counts": {name: int(group[name].sum()) for name in flags}}
    for name in ["schedule_minus_block_s", "aobt_minus_block_s", "lobt_minus_block_s", "raw_frozen_offset_residual_s"]:
        result[name] = finite_summary(group[name])
    return result


def main(root):
    destination = root / "docs/source_geometry/fitting_source_audit.json"
    if destination.exists():
        raise ValueError("Preserve the original fitting-source audit.")
    config = checked_config(root)
    with ResourceMonitor() as monitor:
        raw, receipt = load_fitting(root, config, with_labels=True)
        data = prepare(raw, config)
        specifications = {
            "airport": ["airport"], "airport_month": ["airport", "month"],
            "airport_nm_availability": ["airport", "nm_available"],
            "airport_nm_availability_aircraft_missing": ["airport", "nm_available", "aircraft_missing"],
            "airport_nm_availability_schedule_interval": ["airport", "nm_available", "schedule_band"],
            "airport_nm_availability_stand": ["airport", "nm_available", "stand"],
            "airport_nm_availability_schedule_midnight": ["airport", "nm_available", "schedule_midnight"],
        }
        groups, stand_months = {}, []
        for name, keys in specifications.items():
            groups[name] = []
            for values, part in data.groupby(keys, observed=True, sort=True, dropna=False):
                values = values if isinstance(values, tuple) else (values,)
                if name.endswith("_stand") and (len(part) < config["source_audit"]["minimum_stand_group_rows"]
                        or part.month.nunique() < config["source_audit"]["minimum_stand_active_months"]):
                    continue
                key = dict(zip(keys, [v.item() if isinstance(v, np.generic) else v for v in values]))
                groups[name].append({"key": key, **summarize(part)})
                if name.endswith("_stand"):
                    for month, sample in part.groupby("month", sort=True):
                        stand_months.append({**key, "month": month, "rows": len(sample),
                            "near_schedule_rows": int(sample.block_near_schedule.sum()),
                            "over3h_rows": int(sample.target_over_3h.sum()),
                            "raw_offset_residual_median_seconds": float(sample.raw_frozen_offset_residual_s.median())})
        comparisons = []
        for airport, part in data.loc[data.nm_available].groupby("airport", sort=True):
            for threshold in config["source_audit"]["clock_disagreement_thresholds_seconds"]:
                subset = part.loc[part.aobt_lobt_difference_s.abs().gt(threshold)]
                comparisons.append({"airport": airport, "clock_disagreement_over_seconds": threshold,
                    "rows": len(subset), "aobt_closer_rows": int((subset.aobt_minus_block_s.abs() < subset.lobt_minus_block_s.abs()).sum()),
                    "lobt_closer_rows": int((subset.lobt_minus_block_s.abs() < subset.aobt_minus_block_s.abs()).sum()),
                    "aobt_raw_residual": finite_summary(subset.aobt_minus_block_s),
                    "lobt_raw_residual": finite_summary(subset.lobt_minus_block_s)})
    rows_path = root / "docs/source_geometry/stand_month_recurrence.csv"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    if rows_path.exists():
        raise ValueError("Preserve the original recurrence table.")
    pd.DataFrame(stand_months).to_csv(rows_path, index=False)
    report = {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(root / CONFIG),
        "population": receipt, "fitting_months": sorted(data.month.unique()),
        "groups": groups, "clock_comparisons": comparisons, "resources": monitor.stats,
        "recurrence_table_sha256": sha256_file(rows_path), "audit_used": False,
        "models_fitted": 0, "predictions_changed": False,
        "code_sha256": {str(path.relative_to(root)).replace('\\', '/'): sha256_file(path)
                          for path in [Path(__file__), Path(__file__).with_name("common.py"), root / "tools/audit_source_reporting.py"]},
        "interpretation": config["source_audit"]["interpretation"]}
    json.dumps(report, allow_nan=False)
    write_json(destination, report)
    print(json.dumps({"rows": len(data), "airport_routes": [
        {"key": item["key"], "rows": item["rows"], "counts": item["counts"]}
        for item in groups["airport_nm_availability"]]}, indent=2))
