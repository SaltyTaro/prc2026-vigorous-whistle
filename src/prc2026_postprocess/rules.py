"""Fixed post-processing rules for extreme rows; parameters come from fitting labels only.

Rules are pure functions of raw movement columns, an incumbent prediction and a
no-clock fallback prediction. Nothing here reads scoring labels: `fit_parameters`
sees fitting rows, `apply_rules` sees no label at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, TARGET, TIME

POLICY = "declared-extreme-row-rules-v1"
RULE_NAMES = ["route_negative_proxy", "hedge_long_schedule", "floor"]
RAW_COLUMNS = [ID, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt", "LOBT_flt"]
HEDGE_AIRPORT = "LIRF"
ONE_DAY_SECONDS = 86400.0


def checked_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if set(RAW_COLUMNS) - set(frame):
        raise ValueError("Missing raw movement columns for post-processing rules.")
    rows = frame[RAW_COLUMNS].reset_index(drop=True)
    if rows[ID].isna().any() or rows[ID].duplicated().any():
        raise ValueError("Rule inputs need unique, nonnull movement IDs.")
    for name in [TIME, "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt"]:
        if not isinstance(rows[name].dtype, pd.DatetimeTZDtype) or str(rows[name].dt.tz) != "UTC":
            raise ValueError("Rule clocks require explicit UTC timestamps.")
    if rows[TIME].isna().any() or rows["SCHED_TIME_UTC_mvt"].isna().any():
        raise ValueError("Movement and schedule clocks must be complete.")
    return rows


def row_context(frame: pd.DataFrame) -> pd.DataFrame:
    """Label-free routing quantities, defined exactly as the frozen NM clock features."""
    rows = checked_rows(frame)
    aobt = (rows[TIME] - rows["AOBT_3_flt"]).dt.total_seconds().to_numpy(dtype=float)
    lobt = (rows[TIME] - rows["LOBT_flt"]).dt.total_seconds().to_numpy(dtype=float)
    have_aobt, have_lobt = np.isfinite(aobt), np.isfinite(lobt)
    proxy = np.where(have_aobt, aobt, lobt)
    schedule = (rows[TIME] - rows["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(dtype=float)
    context = pd.DataFrame({
        ID: rows[ID].to_numpy(),
        "clock_available": have_aobt | have_lobt,
        "proxy_s": proxy,
        "schedule_interval_s": schedule,
        "hedge_population": (rows["ADEP_mvt"].eq(HEDGE_AIRPORT) & ~(have_aobt | have_lobt)
                             & rows["AIRCRAFT_TYPE_mvt"].isna()).to_numpy(),
    })
    if not np.array_equal(context["clock_available"].to_numpy(), np.isfinite(context["proxy_s"].to_numpy())):
        raise ValueError("Clock availability and finite proxy disagree.")
    if not np.isfinite(schedule).all():
        raise ValueError("Schedule interval must be finite.")
    return context


def fit_parameters(fit_frame: pd.DataFrame, config: dict) -> dict:
    """Rule parameters from fitting departures and their labels; never from scoring rows."""
    rules = config["rules"]
    labels = fit_frame[TARGET].to_numpy(dtype=float)
    if len(labels) == 0 or not np.isfinite(labels).all():
        raise ValueError("Fitting labels must be complete and finite.")
    context = row_context(fit_frame)
    threshold = float(rules["hedge_long_schedule"]["schedule_interval_threshold_seconds"])
    low, high = (float(v) for v in rules["hedge_long_schedule"]["one_day_label_interval_seconds"])
    population = context["hedge_population"].to_numpy()
    one_day = population & (labels >= low) & (labels <= high)
    group = population & (context["schedule_interval_s"].to_numpy() > threshold)
    schedule_fill = group & (np.abs(labels - context["schedule_interval_s"].to_numpy()) <= 120)
    ordinary = group & (labels < 3600)
    if group.sum() == 0 or one_day.sum() == 0:
        raise ValueError("Fitting rows lack the long-schedule group or one-day rows.")
    return {
        "floor_seconds": float(labels.min()),
        "fitting_rows": int(len(labels)),
        "negative_fitting_labels": int((labels < 0).sum()),
        "hedge_population_rows": int(population.sum()),
        "one_day_rows": int(one_day.sum()),
        "one_day_taxi_median_seconds": float(np.median(labels[one_day] - ONE_DAY_SECONDS)),
        "long_schedule_group_rows": int(group.sum()),
        "long_schedule_group_one_day_rows": int((group & one_day).sum()),
        "long_schedule_group_schedule_fill_rows": int(schedule_fill.sum()),
        "long_schedule_group_ordinary_rows": int(ordinary.sum()),
        "one_day_share": float((group & one_day).sum() / group.sum()),
        "schedule_interval_threshold_seconds": threshold,
    }


def apply_rules(frame: pd.DataFrame, prediction, fallback, params: dict, rules: list[str]) -> tuple[np.ndarray, dict]:
    """Apply the declared rules in declared order; returns new predictions and row counts."""
    unknown = set(rules) - set(RULE_NAMES)
    if unknown:
        raise ValueError(f"Undeclared rule: {sorted(unknown)}")
    if list(rules) != [name for name in RULE_NAMES if name in rules]:
        raise ValueError("Rules must follow the declared order.")
    context = row_context(frame)
    values = np.array(prediction, dtype=float, copy=True)
    fallback = np.asarray(fallback, dtype=float)
    if values.shape != (len(context),) or fallback.shape != values.shape:
        raise ValueError("Predictions and fallback must align with the rows.")
    if not np.isfinite(values).all() or not np.isfinite(fallback).all():
        raise ValueError("Predictions and fallback must be finite.")
    counts = {}
    if "route_negative_proxy" in rules:
        mask = context["clock_available"].to_numpy() & (context["proxy_s"].to_numpy() < 0)
        values[mask] = fallback[mask]
        counts["route_negative_proxy"] = int(mask.sum())
    if "hedge_long_schedule" in rules:
        schedule = context["schedule_interval_s"].to_numpy()
        mask = context["hedge_population"].to_numpy() & (schedule > params["schedule_interval_threshold_seconds"])
        share, median = params["one_day_share"], params["one_day_taxi_median_seconds"]
        values[mask] = (1 - share) * schedule[mask] + share * (ONE_DAY_SECONDS + median)
        counts["hedge_long_schedule"] = int(mask.sum())
    if "floor" in rules:
        mask = values < params["floor_seconds"]
        values[mask] = params["floor_seconds"]
        counts["floor"] = int(mask.sum())
    counts["changed_rows"] = int((values != np.asarray(prediction, dtype=float)).sum())
    return values, counts
