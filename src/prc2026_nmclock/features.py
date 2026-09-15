"""Retrospective clock differences; never departure labels or airport block times."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, TIME
from prc2026_weather.features import (
    FEATURE_NAMES as PREVIOUS_NAMES, CORE_NAMES, CORE_CATEGORIES, WEATHER_NAMES,
)

POLICY = "supplied-nm-offblock-clock-v1"
INPUT_COLUMNS = [ID, TIME, "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt", "LOBT_flt"]
CLOCKS = {"aobt": "AOBT_3_flt", "lobt": "LOBT_flt"}
PER_CLOCK = ["missing", "movement_minus_s", "minus_schedule_s", "minus_eobt1_s", "minus_iobt_s",
             "movement_day_delta", "utc_hour_sin", "utc_hour_cos", "second_of_minute",
             "negative_interval", "over3h_interval"]
CLOCK_NAMES = ([f"nmclock_{clock}_{name}" for clock in CLOCKS for name in PER_CLOCK]
               + ["nmclock_available", "nmclock_source", "nmclock_proxy_s", "nmclock_clock_difference_s",
                  "nmclock_agree_60s", "nmclock_both_present", "nmclock_proxy_mod_day_s"])
FEATURE_NAMES = list(PREVIOUS_NAMES) + CLOCK_NAMES
MODEL_NAMES = list(CORE_NAMES) + WEATHER_NAMES + CLOCK_NAMES
CATEGORICAL_COLUMNS = list(CORE_CATEGORIES)


def clock_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw[INPUT_COLUMNS].reset_index(drop=True).copy()
    ids = data[ID].to_numpy(dtype=float)
    if not np.isfinite(ids).all() or np.any(ids % 1) or data[ID].duplicated().any():
        raise ValueError("Unique finite integral movement IDs required.")
    for name in INPUT_COLUMNS[1:]:
        if not isinstance(data[name].dtype, pd.DatetimeTZDtype) or str(data[name].dt.tz) != "UTC":
            raise ValueError("Clock inputs must have explicit UTC timestamps.")
    if data[TIME].isna().any():
        raise ValueError("Movement time is required for every row.")
    allowed = [f"2025-{month:02}" for month in range(1, 12)] + ["2026-01", "2026-07"]
    if not data[TIME].dt.strftime("%Y-%m").isin(allowed).all():
        raise ValueError("Reserved audit or undeclared movement period.")
    out = {ID: ids}
    for name, column in CLOCKS.items():
        stamp = data[column]
        duration = (data[TIME]-stamp).dt.total_seconds().to_numpy(dtype=float)
        hour = stamp.dt.hour + stamp.dt.minute/60 + stamp.dt.second/3600 + stamp.dt.microsecond/3.6e9
        values = {
            "missing": stamp.isna().to_numpy(dtype=float),
            "movement_minus_s": duration,
            "minus_schedule_s": (stamp-data["SCHED_TIME_UTC_mvt"]).dt.total_seconds(),
            "minus_eobt1_s": (stamp-data["EOBT_1_flt"]).dt.total_seconds(),
            "minus_iobt_s": (stamp-data["IOBT_flt"]).dt.total_seconds(),
            "movement_day_delta": (data[TIME].dt.normalize()-stamp.dt.normalize()).dt.total_seconds()/86400,
            "utc_hour_sin": np.sin(2*np.pi*hour/24),
            "utc_hour_cos": np.cos(2*np.pi*hour/24),
            "second_of_minute": stamp.dt.second + stamp.dt.microsecond/1e6,
            "negative_interval": np.where(np.isfinite(duration), (duration < 0).astype(float), np.nan),
            "over3h_interval": np.where(np.isfinite(duration), (duration > 10800).astype(float), np.nan),
        }
        out.update({f"nmclock_{name}_{key}": np.asarray(value, dtype=float) for key, value in values.items()})
    aobt, lobt = out["nmclock_aobt_movement_minus_s"], out["nmclock_lobt_movement_minus_s"]
    have_aobt, have_lobt = np.isfinite(aobt), np.isfinite(lobt)
    both = have_aobt & have_lobt
    proxy = np.where(have_aobt, aobt, lobt)
    difference = (data["AOBT_3_flt"]-data["LOBT_flt"]).dt.total_seconds().to_numpy(dtype=float)
    out.update(nmclock_available=(have_aobt | have_lobt).astype(float),
               nmclock_source=np.where(have_aobt, 1., np.where(have_lobt, 2., 0.)),
               nmclock_proxy_s=proxy, nmclock_clock_difference_s=difference,
               nmclock_agree_60s=np.where(both, (np.abs(difference) <= 60).astype(float), np.nan),
               nmclock_both_present=both.astype(float), nmclock_proxy_mod_day_s=np.mod(proxy, 86400.))
    result = pd.DataFrame(out)[[ID, *CLOCK_NAMES]]
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("Clock features cannot contain infinities.")
    return result


def route_mask(features: pd.DataFrame) -> np.ndarray:
    available = features["nmclock_available"].to_numpy(dtype=float)
    proxy = features["nmclock_proxy_s"].to_numpy(dtype=float)
    if not np.isin(available, [0., 1.]).all() or not np.array_equal(available == 1, np.isfinite(proxy)):
        raise ValueError("Clock availability and finite proxy disagree.")
    return available == 1


def target_offset(features: pd.DataFrame, target: str) -> np.ndarray:
    if target == "direct":
        return np.zeros(len(features), dtype=float)
    if target == "nm_residual":
        if not route_mask(features).all():
            raise ValueError("Residual nodes require a finite clock for every routed row.")
        return features["nmclock_proxy_s"].to_numpy(dtype=float)
    raise ValueError("Unknown clock target formulation.")


def coverage(frame: pd.DataFrame) -> dict:
    values = frame["nmclock_proxy_s"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    return {"rows": len(frame), "clock_available_rows": int(route_mask(frame).sum()),
            "lobt_only_rows": int(frame["nmclock_source"].eq(2).sum()),
            "both_present_rows": int(frame["nmclock_both_present"].sum()),
            "clocks_disagree_over_60s_rows": int(frame["nmclock_agree_60s"].eq(0).sum()),
            "negative_proxy_rows": int((finite < 0).sum()), "over_three_hour_proxy_rows": int((finite > 10800).sum()),
            "proxy_quantiles_seconds": dict(zip(["minimum", "p01", "median", "p99", "maximum"],
                map(float, np.quantile(finite, [0, .01, .5, .99, 1])))) if finite.size else {}}
