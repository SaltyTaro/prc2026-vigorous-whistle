"""Day-level density features for the LIRF population classifier; label-free by construction.

Every quantity counts or averages supplied movement columns of the same local day
(schedule intervals, NM clock presence, population membership). No departure label
or departure block time is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME
from prc2026_lirf_mixture.features import AIRPORT, DEPARTURE_COLUMNS, FEATURE_NAMES, population_rows

DENSITY_NAMES = ["day_departures", "day_noclock_departures", "day_delayed_share",
                 "day_population_rows", "day_population_mean_s", "window2h_population_rows"]
EXTENDED_FEATURE_NAMES = FEATURE_NAMES + DENSITY_NAMES


def _seconds(stamps: pd.Series) -> np.ndarray:
    return stamps.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9


def density_features(population: pd.DataFrame, movements: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Six declared day-level quantities for each population row."""
    if set(DEPARTURE_COLUMNS) - set(movements):
        raise ValueError("Missing movement columns for density features.")
    spec = config["density_features"]
    if spec["names"] != DENSITY_NAMES:
        raise ValueError("Declared density feature names differ from the implementation.")
    zone, threshold, window = spec["timezone"], float(spec["delayed_threshold_seconds"]), float(spec["window_seconds"])
    departures = movements.loc[movements[PHASE].eq("DEP") & movements["ADEP_mvt"].eq(AIRPORT)]
    if departures[TIME].isna().any() or departures["SCHED_TIME_UTC_mvt"].isna().any():
        raise ValueError("Density features need complete movement and schedule clocks.")
    date = departures[TIME].dt.tz_convert(zone).dt.strftime("%Y-%m-%d")
    interval = (departures[TIME] - departures["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    noclock = departures["AOBT_3_flt"].isna() & departures["LOBT_flt"].isna()
    days = (pd.DataFrame({"date": date.to_numpy(), "noclock": noclock.to_numpy(), "delayed": (interval > threshold).to_numpy()})
            .groupby("date").agg(day_departures=("noclock", "size"), day_noclock_departures=("noclock", "sum"), day_delayed_share=("delayed", "mean")))
    if set(population[ID]) - set(population_rows(movements)[ID]):
        raise ValueError("Population rows must come from the same movements.")
    population_date = population[TIME].dt.tz_convert(zone).dt.strftime("%Y-%m-%d")
    population_interval = (population[TIME] - population["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    population_days = (pd.DataFrame({"date": population_date.to_numpy(), "s": population_interval.to_numpy()})
                       .groupby("date").agg(day_population_rows=("s", "size"), day_population_mean_s=("s", "mean")))
    stamps = _seconds(population[TIME])
    ordered = np.sort(stamps)
    within = np.searchsorted(ordered, stamps + window, side="right") - np.searchsorted(ordered, stamps - window, side="left")
    out = pd.DataFrame({ID: population[ID].to_numpy(), "date": population_date.to_numpy()})
    out = out.merge(days, left_on="date", right_index=True, how="left").merge(population_days, left_on="date", right_index=True, how="left")
    out["window2h_population_rows"] = within.astype(float)
    out = out.drop(columns=["date"])
    for name in ["day_departures", "day_noclock_departures", "day_population_rows"]:
        out[name] = out[name].to_numpy(dtype=float)
    values = out[DENSITY_NAMES].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Density features must be finite for every population row.")
    return out[[ID, *DENSITY_NAMES]]
