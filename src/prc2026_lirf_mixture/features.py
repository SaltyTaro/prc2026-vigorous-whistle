"""Population selection and label-free features for the LIRF schedule-fill mixture.

All quantities derive from supplied movement columns and arrival in-block times,
which the ranking data keeps. No departure label or departure block time is read.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME

AIRPORT = "LIRF"
DEPARTURE_COLUMNS = [ID, PHASE, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
                     "AOBT_3_flt", "LOBT_flt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt"]
ARRIVAL_COLUMNS = [PHASE, "ADES_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt"]
NUMERIC = ["schedule_interval_s", "signed_log_schedule_interval", "local_hour", "scheduled_local_hour", "date_shift_days",
           "day_of_week", "month", "flight_missing", "flight_doubled_prefix", "flight_alpha_tail",
           "schedule_minus_previous_inblock_s", "takeoff_minus_previous_inblock_s", "next_inblock_minus_takeoff_s"]
CATEGORICAL = ["flight_prefix", "stand_prefix", "runway", "destination_region"]
FEATURE_NAMES = NUMERIC + CATEGORICAL


def _seconds(stamps: pd.Series) -> np.ndarray:
    return stamps.to_numpy(dtype="datetime64[ns]").astype("int64") / 1e9


def population_rows(movements: pd.DataFrame) -> pd.DataFrame:
    """LIRF departures without any NM clock and without an airport aircraft type."""
    if set(DEPARTURE_COLUMNS) - set(movements):
        raise ValueError("Missing movement columns for the LIRF population.")
    frame = movements.loc[movements[PHASE].eq("DEP") & movements["ADEP_mvt"].eq(AIRPORT)
                          & movements["AOBT_3_flt"].isna() & movements["LOBT_flt"].isna()
                          & movements["AIRCRAFT_TYPE_mvt"].isna()].copy()
    if frame[ID].duplicated().any():
        raise ValueError("Duplicate movement IDs in the population.")
    return frame.reset_index(drop=True)


def turnaround(population: pd.DataFrame, movements: pd.DataFrame, margin_seconds: float) -> pd.DataFrame:
    """Previous and next arrival in-block at the departure's stand, from supplied arrival rows only."""
    if set(ARRIVAL_COLUMNS) - set(movements):
        raise ValueError("Missing arrival columns for turnaround features.")
    arrivals = movements.loc[movements[PHASE].eq("ARR") & movements["ADES_mvt"].eq(AIRPORT)
                             & movements["STAND_mvt"].notna() & movements["BLOCK_TIME_UTC_mvt"].notna()]
    inblocks = {stand: np.sort(_seconds(group["BLOCK_TIME_UTC_mvt"])) for stand, group in arrivals.groupby("STAND_mvt")}
    takeoff = _seconds(population[TIME])
    previous, following = np.full(len(population), np.nan), np.full(len(population), np.nan)
    for i, (stand, t) in enumerate(zip(population["STAND_mvt"].to_numpy(), takeoff)):
        times = inblocks.get(stand)
        if times is None or not isinstance(stand, str):
            continue
        k = int(np.searchsorted(times, t - margin_seconds, side="left"))
        if k > 0:
            previous[i] = times[k - 1]
        if k < len(times):
            following[i] = times[k]
    schedule = _seconds(population["SCHED_TIME_UTC_mvt"])
    return pd.DataFrame({ID: population[ID].to_numpy(),
                         "schedule_minus_previous_inblock_s": schedule - previous,
                         "takeoff_minus_previous_inblock_s": takeoff - previous,
                         "next_inblock_minus_takeoff_s": following - takeoff})


def classifier_features(population: pd.DataFrame, movements: pd.DataFrame, margin_seconds: float) -> pd.DataFrame:
    stamps, schedule = population[TIME], population["SCHED_TIME_UTC_mvt"]
    if stamps.isna().any() or schedule.isna().any():
        raise ValueError("Population rows need movement and schedule clocks.")
    interval = (stamps - schedule).dt.total_seconds().to_numpy(dtype=float)
    local, scheduled_local = stamps.dt.tz_convert("Europe/Rome"), schedule.dt.tz_convert("Europe/Rome")
    flight = population["FLIGHT_mvt"].fillna("").astype(str)
    out = pd.DataFrame({
        ID: population[ID].to_numpy(),
        "schedule_interval_s": interval,
        "signed_log_schedule_interval": np.sign(interval) * np.log1p(np.abs(interval)),
        "local_hour": (local.dt.hour + local.dt.minute / 60).to_numpy(dtype=float),
        "scheduled_local_hour": (scheduled_local.dt.hour + scheduled_local.dt.minute / 60).to_numpy(dtype=float),
        "date_shift_days": (local.dt.normalize() - scheduled_local.dt.normalize()).dt.days.to_numpy(dtype=float),
        "day_of_week": local.dt.dayofweek.to_numpy(dtype=float),
        "month": local.dt.month.to_numpy(dtype=float),
        "flight_missing": flight.eq("").to_numpy(dtype=float),
        "flight_doubled_prefix": flight.map(lambda s: float(len(s) >= 5 and s[1:3] == s[3:5])).to_numpy(),
        "flight_alpha_tail": flight.map(lambda s: float(bool(re.search(r"[A-Z]$", s)))).to_numpy(),
        "flight_prefix": flight.str[:3].where(flight.ne(""), "?").to_numpy(),
        "stand_prefix": population["STAND_mvt"].fillna("?").astype(str).str[:1].to_numpy(),
        "runway": population["RUNWAY_mvt"].fillna("?").astype(str).to_numpy(),
        "destination_region": population["ADES_mvt"].fillna("??").astype(str).str[:2].to_numpy(),
    })
    out = out.merge(turnaround(population, movements, margin_seconds), on=ID, how="left", validate="one_to_one")
    if np.isinf(out[NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("Infinite mixture feature.")
    return out[[ID, *FEATURE_NAMES]]


def label_modes(labels: np.ndarray, interval: np.ndarray, config: dict) -> np.ndarray:
    """0 = schedule fill, 1 = ordinary or other, 2 = one-day; fitting labels only."""
    tolerance = float(config["fill_tolerance_seconds"])
    low, high = (float(v) for v in config["mixture"]["one_day_label_interval_seconds"])
    return np.select([np.abs(labels - interval) <= tolerance, (labels >= low) & (labels <= high)], [0, 2], 1)
