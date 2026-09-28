"""Fill pin (configs/fill_pin_v1.json): LIRF departures whose recorded taxi time can only be a schedule fill or a one-day date error.

Label-free rule on any movement file with the ranking schema. Two scopes, both LIRF departures with a schedule interval
(takeoff minus scheduled departure) above 43,200 s:
- population: no NM flight record (ADEP_flt missing) and no airport aircraft type, the v9 mixture's population, on which the frozen v7
  keeps a learned partial one-day hedge (prediction = interval + 2.8..9.1 ks on the ranking file);
- clock_day: an NM off-block clock that is one day early (takeoff minus AOBT_3 within 7,200 s of 86,400 s), where the NM residual
  models follow the clock (prediction ~ 86,400 s) although the recorded label is either the interval (the 2025 precedent) or a
  one-day error (86,400 s plus an ordinary taxi), both above it.
Pinned prediction = the schedule interval in seconds. Every other row keeps its layer-set value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME

AIRPORT = "LIRF"
INTERVAL_MIN_S = 43200.0
CLOCK_DAY_S, CLOCK_DAY_TOLERANCE_S = 86400.0, 7200.0
SCOPES = ["population", "clock_day"]
COLUMNS = [ID, PHASE, "ADEP_mvt", TIME, "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"]


def _seconds(a: pd.Series, b: pd.Series) -> np.ndarray:
    return (pd.to_datetime(a, utc=True) - pd.to_datetime(b, utc=True)).dt.total_seconds().to_numpy(dtype=float)


def fill_pins(raw: pd.DataFrame) -> pd.DataFrame:
    """[ID, pin_s, scope, interval_s, proxy_s] for the in-scope departures of `raw` (RAW columns); empty when there are none."""
    missing = [c for c in COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"fill_pins needs columns {missing}")
    dep = raw.loc[raw[PHASE].eq("DEP") & raw["ADEP_mvt"].eq(AIRPORT), COLUMNS].reset_index(drop=True)
    interval = _seconds(dep[TIME], dep["SCHED_TIME_UTC_mvt"])
    proxy = _seconds(dep[TIME], dep["AOBT_3_flt"])
    long_interval = np.isfinite(interval) & (interval > INTERVAL_MIN_S)
    population = dep["ADEP_flt"].isna().to_numpy() & dep["AIRCRAFT_TYPE_mvt"].isna().to_numpy()
    clock_day = np.isfinite(proxy) & (np.abs(proxy - CLOCK_DAY_S) <= CLOCK_DAY_TOLERANCE_S)
    scope = np.where(long_interval & population, SCOPES[0], np.where(long_interval & clock_day, SCOPES[1], ""))
    keep = scope != ""
    out = pd.DataFrame({ID: dep[ID].to_numpy()[keep], "pin_s": interval[keep], "scope": scope[keep], "interval_s": interval[keep], "proxy_s": proxy[keep]})
    if out[ID].duplicated().any():
        raise ValueError("Duplicate departure IDs in the fill-pin scope.")
    return out


def candidate_pins(pins: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    """A declared candidate's rows: its `scopes`, with population rows restricted to `population_interval_bands_s` (closed intervals in
    seconds) or, in configs/fill_pin_v1.json's form, to intervals at or above `population_interval_s_min`."""
    if len(pins) == 0:
        return pins
    bands = candidate.get("population_interval_bands_s") or [[float(candidate["population_interval_s_min"]), float("inf")]]
    interval = pins["interval_s"].to_numpy(dtype=float)
    in_band = np.zeros(len(pins), dtype=bool)
    for lo, hi in bands:
        in_band |= (interval >= float(lo)) & (interval <= float(hi))
    keep = pins["scope"].isin(candidate["scopes"]).to_numpy() & (~pins["scope"].eq(SCOPES[0]).to_numpy() | in_band)
    return pins.loc[keep].reset_index(drop=True)


def apply_pins(values: np.ndarray, ids: np.ndarray, pins: pd.DataFrame, airports: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`values` with the pinned rows replaced by their pin; returns (new values, pinned mask in `ids` order). Every pin must be one of `ids`."""
    values = np.asarray(values, dtype=float)
    ids = np.asarray(ids)
    if len(pins) == 0:
        return values.copy(), np.zeros(len(ids), dtype=bool)
    if not np.isin(pins[ID].to_numpy(), ids).all():
        raise ValueError("Every fill pin must be a departure of the file.")
    aligned = pd.DataFrame({ID: ids}).merge(pins[[ID, "pin_s"]], on=ID, how="left", validate="one_to_one")
    pin = aligned["pin_s"].to_numpy(dtype=float)
    pinned = np.isfinite(pin)
    if airports is not None and pinned.any() and not (np.asarray(airports)[pinned] == AIRPORT).all():
        raise ValueError("A fill pin lies outside LIRF.")
    return np.where(pinned, pin, values), pinned
