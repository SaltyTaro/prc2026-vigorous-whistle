"""Explicit feature contract shared by training and ranking transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
BLOCK = "BLOCK_TIME_UTC_mvt"
TIME = "MVT_TIME_UTC_mvt"
PHASE = "PHASE_mvt"
POLICY = "documented-core-v1"
TIMEZONES = {
    "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London",
    "EHAM": "Europe/Amsterdam", "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid",
    "LFPG": "Europe/Paris", "LIRF": "Europe/Rome", "LSZH": "Europe/Zurich",
    "LTFM": "Europe/Istanbul",
}
CATEGORY_SOURCES = {
    "airport": "ADEP_mvt", "destination": "ADES_mvt", "runway": "RUNWAY_mvt",
    "stand": "STAND_mvt", "aircraft_type": "AIRCRAFT_TYPE_mvt",
    "movement_rule": "FLIGHT_RULE_mvt", "nm_origin": "ADEP_flt",
    "nm_destination": "ADES_flt", "nm_aircraft_type": "AIRCRAFT_TYPE_flt",
    "market_segment": "MARKET_SEGMENT_flt", "flight_type": "FLIGHT_TYPE_flt",
    "wake_category": "WK_TBL_CAT_flt", "operator": "AIRCRAFT_OPERATOR_flt",
}
# IDs are alignment/grouping keys, never predictors. Actual NM clocks are
# deliberately outside this initial contract while organizer clarifications
# are checked; this choice is not a claim that the published rules ban them.
EXCLUDED_PREDICTORS = {
    ID, "FLIGHT_ID_mvt", TARGET, BLOCK, "AOBT_3_flt", "LOBT_flt", "ARVT_3_flt",
}
NUMERIC_FEATURES = [
    "utc_hour", "utc_weekday", "month", "minute_of_day", "local_hour",
    "local_weekday", "schedule_to_movement_s", "movement_minus_eobt1_s",
    "movement_minus_iobt_s", "planned_duration_s", "nm_missing",
    "nm_origin_agrees",
]
FEATURE_NAMES = list(CATEGORY_SOURCES) + NUMERIC_FEATURES


def validate_movements(frame: pd.DataFrame) -> None:
    required = {ID, "FLIGHT_ID_mvt", PHASE, TIME, BLOCK, TARGET, "SCHED_TIME_UTC_mvt",
                "EOBT_1_flt", "IOBT_flt", "ARVT_1_flt", *CATEGORY_SOURCES.values()}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    ids = frame[ID].to_numpy(dtype=float)
    if not np.isfinite(ids).all() or (ids % 1 != 0).any() or frame[ID].duplicated().any():
        raise ValueError("Movement IDs must be finite, integral, nonnull, and unique.")
    if not frame[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Unknown movement phase.")
    if frame[TIME].isna().any() or str(frame[TIME].dt.tz) != "UTC":
        raise ValueError("Movement timestamps must be nonnull and UTC-aware.")
    airport = frame["ADEP_mvt"].where(frame[PHASE] == "DEP", frame["ADES_mvt"])
    if airport.isna().any() or not airport.isin(TIMEZONES).all():
        raise ValueError("Unknown reporting airport; update the explicit airport/timezone contract.")


def ranking_view(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove every departure's label and airport off-block time before features."""
    view = frame.copy()
    departures = view[PHASE].eq("DEP")
    view.loc[departures, TARGET] = np.nan
    view.loc[departures, BLOCK] = pd.NaT
    return view


def departure_features(view: pd.DataFrame) -> pd.DataFrame:
    dep = view.loc[view[PHASE].eq("DEP")]
    if dep[TARGET].notna().any() or dep[BLOCK].notna().any():
        raise ValueError("Departure labels and block times must be masked before feature generation.")
    out = pd.DataFrame({ID: dep[ID].to_numpy()}, index=dep.index)
    for name, source in CATEGORY_SOURCES.items():
        out[name] = dep[source].astype("string").fillna("__MISSING__")
    time = dep[TIME]
    out["utc_hour"] = time.dt.hour
    out["utc_weekday"] = time.dt.dayofweek
    out["month"] = time.dt.month
    out["minute_of_day"] = time.dt.hour * 60 + time.dt.minute
    out["local_hour"] = np.nan
    out["local_weekday"] = np.nan
    for airport, indexes in dep.groupby("ADEP_mvt").groups.items():
        if airport not in TIMEZONES:
            raise ValueError(f"No timezone configured for {airport}.")
        local = time.loc[indexes].dt.tz_convert(TIMEZONES[airport])
        out.loc[indexes, "local_hour"] = local.dt.hour
        out.loc[indexes, "local_weekday"] = local.dt.dayofweek
    out["schedule_to_movement_s"] = (time - dep["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    out["movement_minus_eobt1_s"] = (time - dep["EOBT_1_flt"]).dt.total_seconds()
    out["movement_minus_iobt_s"] = (time - dep["IOBT_flt"]).dt.total_seconds()
    out["planned_duration_s"] = (dep["ARVT_1_flt"] - dep["EOBT_1_flt"]).dt.total_seconds()
    out["nm_missing"] = dep["ADEP_flt"].isna().astype(int)
    out["nm_origin_agrees"] = dep["ADEP_flt"].eq(dep["ADEP_mvt"]).fillna(False).astype(int)
    for name in NUMERIC_FEATURES:
        out[name] = out[name].astype("float32")
    if np.isinf(out[NUMERIC_FEATURES].to_numpy()).any():
        raise ValueError("Infinite numeric feature.")
    return out[[ID, *FEATURE_NAMES]].reset_index(drop=True)


def movement_context(view: pd.DataFrame) -> pd.DataFrame:
    """Persist arrival context with explicit names and no departure target channel."""
    dep = view[PHASE].eq("DEP")
    if view.loc[dep, TARGET].notna().any() or view.loc[dep, BLOCK].notna().any():
        raise ValueError("Unmasked departure targets in movement context.")
    columns = [ID, PHASE, "ADEP_mvt", "ADES_mvt", TIME, "SCHED_TIME_UTC_mvt",
               "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
    out = view[columns].copy()
    out["airport"] = view["ADEP_mvt"].where(dep, view["ADES_mvt"])
    out["arrival_taxi_in_s"] = view[TARGET].where(~dep).astype("float32")
    out["arrival_block_time_utc"] = view[BLOCK].where(~dep)
    return out.reset_index(drop=True)
