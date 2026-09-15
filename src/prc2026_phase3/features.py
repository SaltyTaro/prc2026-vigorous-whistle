"""Label-independent departure features and explicitly timed arrival context."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import (BLOCK, CATEGORY_SOURCES, FEATURE_NAMES, ID, PHASE,
                              TARGET, TIME, TIMEZONES, departure_features)

MISSING = "__MISSING__"
POLICY = "airport-timing-past-traffic-v1"
# Projection at the parquet reader prevents departure labels/actual off-block
# clocks from entering the row-feature transformation at all.
INPUT_COLUMNS = list(dict.fromkeys([
    ID, PHASE, TIME, "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "ARVT_1_flt",
    *CATEGORY_SOURCES.values(), "ADES_FILED_flt", "FLIGHT_RULE_flt", "FLIGHT_mvt",
]))
CONTEXT_COLUMNS = [ID, PHASE, TIME, "airport", "RUNWAY_mvt", "STAND_mvt",
                   "SCHED_TIME_UTC_mvt", "arrival_taxi_in_s", "arrival_block_time_utc"]

LAYOUT_CATEGORIES = [
    "airport_runway", "airport_stand", "airport_stand_runway", "airport_stand_area",
    "airport_area_runway", "runway_side", "stand_suffix", "airport_operator",
    "airport_aircraft", "airport_destination", "airport_wake", "flight_prefix",
    "airport_flight_prefix",
]
LAYOUT_NUMERIC = ["stand_number", "stand_hundreds", "stand_tens", "runway_number",
                  "runway_heading_sin", "runway_heading_cos"]
TIMING_NUMERIC = [
    "local_minute_of_day", "local_time_sin", "local_time_cos", "local_weekday_sin",
    "local_weekday_cos", "local_weekend", "utc_offset_hours", "day_of_year_sin",
    "day_of_year_cos", "movement_second", "scheduled_utc_minute", "scheduled_weekday",
    "eobt_utc_minute", "eobt_weekday", "iobt_utc_minute", "iobt_weekday",
    "eobt_minus_schedule_s", "iobt_minus_schedule_s", "eobt_minus_iobt_s",
    "planned_duration_iobt_s", "movement_minus_arvt1_s", "schedule_missing",
    "eobt_missing", "iobt_missing", "arvt1_missing", "nm_destination_agrees",
    "nm_aircraft_agrees", "nm_rule_agrees", "nm_filed_destination_agrees",
]


def history_names(prefix: str, windows: tuple[int, ...], values: bool = False) -> list[str]:
    names = [f"{prefix}_last_s"]
    for window in windows:
        names.append(f"{prefix}_count_{window}m")
        if values:
            names.extend([f"{prefix}_mean_{window}m", f"{prefix}_std_{window}m"])
    return names


TRAFFIC_NUMERIC = (
    history_names("airport_dep", (5, 15, 30, 60))
    + history_names("airport_arr", (5, 15, 30, 60))
    + history_names("runway_dep", (15, 60))
    + history_names("runway_arr", (15, 60))
    + history_names("stand_arr", (60,))
    + history_names("dep_schedule_delay", (30, 60), True)
    + history_names("arr_schedule_delay", (30, 60), True)
)
ARRIVAL_NUMERIC = (history_names("completed_arrival_taxi", (30, 60), True)
                   + history_names("runway_completed_arrival_taxi", (60,), True))
FAMILIES = {
    "layout": {"categorical": LAYOUT_CATEGORIES, "numeric": LAYOUT_NUMERIC},
    "timing": {"categorical": [], "numeric": TIMING_NUMERIC},
    "traffic": {"categorical": [], "numeric": TRAFFIC_NUMERIC},
    "arrivals": {"categorical": [], "numeric": ARRIVAL_NUMERIC},
}
ALL_FEATURE_NAMES = FEATURE_NAMES + [n for spec in FAMILIES.values()
                                    for n in spec["categorical"] + spec["numeric"]]


def feature_contract(families: list[str]) -> tuple[list[str], list[str]]:
    if len(families) != len(set(families)) or set(families) - set(FAMILIES):
        raise ValueError("Unknown or repeated feature family.")
    names, categories = list(FEATURE_NAMES), list(CATEGORY_SOURCES)
    for family in families:
        names.extend(FAMILIES[family]["categorical"] + FAMILIES[family]["numeric"])
        categories.extend(FAMILIES[family]["categorical"])
    return names, categories


def clean_codes(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.upper().replace(
        {"": MISSING, "NA": MISSING, "N/A": MISSING}).fillna(MISSING)


def row_features(raw: pd.DataFrame) -> pd.DataFrame:
    if set(INPUT_COLUMNS) - set(raw):
        raise ValueError("Missing declared row-feature inputs.")
    # Ignore extra columns defensively as well as projecting them at ingestion.
    view = raw[INPUT_COLUMNS].copy()
    view[TARGET] = np.nan
    view[BLOCK] = pd.Series(pd.NaT, index=view.index, dtype="datetime64[us, UTC]")
    out = departure_features(view)
    dep = view.loc[view[PHASE].eq("DEP")].reset_index(drop=True)
    airport = clean_codes(dep["ADEP_mvt"])
    runway, stand = clean_codes(dep["RUNWAY_mvt"]), clean_codes(dep["STAND_mvt"])
    number = pd.to_numeric(stand.str.extract(r"(\d+)", expand=False), errors="coerce")
    prefix = stand.str.extract(r"^([A-Z]+)", expand=False)
    area = prefix.fillna("#" + (number // 100).astype("Int64").astype("string")).fillna(MISSING)
    runway_parts = runway.str.extract(r"^(\d{1,2})([LRC]?)$")
    runway_number = pd.to_numeric(runway_parts[0], errors="coerce")
    runway_number = runway_number.where(runway_number.between(1, 36))
    flight_prefix = clean_codes(dep["FLIGHT_mvt"]).str.extract(r"^([A-Z]{2,3})", expand=False).fillna(MISSING)
    out["airport_runway"] = airport + "|" + runway
    out["airport_stand"] = airport + "|" + stand
    out["airport_stand_runway"] = airport + "|" + stand + "|" + runway
    out["airport_stand_area"] = airport + "|" + area
    out["airport_area_runway"] = airport + "|" + area + "|" + runway
    out["runway_side"] = runway_parts[1].fillna(MISSING)
    out["stand_suffix"] = stand.str.extract(r"\d([A-Z]+)$", expand=False).fillna(MISSING)
    for name, source in [("operator", "AIRCRAFT_OPERATOR_flt"), ("aircraft", "AIRCRAFT_TYPE_mvt"),
                         ("destination", "ADES_mvt"), ("wake", "WK_TBL_CAT_flt")]:
        out[f"airport_{name}"] = airport + "|" + clean_codes(dep[source])
    out["flight_prefix"] = flight_prefix
    out["airport_flight_prefix"] = airport + "|" + flight_prefix
    out["stand_number"], out["stand_hundreds"], out["stand_tens"] = number, number // 100, number // 10
    out["runway_number"] = runway_number
    out["runway_heading_sin"] = np.sin(runway_number.astype(float) * np.pi / 18)
    out["runway_heading_cos"] = np.cos(runway_number.astype(float) * np.pi / 18)

    time = dep[TIME]
    local_minute, local_weekday, offset = (np.full(len(dep), np.nan) for _ in range(3))
    for code, indexes in dep.groupby("ADEP_mvt", sort=False).groups.items():
        local = time.loc[indexes].dt.tz_convert(TIMEZONES[code])
        local_minute[indexes] = local.dt.hour * 60 + local.dt.minute + local.dt.second / 60
        local_weekday[indexes] = local.dt.dayofweek
        offset[indexes] = ((local.dt.tz_localize(None) - time.loc[indexes].dt.tz_localize(None))
                           .dt.total_seconds() / 3600)
    out["local_minute_of_day"] = local_minute
    out["local_time_sin"], out["local_time_cos"] = np.sin(local_minute * 2*np.pi/1440), np.cos(local_minute * 2*np.pi/1440)
    out["local_weekday_sin"], out["local_weekday_cos"] = np.sin(local_weekday * 2*np.pi/7), np.cos(local_weekday * 2*np.pi/7)
    out["local_weekend"], out["utc_offset_hours"] = (local_weekday >= 5).astype(float), offset
    days_in_year = np.where(time.dt.is_leap_year, 366, 365)
    angle = (time.dt.dayofyear - 1) * 2*np.pi/days_in_year
    out["day_of_year_sin"], out["day_of_year_cos"] = np.sin(angle), np.cos(angle)
    out["movement_second"] = time.dt.second
    for name, source in [("scheduled", "SCHED_TIME_UTC_mvt"), ("eobt", "EOBT_1_flt"), ("iobt", "IOBT_flt")]:
        out[f"{name}_utc_minute"] = dep[source].dt.hour * 60 + dep[source].dt.minute
        out[f"{name}_weekday"] = dep[source].dt.dayofweek
    for name, left, right in [
        ("eobt_minus_schedule_s", "EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
        ("iobt_minus_schedule_s", "IOBT_flt", "SCHED_TIME_UTC_mvt"),
        ("eobt_minus_iobt_s", "EOBT_1_flt", "IOBT_flt"),
        ("planned_duration_iobt_s", "ARVT_1_flt", "IOBT_flt"),
        ("movement_minus_arvt1_s", TIME, "ARVT_1_flt"),
    ]:
        out[name] = (dep[left] - dep[right]).dt.total_seconds()
    for name, source in [("schedule", "SCHED_TIME_UTC_mvt"), ("eobt", "EOBT_1_flt"),
                         ("iobt", "IOBT_flt"), ("arvt1", "ARVT_1_flt")]:
        out[f"{name}_missing"] = dep[source].isna().astype(float)
    for name, left, right in [("destination", "ADES_flt", "ADES_mvt"),
                             ("aircraft", "AIRCRAFT_TYPE_flt", "AIRCRAFT_TYPE_mvt"),
                             ("rule", "FLIGHT_RULE_flt", "FLIGHT_RULE_mvt"),
                             ("filed_destination", "ADES_FILED_flt", "ADES_mvt")]:
        out[f"nm_{name}_agrees"] = dep[left].eq(dep[right]).fillna(False).astype(float)
    for name in LAYOUT_CATEGORIES:
        out[name] = out[name].astype("string").fillna(MISSING)
    for name in LAYOUT_NUMERIC + TIMING_NUMERIC:
        out[name] = out[name].astype("float32")
    names, _ = feature_contract(["layout", "timing"])
    return out[[ID, *names]]


def _seconds(times: pd.Series) -> np.ndarray:
    if times.isna().any() or str(times.dt.tz) != "UTC":
        raise ValueError("History timestamps must be nonnull UTC.")
    return times.dt.as_unit("us").astype("int64").to_numpy() / 1_000_000


def past_history(events: pd.DataFrame, queries: pd.DataFrame, keys: list[str], prefix: str,
                 windows: tuple[int, ...], value: str | None = None) -> pd.DataFrame:
    """Windows are [t-window, t); all equal-time events and future events are excluded.

    Last-event gaps above six hours are missing. Statistics use finite event
    values only; counts still include all observed events. No target joins.
    """
    names = history_names(prefix, windows, value is not None)
    arrays = {name: np.full(len(queries), 0.0 if "_count_" in name else np.nan,
                            dtype="float32") for name in names}
    q = queries.reset_index(drop=True)
    _seconds(q[TIME])
    event_groups = {k: g.sort_values(TIME, kind="stable")
                    for k, g in events.groupby(keys, sort=False, dropna=False)}
    # Always use list keys on both sides, including a one-element key list.
    for key, group in q.groupby(keys, sort=False, dropna=False):
        event = event_groups.get(key)
        if event is None or event.empty:
            continue
        index = group.index.to_numpy()
        times, query_times = _seconds(event[TIME]), _seconds(group[TIME])
        hi = np.searchsorted(times, query_times, side="left")
        last = query_times - times[np.maximum(hi-1, 0)]
        last[(hi == 0) | (last > 6*3600)] = np.nan
        arrays[f"{prefix}_last_s"][index] = last
        if value is not None:
            values = event[value].to_numpy(dtype=float, na_value=np.nan)
            valid = np.isfinite(values)
            total = np.r_[0.0, np.cumsum(np.where(valid, values, 0.0))]
            squared = np.r_[0.0, np.cumsum(np.where(valid, values**2, 0.0))]
            count = np.r_[0, np.cumsum(valid)]
        for window in windows:
            lo = np.searchsorted(times, query_times-window*60, side="left")
            arrays[f"{prefix}_count_{window}m"][index] = hi-lo
            if value is not None:
                n = count[hi]-count[lo]
                mean = np.divide(total[hi]-total[lo], n, out=np.full(len(n), np.nan), where=n > 0)
                second = np.divide(squared[hi]-squared[lo], n, out=np.full(len(n), np.nan), where=n > 0)
                arrays[f"{prefix}_mean_{window}m"][index] = mean
                arrays[f"{prefix}_std_{window}m"][index] = np.sqrt(np.maximum(0, second-mean**2))
    return pd.DataFrame(arrays)


def traffic_features(context: pd.DataFrame, departures: pd.DataFrame) -> pd.DataFrame:
    if set(CONTEXT_COLUMNS) - set(context):
        raise ValueError("Missing declared context inputs.")
    if context[ID].isna().any() or context[ID].duplicated().any():
        raise ValueError("Context IDs must be unique and nonnull.")
    if not context[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Unknown context movement phase.")
    is_dep = context[PHASE].eq("DEP")
    if context.loc[is_dep, ["arrival_taxi_in_s", "arrival_block_time_utc"]].notna().any().any():
        raise ValueError("Departure labels must be masked in arrival context.")
    q = departures[[ID, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]].rename(
        columns={"ADEP_mvt": "airport"}).reset_index(drop=True)
    if q[ID].duplicated().any() or q[ID].isna().any():
        raise ValueError("Query IDs must be unique and nonnull.")
    event = context[CONTEXT_COLUMNS].copy()
    for name in ["airport", "RUNWAY_mvt", "STAND_mvt"]:
        event[name], q[name] = clean_codes(event[name]), clean_codes(q[name])
    event["schedule_delay_s"] = (event[TIME]-event["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    dep, arr = event.loc[is_dep], event.loc[~is_dep]
    frames = [q[[ID]]]
    for data, keys, prefix, windows, value in [
        (dep, ["airport"], "airport_dep", (5, 15, 30, 60), None),
        (arr, ["airport"], "airport_arr", (5, 15, 30, 60), None),
        (dep, ["airport", "RUNWAY_mvt"], "runway_dep", (15, 60), None),
        (arr, ["airport", "RUNWAY_mvt"], "runway_arr", (15, 60), None),
        (arr, ["airport", "STAND_mvt"], "stand_arr", (60,), None),
        (dep, ["airport"], "dep_schedule_delay", (30, 60), "schedule_delay_s"),
        (arr, ["airport"], "arr_schedule_delay", (30, 60), "schedule_delay_s"),
    ]:
        frames.append(past_history(data, q, keys, prefix, windows, value))
    # Landing information is insufficient to reveal completed taxi-in duration.
    # Make this channel available only at the recorded arrival in-block time.
    completed = arr.loc[arr["arrival_block_time_utc"].ge(arr[TIME])
                        & np.isfinite(arr["arrival_taxi_in_s"])].copy()
    completed[TIME] = completed["arrival_block_time_utc"]
    frames.append(past_history(completed, q, ["airport"], "completed_arrival_taxi", (30, 60), "arrival_taxi_in_s"))
    frames.append(past_history(completed, q, ["airport", "RUNWAY_mvt"],
                               "runway_completed_arrival_taxi", (60,), "arrival_taxi_in_s"))
    out = pd.concat(frames, axis=1)
    return out[[ID, *TRAFFIC_NUMERIC, *ARRIVAL_NUMERIC]]
