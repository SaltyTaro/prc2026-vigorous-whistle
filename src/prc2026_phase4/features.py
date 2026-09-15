"""Input-only reporting-quality features and a conditional schedule offset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME
from prc2026_phase3.features import feature_contract as previous_contract

POLICY = "record-quality-conditional-schedule-v1"
PREVIOUS_NAMES, PREVIOUS_CATEGORIES = previous_contract(["timing", "traffic"])
INPUT_COLUMNS = [ID, PHASE, TIME, "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt",
                 "FLIGHT_mvt", "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "ARVT_1_flt"]
CATEGORIES = ["record_flight_prefix", "record_flight_shape"]
NUMERIC = [
    "aircraft_missing", "flight_missing", "flight_designator_length", "flight_prefix_length",
    "flight_suffix_length", "flight_digit_count", "flight_prefix_unique_letters",
    "flight_prefix_adjacent_repeats", "flight_prefix_tail_repeat", "flight_standard_shape",
    "schedule_second", "eobt_second", "iobt_second", "schedule_day_offset",
    "eobt_day_offset", "iobt_day_offset", "arvt1_day_offset", "schedule_delta_modday_s",
    "eobt_delta_modday_s", "iobt_delta_modday_s", "schedule_clock_delta_s", "nm_clock_count",
    "lirf_missing_record", "conditional_schedule_offset_s",
]
FEATURE_NAMES = PREVIOUS_NAMES + CATEGORIES + NUMERIC
CATEGORICAL_COLUMNS = PREVIOUS_CATEGORIES + CATEGORIES


def quality_features(raw: pd.DataFrame) -> pd.DataFrame:
    if set(INPUT_COLUMNS) - set(raw):
        raise ValueError("Missing declared phase-4 feature inputs.")
    dep = raw.loc[raw[PHASE].eq("DEP"), INPUT_COLUMNS].reset_index(drop=True)
    if dep[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
        raise ValueError("Reserved audit features cannot be materialized in Phase 4.")
    if dep[ID].isna().any() or dep[ID].duplicated().any():
        raise ValueError("Feature IDs must be unique and nonnull.")
    out = dep[[ID]].copy()
    flight = dep["FLIGHT_mvt"].astype("string").str.strip().str.upper().fillna("")
    prefix = flight.str.extract(r"^([A-Z]+)", expand=False).fillna("")
    suffix = flight.str.extract(r"\d([A-Z]+)$", expand=False).fillna("")
    out["record_flight_prefix"] = prefix.str[:3].replace("", "__MISSING__")
    # Shape retains only lengths, never the flight number or a full identifier.
    out["record_flight_shape"] = ("L" + prefix.str.len().astype("string") + "D"
                                  + flight.str.count(r"\d").astype("string") + "S"
                                  + suffix.str.len().astype("string"))
    out["aircraft_missing"] = dep["AIRCRAFT_TYPE_mvt"].isna()
    out["flight_missing"] = flight.eq("")
    out["flight_designator_length"] = flight.str.len()
    out["flight_prefix_length"] = prefix.str.len()
    out["flight_suffix_length"] = suffix.str.len()
    out["flight_digit_count"] = flight.str.count(r"\d")
    unique = {p: len(set(p)) for p in prefix.unique()}
    repeats = {p: sum(a == b for a, b in zip(p, p[1:])) for p in prefix.unique()}
    tail = {p: len(p) >= 4 and p[-2:] == p[-4:-2] for p in prefix.unique()}
    out["flight_prefix_unique_letters"] = prefix.map(unique)
    out["flight_prefix_adjacent_repeats"] = prefix.map(repeats)
    out["flight_prefix_tail_repeat"] = prefix.map(tail)
    out["flight_standard_shape"] = flight.str.fullmatch(r"[A-Z]{2,3}\d[A-Z0-9]{0,4}")
    movement = dep[TIME]
    for name, source in [("schedule", "SCHED_TIME_UTC_mvt"), ("eobt", "EOBT_1_flt"),
                         ("iobt", "IOBT_flt"), ("arvt1", "ARVT_1_flt")]:
        clock = dep[source]
        out[name + "_day_offset"] = (movement.dt.floor("D") - clock.dt.floor("D")).dt.total_seconds() / 86400
        if name != "arvt1":
            out[name + "_second"] = clock.dt.second
            out[name + "_delta_modday_s"] = (movement - clock).dt.total_seconds() % 86400
    schedule = dep["SCHED_TIME_UTC_mvt"]
    out["schedule_clock_delta_s"] = ((movement-movement.dt.floor("D"))
                                      - (schedule-schedule.dt.floor("D"))).dt.total_seconds()
    out["nm_clock_count"] = dep[["EOBT_1_flt", "IOBT_flt", "ARVT_1_flt"]].notna().sum(axis=1)
    gate = dep["ADEP_mvt"].eq("LIRF") & dep["ADEP_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()
    interval = (movement-schedule).dt.total_seconds().to_numpy(dtype=float)
    out["lirf_missing_record"] = gate
    out["conditional_schedule_offset_s"] = np.where(gate & np.isfinite(interval), interval, 0.0)
    for name in CATEGORIES:
        out[name] = out[name].astype("string").fillna("__MISSING__")
    for name in NUMERIC:
        out[name] = out[name].astype("float32")
    return out[[ID, *CATEGORIES, *NUMERIC]]


def route_mask(features: pd.DataFrame, route: str) -> np.ndarray:
    if route == "all":
        return np.ones(len(features), dtype=bool)
    if route == "lirf_missing_record":
        return features["lirf_missing_record"].eq(1).to_numpy()
    if route.startswith("airport:"):
        return features["airport"].eq(route.split(":", 1)[1]).to_numpy()
    raise ValueError("Unknown specialist routing rule.")


def prediction_offset(features: pd.DataFrame, policy: str) -> np.ndarray:
    if policy == "none":
        return np.zeros(len(features))
    if policy == "lirf_missing_schedule":
        values = features["conditional_schedule_offset_s"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Prediction offsets must be finite.")
        return values
    raise ValueError("Unknown conditional offset policy.")
