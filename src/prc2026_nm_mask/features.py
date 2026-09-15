"""Reconstruct an airport-only view without reading NM fields or departure labels."""
import numpy as np
import pandas as pd

from prc2026.features import ID, TIME, PHASE, TARGET, BLOCK
from prc2026_phase3.features import (INPUT_COLUMNS as ROW_INPUTS, row_features,
    feature_contract, TRAFFIC_NUMERIC)
from prc2026_phase4.features import INPUT_COLUMNS as QUALITY_INPUTS, quality_features
from prc2026_phase5r.features import INPUT_COLUMNS as GROUND_INPUTS, ground_features
from prc2026_weather.features import FEATURE_NAMES, CATEGORICAL_COLUMNS, WEATHER_NAMES

INPUT_COLUMNS = list(dict.fromkeys(ROW_INPUTS + QUALITY_INPUTS + GROUND_INPUTS))
AIRPORT_INPUTS = [name for name in INPUT_COLUMNS if name.endswith("_mvt")]
MASKED_FIELDS = [name for name in INPUT_COLUMNS if name.endswith("_flt")]
MASKED_TIMESTAMPS = {"EOBT_1_flt", "IOBT_flt", "ARVT_1_flt"}
UNCHANGED_NAMES = list(TRAFFIC_NUMERIC) + list(WEATHER_NAMES)
ROW_NAMES, _ = feature_contract(["timing"])


def airport_only_view(raw):
    """Extra columns are ignored; NM fields are created as missing from scratch."""
    if set(AIRPORT_INPUTS) - set(raw):
        raise ValueError("Missing declared airport inputs.")
    if TARGET in AIRPORT_INPUTS or BLOCK in AIRPORT_INPUTS:
        raise ValueError("Departure outcomes cannot enter the feature projection.")
    out = raw[AIRPORT_INPUTS].reset_index(drop=True).copy()
    if out[ID].isna().any() or out[ID].duplicated().any():
        raise ValueError("Movement IDs must be unique and nonnull.")
    if out[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
        raise ValueError("Reserved December movements are excluded.")
    if out[TIME].isna().any() or str(out[TIME].dt.tz) != "UTC":
        raise ValueError("Actual UTC movement dates are required.")
    for name in MASKED_FIELDS:
        out[name] = (pd.Series(pd.NaT, index=out.index, dtype="datetime64[us, UTC]")
                     if name in MASKED_TIMESTAMPS
                     else pd.Series(pd.NA, index=out.index, dtype="string"))
    return out[INPUT_COLUMNS]


def masked_row_features(raw):
    view = airport_only_view(raw)
    rows = row_features(view)[[ID, *ROW_NAMES]]
    quality = quality_features(view)
    if not np.array_equal(rows[ID], quality[ID]):
        raise ValueError("Masked row feature identities differ.")
    return pd.concat([rows, quality.drop(columns=ID)], axis=1)


def masked_ground_features(raw, query_ids):
    return ground_features(airport_only_view(raw), query_ids)


def eligible_route(routing):
    for name in ["nm_missing", "nmclock_available"]:
        if not routing[name].isin([0, 1]).all():
            raise ValueError("Routing flags must be complete binary values.")
    return (routing["nm_missing"].eq(1) & routing["nmclock_available"].eq(0)).to_numpy()


def offset(features):
    values = features["conditional_schedule_offset_s"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Every conditional offset must be finite.")
    return values
