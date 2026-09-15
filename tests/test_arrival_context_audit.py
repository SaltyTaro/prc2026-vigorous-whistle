"""Availability boundaries for the independent supplied-arrival signal audit."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_arrival_context import context, direct_check
from prc2026.features import ID, TIME, TARGET
from prc2026_phase3.features import MISSING


def frames():
    query = pd.DataFrame({ID: [10.0], "FLIGHT_ID_mvt": [100.0], TIME: pd.to_datetime(["2025-02-01 12:00"], utc=True),
                          "airport": ["LFPG"], "runway": ["09R"], "stand": ["A1"]})
    arrivals = pd.DataFrame({ID: [1., 2., 3., 4., 5.], "FLIGHT_ID_mvt": [100., 200., 300., 400., 500.],
                             "completion": pd.to_datetime(["2025-02-01 11:40", "2025-02-01 12:10", "2025-02-01 12:00", "2025-02-01 11:00", "2025-02-01 11:40"], utc=True),
                             "airport": ["LFPG"] * 5, "runway": ["09R"] * 5, "stand": ["A1"] * 5,
                             TARGET: [9000., 8000., 7000., 600., 1200.]})
    return query, arrivals


def test_arrivals_exclude_focal_flight_and_uncompleted_or_simultaneous_events():
    query, arrivals = frames()
    result = context(query, arrivals)
    assert result.iloc[0].arrival_airport_3600s_count == 2
    assert result.iloc[0].arrival_airport_3600s_raw_mean == 900
    assert direct_check(query, arrivals, result)["maximum_absolute_difference"] == 0


def test_arrival_anomalies_are_flagged_and_do_not_enter_normal_mean():
    query, arrivals = frames()
    arrivals = arrivals.iloc[[4]].copy()
    arrivals[TARGET] = -600
    result = context(query, arrivals).iloc[0]
    assert result.arrival_airport_3600s_raw_mean == -600
    assert result.arrival_airport_3600s_negative_fraction == 1
    assert np.isnan(result.arrival_airport_3600s_normal_mean)


def test_missing_stands_are_not_pooled_and_unknown_flight_id_is_retained():
    query, arrivals = frames()
    query["FLIGHT_ID_mvt"] = np.nan
    query["stand"] = MISSING
    arrivals["stand"] = MISSING
    result = context(query, arrivals).iloc[0]
    assert result.arrival_airport_3600s_count == 3
    assert result.arrival_stand_604800s_count == 0
    assert np.isnan(result.arrival_stand_604800s_normal_mean)
