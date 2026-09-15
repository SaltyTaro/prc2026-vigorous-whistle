"""Extreme-row rules change only their declared rows, in declared order, from fitting parameters."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TARGET, TIME
from prc2026_postprocess.rules import ONE_DAY_SECONDS, apply_rules, fit_parameters, row_context

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/postprocess_v1.json").read_text(encoding="utf-8"))


def rows():
    time = pd.to_datetime(["2025-03-01T10:00:00Z"] * 8)
    frame = pd.DataFrame({
        ID: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        TIME: time,
        "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "EGLL", "EGLL", "LIRF", "LTFM", "LIRF"],
        "AIRCRAFT_TYPE_mvt": [None, None, "A320", None, "B738", None, "A21N", None],
        "SCHED_TIME_UTC_mvt": time - pd.to_timedelta([50000, 20000, 50000, 50000, 1200, 900, 900, 90000], unit="s"),
        "AOBT_3_flt": [pd.NaT, pd.NaT, time[2] - pd.Timedelta(minutes=15), pd.NaT, time[4] + pd.Timedelta(minutes=30), pd.NaT, pd.NaT, pd.NaT],
        "LOBT_flt": [pd.NaT, pd.NaT, time[2] - pd.Timedelta(minutes=16), pd.NaT, time[4] - pd.Timedelta(minutes=5), pd.NaT, time[6] + pd.Timedelta(minutes=2), pd.NaT],
    })
    return frame


PARAMS = {"floor_seconds": -12.0, "one_day_share": 0.25, "one_day_taxi_median_seconds": 900.0,
          "schedule_interval_threshold_seconds": 43200.0}


def test_row_context_matches_frozen_clock_definition():
    context = row_context(rows())
    assert context["clock_available"].tolist() == [False, False, True, False, True, False, True, False]
    # AOBT is preferred; LOBT only when AOBT is missing; negative when the clock postdates takeoff.
    assert context["proxy_s"].tolist()[2] == 900.0 and context["proxy_s"].tolist()[4] == -1800.0
    assert context["proxy_s"].tolist()[6] == -120.0
    assert context["hedge_population"].tolist() == [True, True, False, False, False, True, False, True]
    assert context["schedule_interval_s"].tolist()[0] == 50000.0


def test_rules_change_only_declared_rows_in_declared_order():
    frame = rows()
    prediction = np.array([53000.0, 21000.0, 1000.0, 1200.0, -3000.0, -50.0, 800.0, 95000.0])
    fallback = np.array([1500.0, 1500.0, 1500.0, 1500.0, 700.0, -40.0, 650.0, 1500.0])
    values, counts = apply_rules(frame, prediction, fallback, PARAMS, ["route_negative_proxy", "hedge_long_schedule", "floor"])
    expected = prediction.copy()
    expected[4], expected[6] = 700.0, 650.0                       # clock present, negative proxy -> fallback
    expected[0] = 0.75 * 50000 + 0.25 * (ONE_DAY_SECONDS + 900)  # LIRF, no clock, no type, s > 43200
    expected[7] = 0.75 * 90000 + 0.25 * (ONE_DAY_SECONDS + 900)  # also above one day
    expected[5] = -12.0                                           # floor applied last
    np.testing.assert_allclose(values, expected)
    assert counts == {"route_negative_proxy": 2, "hedge_long_schedule": 2, "floor": 1, "changed_rows": 5}
    # Rows 1 (s below threshold), 2 (has clock), 3 (other airport) are untouched.
    assert values[1] == 21000.0 and values[2] == 1000.0 and values[3] == 1200.0
    only_floor, counts = apply_rules(frame, prediction, fallback, PARAMS, ["floor"])
    assert counts == {"floor": 2, "changed_rows": 2} and only_floor[4] == -12.0 and (only_floor >= prediction).all()


def test_floor_cannot_increase_squared_error_when_labels_respect_it():
    rng = np.random.default_rng(3)
    labels = rng.normal(900, 400, 5000).clip(-12, None)
    prediction = labels + rng.normal(0, 300, 5000)
    floored = np.maximum(prediction, -12.0)
    assert ((floored - labels) ** 2 <= (prediction - labels) ** 2).all()


def test_rules_reject_undeclared_names_and_wrong_order():
    frame = rows()
    prediction = np.ones(8)
    with pytest.raises(ValueError):
        apply_rules(frame, prediction, prediction, PARAMS, ["clip_top"])
    with pytest.raises(ValueError):
        apply_rules(frame, prediction, prediction, PARAMS, ["floor", "route_negative_proxy"])
    with pytest.raises(ValueError):
        apply_rules(frame, prediction[:-1], prediction, PARAMS, ["floor"])


def test_fit_parameters_come_from_fitting_labels_only():
    frame = rows()
    frame[TARGET] = [50003.0, 900.0, 950.0, 1100.0, 700.0, -12.0, 640.0, 86400.0 + 1000.0]
    params = fit_parameters(frame, CONFIG)
    assert params["floor_seconds"] == -12.0 and params["negative_fitting_labels"] == 1
    assert params["hedge_population_rows"] == 4 and params["long_schedule_group_rows"] == 2
    assert params["long_schedule_group_one_day_rows"] == 1 and params["long_schedule_group_schedule_fill_rows"] == 1
    assert params["one_day_share"] == 0.5 and params["one_day_taxi_median_seconds"] == 1000.0
    frame[TARGET] = [50003.0, 900.0, 950.0, 1100.0, 700.0, -12.0, 640.0, 90001.0]
    with pytest.raises(ValueError):
        fit_parameters(frame, CONFIG)  # no one-day row: parameters cannot be estimated


def test_declaration_is_the_fixed_four_candidate_set():
    assert not CONFIG["audit_used"] and "2025-12" not in CONFIG["refit_months"]
    assert CONFIG["rule_order"] == ["route_negative_proxy", "hedge_long_schedule", "floor"]
    assert set(CONFIG["candidates"]) == {"floor", "floor_hedge", "floor_route", "floor_route_hedge"}
    assert all(rules[-1] == "floor" for rules in CONFIG["candidates"].values())
    assert CONFIG["rules"]["hedge_long_schedule"]["schedule_interval_threshold_seconds"] == 43200
    assert CONFIG["rules"]["route_negative_proxy"]["threshold_seconds"] == 0
