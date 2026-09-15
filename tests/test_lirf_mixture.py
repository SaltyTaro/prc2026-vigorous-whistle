"""Turnaround features read only supplied clocks; the mixture touches only declared rows."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_lirf_mixture.features import FEATURE_NAMES, classifier_features, label_modes, population_rows, turnaround
from prc2026_lirf_mixture.model import apply_candidate, band_values, fit_classifier, mixture_values, normal_component, predict_modes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/lirf_mixture_v1.json").read_text(encoding="utf-8"))


def movements():
    t = pd.Timestamp("2025-03-01T12:00:00Z")
    rows = [
        # id, phase, time, sched, adep, ades, type, aobt, lobt, stand, runway, flight, block, target
        (1.0, "DEP", t, t - pd.Timedelta(hours=3), "LIRF", "EGLL", None, pd.NaT, pd.NaT, "301", "25", "BAW551", pd.NaT, 10800.0),
        (2.0, "DEP", t, t - pd.Timedelta(minutes=40), "LIRF", "LEBL", None, pd.NaT, pd.NaT, "302", "25", "WMTMT4SG", pd.NaT, 900.0),
        (3.0, "DEP", t, t - pd.Timedelta(hours=14), "LIRF", "LLBG", None, pd.NaT, pd.NaT, "303", "25", "ISR342", pd.NaT, 87300.0),
        (4.0, "DEP", t, t - pd.Timedelta(hours=3), "LIRF", "EGLL", "A320", pd.NaT, pd.NaT, "301", "25", "BAW552", pd.NaT, 950.0),
        (5.0, "DEP", t, t - pd.Timedelta(hours=3), "EGLL", "LIRF", None, pd.NaT, pd.NaT, "5", "27R", "BAW553", pd.NaT, 950.0),
        (6.0, "DEP", t, t - pd.Timedelta(hours=3), "LIRF", "EGLL", None, t - pd.Timedelta(minutes=15), pd.NaT, "301", "25", "BAW554", pd.NaT, 900.0),
        (7.0, "ARR", t - pd.Timedelta(hours=2), t - pd.Timedelta(hours=2), "EGLL", "LIRF", "A320", pd.NaT, pd.NaT, "301", "16L", "BAW550", t - pd.Timedelta(hours=1, minutes=50), 600.0),
        (8.0, "ARR", t - pd.Timedelta(minutes=10), t - pd.Timedelta(minutes=10), "EGLL", "LIRF", "A320", pd.NaT, pd.NaT, "301", "16L", "BAW560", t - pd.Timedelta(minutes=5), 300.0),
        (9.0, "ARR", t - pd.Timedelta(hours=5), t - pd.Timedelta(hours=5), "LEBL", "LIRF", "A320", pd.NaT, pd.NaT, "302", "16L", "WMT1", t - pd.Timedelta(hours=4, minutes=50), 600.0),
    ]
    frame = pd.DataFrame(rows, columns=[ID, PHASE, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
                                        "AOBT_3_flt", "LOBT_flt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt", "BLOCK_TIME_UTC_mvt", TARGET])
    for name in [TIME, "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "BLOCK_TIME_UTC_mvt"]:
        frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame


def test_population_and_turnaround_use_only_supplied_clocks():
    frame = movements()
    population = population_rows(frame)
    assert population[ID].tolist() == [1.0, 2.0, 3.0]          # 4 has a type, 5 is not LIRF, 6 has an NM clock
    turn = turnaround(population, frame, 1200)
    # Stand 301: previous in-block 1h50 before takeoff (the arrival 5 minutes before takeoff is inside the margin -> 'next')
    row = turn.set_index(ID).loc[1.0]
    assert row["takeoff_minus_previous_inblock_s"] == 6600.0
    assert row["schedule_minus_previous_inblock_s"] == 6600.0 - 3 * 3600
    assert row["next_inblock_minus_takeoff_s"] == -300.0
    # Stand 303 has no arrivals -> missing features, row kept
    assert np.isnan(turn.set_index(ID).loc[3.0]["takeoff_minus_previous_inblock_s"])
    features = classifier_features(population, frame, 1200)
    assert list(features.columns) == [ID, *FEATURE_NAMES]
    assert features.set_index(ID).loc[2.0, "flight_doubled_prefix"] == 1.0
    assert features.set_index(ID).loc[1.0, "schedule_interval_s"] == 10800.0


def test_label_modes_follow_declared_definitions():
    labels = np.array([10800.0, 900.0, 87300.0, 10700.0, 82000.0])
    interval = np.array([10800.0, 2400.0, 50400.0, 10800.0, 50400.0])
    assert label_modes(labels, interval, CONFIG).tolist() == [0, 1, 2, 0, 1]


def test_mixture_changes_only_eligible_population_rows_and_floors():
    incumbent = np.array([5000.0, 2500.0, 60000.0, 1000.0, -100.0])
    index = np.array([0, 1, 2])
    interval = np.array([10800.0, 2400.0, 50400.0])
    mixture = np.array([9000.0, 1000.0, 50400.0])
    values, counts = apply_candidate(incumbent, index, interval, mixture, 0.75, 43200.0, -12.0)
    np.testing.assert_allclose(values, [0.25 * 5000 + 0.75 * 9000, 0.25 * 2500 + 0.75 * 1000, 60000.0, 1000.0, -12.0])
    assert counts == {"population_rows": 3, "mixture_rows": 2, "floored_rows": 1, "changed_rows": 3}


def test_normal_component_and_mixture_arithmetic():
    labels = np.array([800.0, 1200.0, 3000.0, 5000.0, 9000.0, 87000.0])
    interval = np.array([1000.0, 2000.0, 5000.0, 5000.0, 9000.0, 50000.0])
    modes = np.array([1, 1, 1, 0, 0, 2])
    component = normal_component(labels, interval, modes, CONFIG)
    assert component["band_medians"]["0"] == 800.0 and component["band_medians"]["1"] == 1200.0 and component["band_medians"]["2"] == 3000.0
    assert component["band_medians"]["5"] is None and component["overall_median"] == 1200.0
    np.testing.assert_allclose(band_values(component, np.array([500.0, 6000.0, 60000.0])), [800.0, 3000.0, 1200.0])
    probabilities = np.array([[0.6, 0.2, 0.2], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(mixture_values(probabilities, np.array([6000.0, 2000.0]), component), [0.75 * 6000 + 0.25 * 3000, 1200.0])


def test_classifier_is_deterministic_and_grouped_by_date():
    rng = np.random.default_rng(7)
    n = 400
    interval = rng.uniform(1000, 40000, n)
    turn = rng.normal(0, 5000, n)
    mode = np.where(turn < -2000, 0, 1)
    mode[:6] = 2
    dates = np.repeat(np.arange(40), 10).astype(str)
    features = pd.DataFrame({ID: np.arange(n, dtype=float), "schedule_interval_s": interval, "signed_log_schedule_interval": np.log1p(interval),
                             "local_hour": rng.uniform(0, 24, n), "scheduled_local_hour": rng.uniform(0, 24, n), "date_shift_days": 0.0,
                             "day_of_week": 1.0, "month": 3.0, "flight_missing": 0.0, "flight_doubled_prefix": 0.0, "flight_alpha_tail": 0.0,
                             "schedule_minus_previous_inblock_s": turn, "takeoff_minus_previous_inblock_s": turn + interval, "next_inblock_minus_takeoff_s": 600.0,
                             "flight_prefix": "RYR", "stand_prefix": "3", "runway": "25", "destination_region": "EG"})
    first = fit_classifier(features, mode, dates, CONFIG)
    second = fit_classifier(features, mode, dates, CONFIG)
    assert first["rounds"] == second["rounds"] and first["rounds"] >= 50
    np.testing.assert_array_equal(predict_modes(first, features), predict_modes(second, features))
    probabilities = predict_modes(first, features)
    assert probabilities.shape == (n, 3) and np.allclose(probabilities.sum(axis=1), 1)
    assert probabilities[mode == 0, 0].mean() > probabilities[mode == 1, 0].mean()
    with pytest.raises(ValueError):
        fit_classifier(features, np.full(n, 4), dates, CONFIG)


def test_declaration_is_fixed():
    assert not CONFIG["audit_used"] and "2025-12" not in CONFIG["refit_months"]
    assert set(CONFIG["candidates"]) == {"mixture_50", "mixture_75", "mixture_100"}
    assert CONFIG["minimum_improvement_fraction_each_fold"] == 0.005
    assert CONFIG["mixture"]["schedule_interval_threshold_seconds"] == 43200
    assert CONFIG["classifier"]["params"]["deterministic"] is True
