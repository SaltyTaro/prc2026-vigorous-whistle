"""Counterfactual feature availability, scope and weighting boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TIME, TARGET, BLOCK, PHASE
from prc2026_nm_mask.features import (AIRPORT_INPUTS, MASKED_FIELDS, MASKED_TIMESTAMPS,
    airport_only_view, masked_row_features, masked_ground_features, eligible_route)
from prc2026_nm_mask.common import sample_weights, compose, configurations, validate_truth


def raw_fixture(n=80):
    raw = pd.DataFrame({name: pd.Series(["A"] * n, dtype="string") for name in AIRPORT_INPUTS})
    raw[ID] = np.arange(n, dtype=float) + 1000
    raw[TIME] = pd.date_range("2025-05-01T08:00:00Z", periods=n, freq="1min")
    raw["SCHED_TIME_UTC_mvt"] = raw[TIME] - pd.Timedelta(minutes=15)
    raw[PHASE] = np.where(np.arange(n) % 4 == 0, "ARR", "DEP")
    raw["ADEP_mvt"] = "LIRF"
    raw["ADES_mvt"] = "LIRF"
    raw["RUNWAY_mvt"] = "16L"
    raw["STAND_mvt"] = "201"
    raw["AIRCRAFT_TYPE_mvt"] = "B738"
    raw.loc[np.arange(n) % 3 == 0, "AIRCRAFT_TYPE_mvt"] = pd.NA
    raw["FLIGHT_mvt"] = "AAA123"
    raw["FLIGHT_RULE_mvt"] = "I"
    return raw


def test_nm_fields_and_outcomes_cannot_change_row_features():
    base = raw_fixture()
    adversarial = base.copy()
    for name in MASKED_FIELDS:
        adversarial[name] = base[TIME] + pd.Timedelta(days=20) if name in MASKED_TIMESTAMPS else "LEAKED_NM_VALUE"
    adversarial[TARGET] = np.arange(len(base)) * 1000000
    adversarial[BLOCK] = base[TIME] - pd.Timedelta(days=15)
    pd.testing.assert_frame_equal(masked_row_features(base), masked_row_features(adversarial), check_exact=True)
    view = airport_only_view(adversarial)
    assert view[MASKED_FIELDS].isna().all().all()
    assert TARGET not in view and BLOCK not in view


def test_nm_context_cannot_change_ground_features():
    base = raw_fixture()
    altered = base.copy()
    altered["EOBT_1_flt"] = base[TIME] - pd.Timedelta(minutes=2)
    altered["WK_TBL_CAT_flt"] = "SUPER"
    queries = base.loc[base[PHASE].eq("DEP"), ID]
    left = masked_ground_features(base, queries)
    pd.testing.assert_frame_equal(left, masked_ground_features(altered, queries), check_exact=True)
    assert left.filter(regex=r"gd_\d+_eobt_count_").eq(0).all().all()
    assert left["gd_0_schedule_count_30m"].max() > 0
    assert not left.service_wake_pair.astype(str).str.contains("SUPER").any()


def test_real_missing_record_offset_is_retained():
    raw = raw_fixture()
    features = masked_row_features(raw)
    dep = raw.loc[raw[PHASE].eq("DEP")].reset_index(drop=True)
    expected = np.where(dep.AIRCRAFT_TYPE_mvt.isna(), 900., 0.)
    np.testing.assert_array_equal(features.conditional_schedule_offset_s, expected)


@pytest.mark.parametrize("problem", ["december", "duplicate", "missing_id", "missing_time", "missing_input"])
def test_invalid_feature_population_rejected(problem):
    raw = raw_fixture()
    if problem == "december":
        raw.loc[0, TIME] = pd.Timestamp("2025-12-01T00:00:00Z")
    elif problem == "duplicate":
        raw.loc[1, ID] = raw.loc[0, ID]
    elif problem == "missing_id":
        raw.loc[0, ID] = np.nan
    elif problem == "missing_time":
        raw.loc[0, TIME] = pd.NaT
    else:
        raw = raw.drop(columns="STAND_mvt")
    with pytest.raises(ValueError):
        airport_only_view(raw)


def test_routing_distinguishes_origin_and_clock_missingness():
    routing = pd.DataFrame({"nm_missing": [1, 0, 1, 0], "nmclock_available": [0, 0, 1, 1]})
    np.testing.assert_array_equal(eligible_route(routing), [True, False, False, False])
    routing.loc[0, "nmclock_available"] = np.nan
    with pytest.raises(ValueError):
        eligible_route(routing)


def test_augmented_cohorts_have_equal_weight_and_preserve_control_total():
    natural = np.array([True] * 4 + [False] * 100)
    weights = sample_weights(natural, "augmented")
    assert weights[natural].sum() == pytest.approx(2.)
    assert weights[~natural].sum() == pytest.approx(2.)
    assert weights.sum() == pytest.approx(sample_weights(np.ones(4, dtype=bool), "natural").sum())


@pytest.mark.parametrize("natural,population", [([True, False], "natural"), ([True, True], "augmented"),
    ([False, False], "augmented"), ([True, False], "unknown")])
def test_invalid_weighting_rejected(natural, population):
    with pytest.raises(ValueError):
        sample_weights(np.array(natural), population)


@pytest.mark.parametrize("weight", [.5, 1.])
def test_blend_preserves_every_ineligible_value_exactly(weight):
    reference = np.array([-80000.123456789, 10., 1000000.12345, 200.])
    eligible = np.array([False, True, False, True])
    result = compose(reference, np.array([-100., 10000.]), eligible, weight)
    np.testing.assert_array_equal(result[~eligible], reference[~eligible])
    np.testing.assert_array_equal(result[eligible], reference[eligible] + weight * (np.array([-100., 10000.]) - reference[eligible]))


@pytest.mark.parametrize("weight", [0., .75, 1.1])
def test_undeclared_weight_rejected(weight):
    with pytest.raises(ValueError):
        compose([1., 2.], [3.], np.array([True, False]), weight)


@pytest.mark.parametrize("problem", ["unmatched_id", "december", "nonfinite"])
def test_truth_population_guard(problem):
    features = pd.DataFrame({ID: [1., 2.]})
    truth = pd.DataFrame({ID: [1., 2.], TARGET: [-50., 200000.], TIME: pd.to_datetime(["2025-05-01", "2025-05-02"], utc=True)})
    validate_truth(features, truth)
    if problem == "unmatched_id":
        truth.loc[0, ID] = 3.
    elif problem == "december":
        truth.loc[0, TIME] = pd.Timestamp("2025-12-01T00:00:00Z")
    else:
        truth.loc[0, TARGET] = np.inf
    with pytest.raises(ValueError):
        validate_truth(features, truth)


@pytest.mark.parametrize("key,value", [("audit_used", True), ("automatic_submission", True),
    ("candidate_weights", [.25, .5, 1.]), ("minimum_improvement_fraction_each_fold", 0.),
    ("target_filter", "normal_only"), ("native_fit_count", 9)])
def test_declaration_guards(tmp_path, key, value):
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for name in ["nm_mask_v1.json", "features_nm_mask_v1.json"]:
        data = json.loads((root / "configs" / name).read_text())
        if name == "nm_mask_v1.json":
            data[key] = value
        (tmp_path / "configs" / name).write_text(json.dumps(data))
    with pytest.raises(ValueError):
        configurations(tmp_path)
