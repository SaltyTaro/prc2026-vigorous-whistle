import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from prc2026.features import (BLOCK, FEATURE_NAMES, ID, PHASE, TARGET, TIME,
                             departure_features, movement_context, ranking_view,
                             validate_movements)


def test_targets_and_actual_nm_clocks_cannot_change_core_features(movements):
    original = movements.copy(deep=True)
    changed = movements.copy(deep=True)
    dep = changed[PHASE].eq("DEP")
    changed.loc[dep, TARGET] = [999999, -500, 0]
    changed.loc[dep, BLOCK] = changed.loc[dep, TIME] + pd.Timedelta(days=40)
    for column in ["AOBT_3_flt", "LOBT_flt", "ARVT_3_flt"]:
        changed[column] += pd.Timedelta(days=100)
    assert_frame_equal(departure_features(ranking_view(movements)),
                       departure_features(ranking_view(changed)))
    assert_frame_equal(movements, original)


def test_masking_preserves_arrivals_and_context_has_no_departure_label_channel(movements):
    view = ranking_view(movements)
    context = movement_context(view)
    assert view.loc[2, TARGET] == 240
    assert view.loc[2, BLOCK] == movements.loc[2, BLOCK]
    dep = context[PHASE].eq("DEP")
    assert context.loc[dep, ["arrival_taxi_in_s", "arrival_block_time_utc"]].isna().all().all()
    assert context.loc[2, "airport"] == "EDDF"
    assert context.loc[2, "arrival_taxi_in_s"] == 240
    assert TARGET not in context and BLOCK not in context


@pytest.mark.parametrize("transform", [departure_features, movement_context])
def test_unmasked_departures_are_rejected(movements, transform):
    with pytest.raises(ValueError, match="mask"):
        transform(movements)


def test_local_clock_handles_dst_and_midnight(movements):
    validate_movements(movements)
    features = departure_features(ranking_view(movements)).set_index(ID)
    assert features.loc[101, "local_hour"] == 0
    assert features.loc[102, "local_hour"] == 2
    assert features.loc[104, "utc_weekday"] == 5
    assert features.loc[104, "local_weekday"] == 6
    assert features.loc[104, "local_hour"] == 1
    assert features.loc[104, "nm_missing"] == 1
    assert set(features) == set(FEATURE_NAMES)


def test_bad_reporting_airport_is_rejected(movements):
    movements.loc[2, "ADES_mvt"] = "XXXX"
    with pytest.raises(ValueError, match="reporting airport"):
        validate_movements(movements)


@pytest.mark.parametrize("value", [np.nan, 101.0, 2.5])
def test_invalid_ids_are_rejected(movements, value):
    movements.loc[1, ID] = value
    with pytest.raises(ValueError, match="Movement IDs"):
        validate_movements(movements)
