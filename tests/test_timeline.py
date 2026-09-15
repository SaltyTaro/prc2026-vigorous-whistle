"""Independent event-boundary and protection checks for the timeline experiment."""
import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, PHASE, TIME
from prc2026_timeline.features import INPUT_COLUMNS, checked_context, flown_features, peer_history, timeline_features


def fixture():
    start = pd.Timestamp("2025-05-01T12:00:00Z")
    rows = []
    for ident, movement, offblock, phase in [(1, 30, 10, "DEP"), (2, 20, 0, "DEP"), (3, 40, 5, "DEP"), (4, 30, 10, "DEP"), (5, 31, 31, "DEP"), (6, 9, -60, "ARR"), (7, 10, -60, "ARR")]:
        row = {ID: float(ident), PHASE: phase, TIME: start+pd.Timedelta(minutes=movement), "ADEP_mvt": "EGLL", "ADES_mvt": "EGLL", "RUNWAY_mvt": "09L", "STAND_mvt": "A1"}
        row.update({name: start+pd.Timedelta(minutes=offblock) for name in INPUT_COLUMNS[7:]})
        row["ARVT_3_flt"] = start+pd.Timedelta(hours=2)
        row["ARVT_1_flt"] = start+pd.Timedelta(minutes=110)
        rows.append(row)
    return pd.DataFrame(rows)


def test_other_flight_events_include_prior_pushes_and_exclude_focal_and_ties():
    result = timeline_features(fixture(), pd.Series([1.]))
    assert result.surface_takeoff_airport_push_count_30m.iloc[0] == 3
    assert result.surface_takeoff_airport_proxy_count_15m.iloc[0] == 1
    assert result.surface_takeoff_airport_proxy_mean_15m.iloc[0] == 1200
    assert result.surface_takeoff_airport_active.iloc[0] == 2
    assert result.surface_offblock_airport_active.iloc[0] == 2
    assert result.surface_offblock_airport_push_count_15m.iloc[0] == 2
    assert result.surface_interval_airport_dep_count.iloc[0] == 1
    assert result.surface_interval_airport_arr_count.iloc[0] == 1
    assert result.surface_offblock_stand_arr_last_s.iloc[0] == 60


def test_departure_target_and_block_columns_cannot_affect_features():
    original = fixture()
    polluted = original.assign(TAXITIME_SEC_mvt=-987654321, BLOCK_TIME_UTC_mvt=pd.Timestamp("2025-05-01T00:00:00Z"))
    pd.testing.assert_frame_equal(timeline_features(original, pd.Series([1.])), timeline_features(polluted, pd.Series([1.])), check_exact=True)


def test_future_clock_events_and_peer_flown_arrivals_do_not_change_past_context():
    raw = fixture()
    original = timeline_features(raw, pd.Series([1.]))
    raw.loc[raw[ID].eq(5.), [TIME, "AOBT_3_flt", "LOBT_flt"]] += pd.Timedelta(hours=3)
    raw.loc[raw[ID].ne(1.), "ARVT_3_flt"] += pd.Timedelta(days=1)
    changed = timeline_features(raw, pd.Series([1.]))
    pd.testing.assert_frame_equal(original, changed, check_exact=True)


@pytest.mark.parametrize("offset_minutes", [-10, 500])
def test_invalid_nm_anchor_uses_declared_past_context_without_clipping_raw_clock(offset_minutes):
    raw = fixture()
    raw.loc[raw[ID].eq(1.), "AOBT_3_flt"] = raw.loc[raw[ID].eq(1.), TIME]-pd.Timedelta(minutes=offset_minutes)
    result = timeline_features(raw, pd.Series([1.]))
    assert result.surface_anchor_nm_used.iloc[0] == 0
    assert result.surface_anchor_lookback_s.iloc[0] == 1200
    assert result.flight_duration_aobt_s.iloc[0] == (90+offset_minutes)*60


def test_missing_nm_anchor_keeps_every_query():
    raw = fixture()
    raw.loc[raw[ID].eq(1.), ["AOBT_3_flt", "LOBT_flt", "ARVT_3_flt"]] = pd.NaT
    result = timeline_features(raw, pd.Series([3., 1.]))
    assert result[ID].tolist() == [3., 1.]
    assert result.flight_arrival_missing.tolist() == [0., 1.]
    assert result.surface_anchor_lookback_s.iloc[1] == 1200


def test_reserved_context_is_rejected_even_when_it_is_not_a_query():
    raw = fixture()
    raw.loc[raw[ID].eq(5.), TIME] = pd.Timestamp("2025-12-01T12:00:00Z")
    with pytest.raises(ValueError, match="Reserved"):
        timeline_features(raw, pd.Series([1.]))


def test_flown_clock_keeps_signed_and_large_intervals():
    raw = fixture().iloc[:2].copy()
    raw.loc[raw.index[0], "ARVT_3_flt"] = raw.loc[raw.index[0], TIME]-pd.Timedelta(seconds=1)
    raw.loc[raw.index[1], "ARVT_3_flt"] = raw.loc[raw.index[1], "AOBT_3_flt"]+pd.Timedelta(days=2)
    result = flown_features(checked_context(raw))
    assert result.flight_arrival_minus_movement_s.iloc[0] == -1
    assert result.flight_arrival_before_movement.iloc[0] == 1
    assert result.flight_duration_aobt_s.iloc[1] == 172800
    assert result.flight_duration_over24h.iloc[1] == 1


def test_peer_history_matches_independent_direct_scan_including_missing_values_and_self():
    rng = np.random.default_rng(913)
    start = pd.Timestamp("2025-05-01T12:00:00Z")
    events = pd.DataFrame({ID: np.arange(50, dtype=float), TIME: start+pd.to_timedelta(rng.integers(0, 3600, 50), unit="s"), "airport": "EGLL", "value": rng.normal(500, 300, 50)})
    events.loc[[2, 7], "value"] = np.nan
    queries = events.iloc[:20][[ID, TIME, "airport"]].copy().reset_index(drop=True)
    queries[TIME] += pd.to_timedelta(rng.integers(0, 1500, 20), unit="s")
    actual = peer_history(events, queries, ["airport"], "peer", (15, 60), "value")
    for i, row in queries.iterrows():
        past = events.loc[events[TIME].lt(row[TIME]) & events[ID].ne(row[ID])]
        for window in (15, 60):
            selected = past.loc[past[TIME].ge(row[TIME]-pd.Timedelta(minutes=window))]
            assert actual.loc[i, f"peer_count_{window}m"] == len(selected)
            np.testing.assert_allclose(actual.loc[i, f"peer_mean_{window}m"], selected.value.mean(), rtol=1e-6, atol=1e-4)
            np.testing.assert_allclose(actual.loc[i, f"peer_std_{window}m"], selected.value.std(ddof=0), rtol=1e-6, atol=1e-4)
