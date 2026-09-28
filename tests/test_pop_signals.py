"""configs/pop_signals_v1.json: previous-arrival features, callsign rates and the bagged classifier (src/prc2026_final/pop_signals.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prc2026.artifacts import read_json  # noqa: E402
from prc2026_final import pop_signals as ps  # noqa: E402
from prc2026_lirf_mixture.features import FEATURE_NAMES  # noqa: E402

T0 = pd.Timestamp("2025-07-10 12:00", tz="UTC")
H = pd.Timedelta(hours=1)


def movements() -> pd.DataFrame:
    """Four LIRF population departures (stands A, B, C, D) and the arrivals around them."""
    dep = pd.DataFrame({
        "MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "PHASE_mvt": ["DEP"] * 4, "ADEP_mvt": ["LIRF"] * 4, "ADES_mvt": ["EGLL", "LFPG", "EDDF", "LEMD"],
        "MVT_TIME_UTC_mvt": [T0] * 4, "SCHED_TIME_UTC_mvt": [T0 - 2 * H, T0 - 3 * H, T0 - H, T0 - 4 * H], "BLOCK_TIME_UTC_mvt": [pd.NaT] * 4,
        "AIRCRAFT_TYPE_mvt": [None] * 4, "AOBT_3_flt": [pd.NaT] * 4, "LOBT_flt": [pd.NaT] * 4, "ADEP_flt": [None] * 4,
        "STAND_mvt": ["A", "B", "C", "D"], "RUNWAY_mvt": ["25"] * 4, "FLIGHT_mvt": ["RYR123", "WMT45", "ITY9", None], "TAXITIME_SEC_mvt": [7200, 900, 1000, 14400]})
    arr = pd.DataFrame({
        "MVT_ID_mvt": [11.0, 12.0, 13.0, 14.0], "PHASE_mvt": ["ARR"] * 4, "ADEP_mvt": ["EGLL", "LFPG", "EDDF", "EGLL"], "ADES_mvt": ["LIRF"] * 4,
        "MVT_TIME_UTC_mvt": [T0 - 3 * H] * 4, "SCHED_TIME_UTC_mvt": [T0 - 3 * H, T0 - 5 * H, T0 - 2 * H, T0 - 6 * H],
        "BLOCK_TIME_UTC_mvt": [T0 - 3 * H + pd.Timedelta(seconds=30), T0 - 4 * H, T0 - 2 * H + 10 * pd.Timedelta(minutes=1), T0 - 30 * pd.Timedelta(minutes=1)],
        "AIRCRAFT_TYPE_mvt": ["B738", "A320", None, "A320"], "AOBT_3_flt": [pd.NaT] * 4, "LOBT_flt": [pd.NaT] * 4, "ADEP_flt": ["EGLL", "LFPG", None, "EGLL"],
        "STAND_mvt": ["A", "B", "C", "A"], "RUNWAY_mvt": ["16"] * 4, "FLIGHT_mvt": ["RYR122", "EZY77", "ITY8", "BAW1"], "TAXITIME_SEC_mvt": [500, 600, 700, 800]})
    return pd.concat([dep, arr], ignore_index=True)


def test_previous_arrival_identity_and_source():
    m = movements()
    pop = m[m["PHASE_mvt"].eq("DEP")].reset_index(drop=True)
    f = ps.previous_arrival_features(pop, m, 1200.0, 120.0).set_index("MVT_ID_mvt")
    # stand A: the latest in-block at or before takeoff - 20 min is RYR122 (BAW1's in-block, 30 min before takeoff, is also before the
    # margin: takeoff - 1200 s = T0 - 20 min, and T0 - 30 min < T0 - 20 min, so BAW1 is the previous arrival at A)
    assert f.at[1.0, "pa_same_prefix"] == 0.0 and f.at[1.0, "pa_fill"] == 0.0 and f.at[1.0, "pa_nonm"] == 0.0
    assert f.at[1.0, "pa_sched_gap"] == pytest.approx((T0 - 2 * H - (T0 - 6 * H)).total_seconds())
    # stand B: EZY77, different prefix, in-block one hour after its schedule (not a fill), NM record present
    assert f.at[2.0, "pa_same_prefix"] == 0.0 and f.at[2.0, "pa_fill"] == 0.0 and f.at[2.0, "pa_nonm"] == 0.0
    # stand C: ITY8 = same prefix as ITY9, in-block 10 min after schedule (not a fill at 120 s), no NM record
    assert f.at[3.0, "pa_same_prefix"] == 1.0 and f.at[3.0, "pa_fill"] == 0.0 and f.at[3.0, "pa_nonm"] == 1.0
    # stand D: no arrival -> every feature missing
    assert np.isnan(f.loc[4.0, ps.PA_NUMERIC].to_numpy(dtype=float)).all()
    # a fill arrival: RYR122 with in-block equal to its schedule within 120 s, after moving BAW1 out of the window
    m2 = m.copy()
    m2.loc[m2["FLIGHT_mvt"].eq("BAW1"), "BLOCK_TIME_UTC_mvt"] = T0 - 10 * pd.Timedelta(minutes=1)
    f2 = ps.previous_arrival_features(pop, m2, 1200.0, 120.0).set_index("MVT_ID_mvt")
    assert f2.at[1.0, "pa_same_prefix"] == 1.0 and f2.at[1.0, "pa_fill"] == 1.0


def test_callsign_rates_exclude_own_month_and_smooth():
    flights = pd.Series(["A", "A", "A", "B", "B", "C"])
    months = pd.Series(["m1", "m1", "m2", "m1", "m2", "m2"])
    fill = np.array([1, 1, 0, 0, 0, 1], dtype=float)  # overall share 0.5
    same = ps.callsign_rates(flights, months, fill, flights, months, smoothing=3.0)
    # row 0 (A in m1): other months of A = m2 with 0 fills of 1 row -> (0 + 1.5) / (1 + 3)
    assert same[0] == pytest.approx(1.5 / 4)
    # row 2 (A in m2): other months = m1 with 2 fills of 2 rows -> (2 + 1.5) / (2 + 3)
    assert same[2] == pytest.approx(3.5 / 5)
    # unseen callsign -> the prior
    unseen = ps.callsign_rates(flights, months, fill, pd.Series(["Z"]), None, smoothing=3.0)
    assert unseen[0] == pytest.approx(0.5)
    every = ps.callsign_rates(flights, months, fill, pd.Series(["A"]), None, smoothing=3.0)
    assert every[0] == pytest.approx((2 + 1.5) / (3 + 3))


def test_candidate_names_extend_v9s_features():
    assert ps.candidate_names({"feature_set": "base"}) == FEATURE_NAMES
    assert ps.candidate_names({"feature_set": "pa"}) == FEATURE_NAMES + ps.PA_NUMERIC
    assert ps.candidate_names({"feature_set": "cs_pa"}) == FEATURE_NAMES + ps.CS_NUMERIC + ps.PA_NUMERIC
    with pytest.raises(KeyError):
        ps.candidate_names({"feature_set": "nope"})


def test_population_table_and_bag_on_synthetic_rows():
    config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    rng = np.random.default_rng(0)
    n = 240
    days = np.repeat(np.arange(30), 8)
    takeoff = pd.Timestamp("2025-07-01 06:00", tz="UTC") + pd.to_timedelta(days, unit="D") + pd.to_timedelta(rng.integers(0, 12 * 3600, n), unit="s")
    interval = rng.integers(3000, 30000, n).astype(float)
    fill = rng.random(n) < np.where(interval > 7200, 0.7, 0.3)
    label = np.where(fill, interval, rng.integers(700, 1500, n)).astype(int)
    dep = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "PHASE_mvt": "DEP", "ADEP_mvt": "LIRF", "ADES_mvt": "EGLL", "MVT_TIME_UTC_mvt": takeoff,
                        "SCHED_TIME_UTC_mvt": takeoff - pd.to_timedelta(interval, unit="s"), "BLOCK_TIME_UTC_mvt": pd.NaT, "AIRCRAFT_TYPE_mvt": None,
                        "AOBT_3_flt": pd.NaT, "LOBT_flt": pd.NaT, "ADEP_flt": None, "STAND_mvt": rng.choice(["A1", "B2"], n), "RUNWAY_mvt": "25",
                        "FLIGHT_mvt": rng.choice(["RYR1", "WMT2", "ITY3"], n), "TAXITIME_SEC_mvt": label})
    arr = pd.DataFrame({"MVT_ID_mvt": 1000 + np.arange(n, dtype=float), "PHASE_mvt": "ARR", "ADEP_mvt": "EGLL", "ADES_mvt": "LIRF",
                        "MVT_TIME_UTC_mvt": takeoff - 3 * H, "SCHED_TIME_UTC_mvt": takeoff - 3 * H, "BLOCK_TIME_UTC_mvt": takeoff - 2 * H,
                        "AIRCRAFT_TYPE_mvt": "B738", "AOBT_3_flt": pd.NaT, "LOBT_flt": pd.NaT, "ADEP_flt": "EGLL", "STAND_mvt": rng.choice(["A1", "B2"], n),
                        "RUNWAY_mvt": "16", "FLIGHT_mvt": rng.choice(["RYR9", "BAW9"], n), "TAXITIME_SEC_mvt": 600})
    m = pd.concat([dep, arr], ignore_index=True)
    table = ps.population_table(m, config)
    assert len(table) == n and set(ps.PA_NUMERIC + FEATURE_NAMES + ["lomo_month", "movement_date_utc", "flight", "label"]) <= set(table)
    assert table["month"].eq(7).all() and table["lomo_month"].eq("2025-07").all()
    modes = ps.modes_of(table, config)
    assert set(np.unique(modes)) <= {0, 1, 2} and (modes == 0).sum() == fill.sum()
    classifier = {**config["classifier"], "max_rounds": 60, "params": {**config["classifier"]["params"], "num_threads": 2}}
    names = ps.candidate_names({"feature_set": "pa"})
    bag = ps.fit_bag(table, modes, table["movement_date_utc"].to_numpy(), names, classifier, [1, 2])
    assert len(bag["members"]) == 2 and all(50 <= mm["rounds"] <= 60 for mm in bag["members"])
    probabilities = ps.predict_bag(bag, table)
    assert probabilities.shape == (n, 3) and np.allclose(probabilities.sum(axis=1), 1.0)
    component = ps.normal_component(table["label"].to_numpy(dtype=float), table["schedule_interval_s"].to_numpy(dtype=float), modes, config)
    mixture, p_fill = ps.mixture_for(bag, component, table)
    assert np.isfinite(mixture).all() and (0 <= p_fill).all() and (p_fill <= 1).all()
    # in-sample the bag separates the synthetic mechanism: mixture closer to the labels than the normal component alone
    normal = ps.band_values(component, table["schedule_interval_s"].to_numpy(dtype=float))
    assert np.sum((mixture - table["label"]) ** 2) < np.sum((normal - table["label"]) ** 2)
