"""configs/pop_signals_v3.json: operator descriptors and stake-weighted classifier fits (src/prc2026_final/pop_signals.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from prc2026.artifacts import read_json  # noqa: E402
from prc2026_final import pop_signals as ps  # noqa: E402
from prc2026_lirf_mixture.features import FEATURE_NAMES  # noqa: E402
from test_pop_signals import H, movements  # noqa: E402


def test_operator_descriptors_per_month_and_prefix():
    m = movements()
    d = ps.operator_descriptors(m, 120.0)
    # arrivals in 2025-07: RYR122 (fill), EZY77, ITY8, BAW1 -> month share 0.25; smoothing 5 towards it
    assert d.loc[("2025-07", "RYR"), "op_arr_fill_m"] == pytest.approx((1 + 5 * 0.25) / (1 + 5))
    assert d.loc[("2025-07", "RYR"), "op_arr_n_m"] == pytest.approx(np.log1p(1))
    assert d.loc[("2025-07", "ITY"), "op_arr_fill_m"] == pytest.approx((0 + 5 * 0.25) / (1 + 5))
    # WMT departs but never arrives: the month's share with count 0
    assert d.loc[("2025-07", "WMT"), "op_arr_fill_m"] == pytest.approx(0.25) and d.loc[("2025-07", "WMT"), "op_arr_n_m"] == 0.0
    # every departure has no NM record and no type; one departure per prefix (the missing flight is prefix '?')
    for prefix in ["RYR", "WMT", "ITY", "?"]:
        assert d.loc[("2025-07", prefix), "prefix_nonm_m"] == 1.0 and d.loc[("2025-07", prefix), "prefix_notype_m"] == 1.0
        assert d.loc[("2025-07", prefix), "prefix_n_m"] == pytest.approx(np.log1p(1))
    # an NM-filed departure of the same prefix in the same month lowers the no-NM share but not the no-type share
    m2 = pd.concat([m, m[m["FLIGHT_mvt"].eq("RYR123")].assign(MVT_ID_mvt=99.0, ADEP_flt="LIRF", FLIGHT_mvt="RYR555")], ignore_index=True)
    d2 = ps.operator_descriptors(m2, 120.0)
    assert d2.loc[("2025-07", "RYR"), "prefix_nonm_m"] == 0.5 and d2.loc[("2025-07", "RYR"), "prefix_notype_m"] == 1.0
    assert d2.loc[("2025-07", "RYR"), "prefix_n_m"] == pytest.approx(np.log1p(2))
    # the population table carries them by (month, prefix)
    table = ps.population_table(m, read_json(ROOT / "configs/lirf_mixture_v1.json"))
    assert set(ps.OP_NUMERIC + ps.OPD_NUMERIC) <= set(table)
    assert table.set_index("flight").loc["RYR123", "op_arr_fill_m"] == pytest.approx((1 + 5 * 0.25) / 6)
    assert table["prefix_nonm_m"].eq(1.0).all()
    assert ps.candidate_names({"feature_set": "pa_op"}) == FEATURE_NAMES + ps.PA_NUMERIC + ps.OP_NUMERIC
    assert ps.candidate_names({"feature_set": "pa_opall"}) == FEATURE_NAMES + ps.PA_NUMERIC + ps.OP_NUMERIC + ps.OPD_NUMERIC


def test_stake_weights_clip_and_scale():
    component = {"bands_seconds": [-1e9, 1800.0, 3600.0, 1e9], "band_medians": {"0": 800.0, "1": 900.0, "2": 1000.0}, "overall_median": 900.0}
    table = pd.DataFrame({"schedule_interval_s": [1000.0, 3000.0, 20000.0, 50000.0]})
    w = ps.stake_weights(table, component, {"floor_s": 600, "cap_s": 20000})
    raw = np.array([600.0, 2100.0, 19000.0, 20000.0])  # |1000-800| floored to 600, 3000-900, 20000-1000, 50000-1000 capped
    assert w == pytest.approx(raw / raw.mean())
    assert w.mean() == pytest.approx(1.0)


def test_weighted_bag_fits_and_rejects_bad_weights():
    config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    rng = np.random.default_rng(1)
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
    table = ps.population_table(pd.concat([dep, arr], ignore_index=True), config)
    modes = ps.modes_of(table, config)
    component = ps.normal_component(table["label"].to_numpy(dtype=float), table["schedule_interval_s"].to_numpy(dtype=float), modes, config)
    weight = ps.stake_weights(table, component, {"floor_s": 600, "cap_s": 20000})
    assert weight.shape == (n,) and (weight > 0).all() and weight.mean() == pytest.approx(1.0)
    classifier = {**config["classifier"], "max_rounds": 60, "params": {**config["classifier"]["params"], "num_threads": 2}}
    names = ps.candidate_names({"feature_set": "pa_op"})
    bag = ps.fit_bag(table, modes, table["movement_date_utc"].to_numpy(), names, classifier, [1], weight)
    assert bag["weighted"] and bag["weight_summary"]["mean"] == pytest.approx(1.0) and 50 <= bag["members"][0]["rounds"] <= 60
    probabilities = ps.predict_bag(bag, table)
    assert probabilities.shape == (n, 3) and np.allclose(probabilities.sum(axis=1), 1.0)
    unweighted = ps.fit_bag(table, modes, table["movement_date_utc"].to_numpy(), names, classifier, [1])
    assert not unweighted["weighted"] and unweighted["weight_summary"] is None
    assert not np.allclose(ps.predict_bag(unweighted, table), probabilities)
    with pytest.raises(ValueError):
        ps.fit_bag(table, modes, table["movement_date_utc"].to_numpy(), names, classifier, [1], weight[:-1])
    with pytest.raises(ValueError):
        ps.fit_bag(table, modes, table["movement_date_utc"].to_numpy(), names, classifier, [1], np.zeros(n))
