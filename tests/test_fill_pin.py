"""configs/fill_pin_v1.json: the label-free scope rule and its application (src/prc2026_final/fill_pin.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prc2026_final import fill_pin  # noqa: E402

T0 = pd.Timestamp("2026-07-10 13:00", tz="UTC")


def rows() -> pd.DataFrame:
    h = pd.Timedelta(hours=1)
    return pd.DataFrame({
        "MVT_ID_mvt": np.arange(1.0, 9.0),
        "PHASE_mvt": ["DEP", "DEP", "DEP", "DEP", "DEP", "ARR", "DEP", "DEP"],
        "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "LIRF", "EDDF", "LIRF", "LIRF", "LIRF"],
        "MVT_TIME_UTC_mvt": [T0] * 8,
        "SCHED_TIME_UTC_mvt": [T0 - 17 * h, T0 - 3 * h, T0 - 18 * h, T0 - 26 * h, T0 - 15 * h, T0 - 15 * h, pd.NaT, T0 - 12 * h - pd.Timedelta(seconds=1)],
        "AOBT_3_flt": [pd.NaT, pd.NaT, T0 - 20 * pd.Timedelta(minutes=1), T0 - 24 * h + 60 * pd.Timedelta(seconds=1), pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "ADEP_flt": [None, None, "LIRF", "LIRF", None, None, None, None],
        "AIRCRAFT_TYPE_mvt": [None, None, "A320", "BCS3", None, None, None, None]})


def test_scope_is_lirf_long_interval_population_or_one_day_clock():
    pins = fill_pin.fill_pins(rows())
    assert pins["MVT_ID_mvt"].tolist() == [1.0, 4.0, 8.0] and pins["scope"].tolist() == ["population", "clock_day", "population"]
    assert pins["pin_s"].tolist() == [17 * 3600.0, 26 * 3600.0, 43201.0]
    # row 2: short interval; row 3: long interval but a normal clock and an NM record; row 5: EDDF; row 6: arrival; row 7: no schedule;
    # row 8: interval 43,201 s is in, exactly 43,200 s is out (strict)
    edge = rows()
    edge.loc[7, "SCHED_TIME_UTC_mvt"] = T0 - pd.Timedelta(seconds=43200)
    assert fill_pin.fill_pins(edge)["MVT_ID_mvt"].tolist() == [1.0, 4.0]
    assert fill_pin.fill_pins(rows().iloc[:0])["MVT_ID_mvt"].tolist() == []


def test_apply_replaces_only_pinned_rows_and_refuses_foreign_or_non_lirf_pins():
    pins = fill_pin.fill_pins(rows()).iloc[:2]
    ids = np.array([4.0, 2.0, 1.0])
    values, pinned = fill_pin.apply_pins(np.array([86000.0, 500.0, 60000.0]), ids, pins, np.array(["LIRF", "LIRF", "LIRF"]))
    assert values.tolist() == [26 * 3600.0, 500.0, 17 * 3600.0] and pinned.tolist() == [True, False, True]
    with pytest.raises(ValueError, match="departure of the file"):
        fill_pin.apply_pins(np.array([1.0]), np.array([2.0]), pins)
    with pytest.raises(ValueError, match="outside LIRF"):
        fill_pin.apply_pins(np.array([1.0, 2.0, 3.0]), ids, pins, np.array(["EDDF", "LIRF", "LIRF"]))
    with pytest.raises(ValueError, match="needs columns"):
        fill_pin.fill_pins(rows().drop(columns=["AOBT_3_flt"]))


def test_candidate_bands_restrict_population_rows_only():
    pins = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "pin_s": [50000.0, 61000.0, 90000.0, 94000.0], "scope": ["population"] * 3 + ["clock_day"],
                         "interval_s": [50000.0, 61000.0, 90000.0, 94000.0], "proxy_s": [np.nan, np.nan, np.nan, 86000.0]})
    nine = {"scopes": ["population"], "population_interval_bands_s": [[43200, 58000], [79200, 1000000000]]}
    assert fill_pin.candidate_pins(pins, nine)["MVT_ID_mvt"].tolist() == [1.0, 3.0]
    v1 = {"scopes": ["population", "clock_day"], "population_interval_s_min": 43200}
    assert fill_pin.candidate_pins(pins, v1)["MVT_ID_mvt"].tolist() == [1.0, 2.0, 3.0, 4.0]
    dominant = {"scopes": ["population", "clock_day"], "population_interval_s_min": 79200}
    assert fill_pin.candidate_pins(pins, dominant)["MVT_ID_mvt"].tolist() == [3.0, 4.0]
    assert len(fill_pin.candidate_pins(pins.iloc[:0], nine)) == 0
    one_day = {"scopes": ["population"], "population_interval_bands_s": [[100000, 1000000000]], "pinned_value": {"one_day_plus_s": 1135.5}}
    pins2 = pins.assign(interval_s=[50000.0, 61000.0, 111654.0, 94000.0], pin_s=[50000.0, 61000.0, 111654.0, 94000.0])
    picked = fill_pin.candidate_pins(pins2, one_day)
    assert picked["MVT_ID_mvt"].tolist() == [3.0] and picked["pin_s"].tolist() == [86400.0 + 1135.5] and picked["interval_s"].tolist() == [111654.0]
    assert fill_pin.candidate_pins(pins2, nine)["pin_s"].tolist() == [50000.0, 111654.0]  # no pinned_value: the interval stays the pin


def test_ranking_scope_is_the_declared_twelve_rows():
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=fill_pin.COLUMNS)
    pins = fill_pin.fill_pins(raw)
    assert len(pins) == 12 and pins["scope"].value_counts().to_dict() == {"population": 11, "clock_day": 1}
    assert (pins["pin_s"] > 43200).all() and float(pins.loc[pins["scope"].eq("clock_day"), "pin_s"].iloc[0]) == 94560.0
