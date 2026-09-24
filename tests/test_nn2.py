"""adsb_nn2: the declared target offset of tools/run_adsb_nn2.py."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_adsb_nn2 import target_offset  # noqa: E402


def frame() -> pd.DataFrame:
    return pd.DataFrame({"nmclock_available": [1, 1, 0, 1, 0], "nmclock_proxy_s": [600.0, -35.0, 900.0, np.nan, np.nan]})


def test_none_is_zero():
    assert np.array_equal(target_offset(frame(), "none"), np.zeros(5))


def test_nm_clock_uses_the_proxy_only_where_the_clock_is_present_and_finite():
    assert np.array_equal(target_offset(frame(), "nm_clock"), np.array([600.0, -35.0, 0.0, 0.0, 0.0]))


def test_offset_round_trip_restores_the_label():
    f, label = frame(), np.array([700.0, 20.0, 950.0, 400.0, 300.0])
    offset = target_offset(f, "nm_clock")
    assert np.allclose((label - offset) + offset, label)


def test_undeclared_offset_fails():
    with pytest.raises(ValueError):
        target_offset(frame(), "adsb")
