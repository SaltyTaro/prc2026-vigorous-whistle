import numpy as np
import pandas as pd

from prc2026_scout import var_alloc as va


def day_table():
    t = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "month": "2025-07", "airport": ["EDDF"] * 3 + ["EGLL"] * 3,
                      "bday": ["2025-07-02"] * 3 + ["2025-07-02"] * 3, "moved": True, "in_figure": True,
                      "cls": ["EDDF_m", "EDDF_u", "EDDF_m", "EGLL_m", "LIRF_pop", "EGLL_u"]})
    return t


def test_basis_sums_to_zero_per_day_and_skips_ineligible_rows():
    t = day_table()
    elig = va.eligible(t)
    assert elig.tolist() == [True, True, True, True, False, True]
    w = np.where(elig, [1.0, 2.0, 3.0, 5.0, 9.0, 1.0], np.nan)
    I = np.array([40.0] * 3 + [-20.0] * 3)
    base = va.basis(t, w, I, ["month", "airport", "bday"])
    assert base[4] == 0.0
    assert abs(base[:3].sum()) < 1e-9 and abs(base[[3, 5]].sum()) < 1e-9
    assert np.allclose(base[:3], 40.0 * (np.array([1.0, 2.0, 3.0]) / 2.0 - 1.0))


def test_guard_rows_are_never_eligible():
    t = day_table()
    assert not va.eligible(t, guard=(2.0,))[1]


def test_slope_recovers_the_allocation_and_update_caps_and_floors():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 50, 5000)
    r = 0.3 * base + rng.normal(0, 20, 5000)
    rows = np.ones(5000, dtype=bool)
    assert abs(va.slope(r, base, rows) - 0.3) < 0.01
    value = np.array([500.0, 500.0, 0.0, 700.0])
    new = va.update(value, 1.0, np.array([400.0, -30.0, -50.0, 80.0]), np.array([True, True, True, False]))
    assert new.tolist() == [600.0, 470.0, -12.0, 700.0]


def test_clip_of_the_implied_shift():
    t = day_table().iloc[:3].reset_index(drop=True)
    base = va.basis(t, np.array([1.0, 1.0, 4.0]), np.array([5000.0] * 3), ["month", "airport", "bday"])
    assert np.allclose(base, va.CLIP_S * (np.array([1.0, 1.0, 4.0]) / 2.0 - 1.0))
