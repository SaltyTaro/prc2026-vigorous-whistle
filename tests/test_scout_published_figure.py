import numpy as np
import pandas as pd
import pytest

from prc2026_scout.published_figure import WIDE_FLOOR, day_terms, fit, fitting_days, in_figure


def test_in_figure_drops_only_vfr_rows_at_ifr_only_airports():
    airport = np.array(["EHAM", "EHAM", "LSZH", "LEBL", "EDDF", "LFPG"])
    ifr = np.array([True, False, False, False, False, True])
    assert in_figure(airport, ifr).tolist() == [True, False, False, False, True, True]


def synthetic_days(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(60):                       # LFPG: published holds dn missing flights of about 1,500 s each
        dn = int(rng.integers(0, 4))
        rows.append({"airport": "LFPG", "bday": f"d{k}", "dn": dn, "diff": 0.0 + rng.normal(0, 20) + sum(rng.gamma(2.0, 750.0) for _ in range(dn))})
    for k in range(60):                       # EHAM: exact up to a constant 140 s
        rows.append({"airport": "EHAM", "bday": f"d{k}", "dn": 0, "diff": 140.0 + rng.normal(0, 60)})
    return pd.DataFrame(rows)


def test_fit_recovers_offset_and_per_flight_term():
    model = fit(synthetic_days())
    lfpg, eham = model["LFPG"], model["EHAM"]
    assert abs(lfpg["c0"]) < 30 and lfpg["s0"] == pytest.approx(30.0, abs=15)
    assert 1100 < lfpg["sign"][1]["mu"] < 1900 and not lfpg["sign"][1]["pooled"]
    assert eham["c0"] == pytest.approx(140.0, abs=30)
    assert eham["sign"][1]["pooled"] and eham["sign"][1]["mu"] == pytest.approx(lfpg["sign"][1]["mu"])
    assert eham["wide"] >= WIDE_FLOOR


def test_day_terms_grow_with_the_count_difference():
    model = fit(synthetic_days())
    o0, s0, _ = day_terms(model, "LFPG", 0)
    o2, s2, w = day_terms(model, "LFPG", 2)
    assert o0 == pytest.approx(model["LFPG"]["c0"]) and s0 == pytest.approx(model["LFPG"]["s0"])
    assert o2 == pytest.approx(model["LFPG"]["c0"] + 2 * model["LFPG"]["sign"][1]["mu"])
    assert s2 > s0 and 0 < w < 1


def test_fitting_days_use_the_true_off_block_day_and_the_coverage_rule():
    rows = pd.DataFrame({"airport": ["X"] * 5, "block_day": ["d1", "d1", "d1", "d2", "d2"], "bday": ["d1", "d1", "d2", "d2", "d2"],
                         "pos_true": [60.0, 0.0, 120.0, 30.0, 30.0], "a": [1.0] * 5, "pub": [200.0, 200.0, 50.0, 50.0, 50.0], "n_pub": [3.0, 3.0, 2.0, 2.0, 2.0]})
    days = fitting_days(rows).set_index("bday")
    assert days.loc["d1", "diff"] == pytest.approx(200.0 - 180.0) and days.loc["d1", "dn"] == 0
    assert days.loc["d2", "diff"] == pytest.approx(50.0 - 60.0) and days.loc["d2", "dn"] == 0
