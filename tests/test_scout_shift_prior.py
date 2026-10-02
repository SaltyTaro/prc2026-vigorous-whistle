import numpy as np
import pandas as pd
import pytest

from prc2026_daily.reconcile import Settings, prepare_day, solve_day
from prc2026_scout.shift_prior import fit_mixture, solve_day_mix

V48 = Settings(heavy_updates=False)


def day(shift: float, seed: int = 1, hedged: bool = False):
    rng = np.random.default_rng(seed)
    ord_a, ord_yhat = rng.uniform(1200, 4000, 1200), rng.uniform(600, 1200, 1200)
    ord_y = ord_yhat + rng.normal(0, 100, 1200)
    n = 400
    a, yhat = rng.uniform(1200, 4000, n), rng.uniform(600, 1200, n)
    y = yhat + rng.normal(0, 100, n) + shift
    groups = [(np.arange(n), (ord_a, ord_yhat, ord_y))]
    if hedged:
        h_a, h_yhat = rng.uniform(30000, 36000, 400), np.full(400, 16000.0)
        h_y = np.where(rng.random(400) < 0.5, h_a + rng.normal(0, 10, 400), 900 + rng.normal(0, 100, 400))
        a, yhat, y = np.append(a, 32000.0), np.append(yhat, 16000.0), np.append(y, 900.0)
        groups.append((np.array([n]), (h_a, h_yhat, h_y)))
    return a, yhat, y, groups, float(np.maximum(a - y, 0).sum())


@pytest.mark.parametrize("shift,hedged", [(0.0, False), (30.0, False), (-60.0, True)])
def test_pi_zero_reproduces_solve_day(shift, hedged):
    a, yhat, y, groups, total = day(shift, hedged=hedged)
    p = prepare_day(a, yhat, groups, V48)
    ref, info_ref = solve_day(p, total, 60.0, 0.02, 0.0, 20.0, V48)
    new, info = solve_day_mix(p, total, 60.0, 0.02, 20.0, 0.0, 1.0, V48)
    assert np.allclose(new, ref, atol=1e-8) and info["shift_s"] == pytest.approx(info_ref["shift_s"], abs=1e-9)


def test_a_disruption_sized_shift_moves_further_under_the_mixture():
    a, yhat, y, groups, total = day(150.0)                      # tau 15 s: a ten-tau day
    p = prepare_day(a, yhat, groups, V48)
    _, gauss = solve_day_mix(p, total, 60.0, 0.02, 15.0, 0.0, 1.0, V48)
    new, mix = solve_day_mix(p, total, 60.0, 0.02, 15.0, 0.05, 5.0, V48)
    assert gauss["shift_s"] < 70.0 and mix["shift_s"] > 120.0
    assert np.sqrt(np.mean((new - y) ** 2)) < 0.8 * np.sqrt(np.mean((yhat - y) ** 2))


def test_an_ordinary_day_is_nearly_unchanged_by_the_mixture():
    a, yhat, y, groups, total = day(10.0)
    p = prepare_day(a, yhat, groups, V48)
    _, gauss = solve_day_mix(p, total, 60.0, 0.02, 15.0, 0.0, 1.0, V48)
    _, mix = solve_day_mix(p, total, 60.0, 0.02, 15.0, 0.05, 5.0, V48)
    assert abs(mix["shift_s"] - gauss["shift_s"]) < 3.0


def test_fit_mixture_recovers_a_known_mixture():
    rng = np.random.default_rng(3)
    rows = []
    for airport, tau in [("A", 10.0), ("B", 20.0)]:
        wide = rng.random(4000) < 0.05
        b = np.where(wide, rng.normal(0, 4 * tau, 4000), rng.normal(0, tau, 4000))
        rows.append(pd.DataFrame({"airport": airport, "mean": b, "sampling_var": np.full(4000, 1e-6)}))
    r = fit_mixture(pd.concat(rows, ignore_index=True), {"A": 10.0, "B": 20.0})
    assert 0.03 <= r["pi"] <= 0.08 and 3.0 <= r["k"] <= 5.0 and r["loglik"] > r["loglik_gaussian"]
