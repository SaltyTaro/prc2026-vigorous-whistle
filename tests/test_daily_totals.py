import numpy as np
import pytest

from prc2026_daily.reconcile import Settings, matched_weights, month_prior, prepare_day, reconcile_day, scenarios, shift_estimate, solve_day


def test_scenarios_transfer_each_kind_of_outcome():
    sample_a = np.array([3000.0, 9000.0, 70000.0, 9000.0, 3000.0])
    sample_yhat = np.array([900.0, 5000.0, 60000.0, 5000.0, 900.0])
    sample_y = np.array([1000.0, 9030.0, 87000.0, 800.0, 2400.0])   # ordinary, fill, before schedule (one-day), far below, far above
    y = scenarios(np.array([6000.0]), np.array([1200.0]), sample_a, sample_yhat, sample_y)[0]
    assert y[0] == pytest.approx(1300.0)            # the residual carries over
    assert y[1] == pytest.approx(6030.0)            # off-block = schedule: the row's own schedule interval
    assert y[2] == pytest.approx(87000.0)           # stays before the row's schedule
    assert y[3] == pytest.approx(800.0)             # the recorded taxi time, not above the prediction
    assert y[4] == pytest.approx(1200.0 + 1500.0)   # a label above the prediction: residual


def test_matched_weights_reproduce_the_prediction():
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(900, 100, (3, 200)), rng.normal(12000, 100, (3, 200))], axis=1)
    target = np.array([2000.0, 6000.0, 11000.0])
    w = matched_weights(y, target)
    assert np.allclose(w.sum(axis=1), 1.0)
    assert np.allclose((w * y).sum(axis=1), target, atol=1.0)
    assert w[0, :200].sum() > 0.85 and w[2, 200:].sum() > 0.85


def day(shift: float, truth_of_hedged: float | None, seed: int = 1):
    rng = np.random.default_rng(seed)
    ord_a, ord_yhat = rng.uniform(1200, 4000, 1200), rng.uniform(600, 1200, 1200)
    ord_y = ord_yhat + rng.normal(0, 100, 1200)
    n = 400
    a, yhat = rng.uniform(1200, 4000, n), rng.uniform(600, 1200, n)
    y = yhat + rng.normal(0, 100, n) + shift
    groups = [(np.arange(n), (ord_a, ord_yhat, ord_y))]
    if truth_of_hedged is not None:
        h_a, h_yhat = rng.uniform(30000, 36000, 400), np.full(400, 16000.0)
        h_y = np.where(rng.random(400) < 0.5, h_a + rng.normal(0, 10, 400), 900 + rng.normal(0, 100, 400))
        a, yhat, y = np.append(a, 32000.0), np.append(yhat, 16000.0), np.append(y, truth_of_hedged)
        groups.append((np.array([n]), (h_a, h_yhat, h_y)))
    return a, yhat, y, groups, float(np.maximum(a - y, 0).sum())


def test_day_shift_is_recovered_and_ordinary_rows_improve():
    a, yhat, y, groups, total = day(30.0, None)
    new, info = reconcile_day(a, yhat, groups, total, 60.0, 0.02, 30.0)
    assert info["used"] and info["heavy"] == 0
    assert info["shift_s"] == pytest.approx(30.0, abs=8.0)
    assert np.sqrt(np.mean((new - y) ** 2)) < np.sqrt(np.mean((yhat - y) ** 2))


def test_a_large_stake_is_resolved_and_left_alone_when_heavy_updates_are_off():
    for truth in (32000.0, 900.0):
        a, yhat, y, groups, total = day(0.0, truth)
        new, info = reconcile_day(a, yhat, groups, total, 60.0, 0.02, 20.0)
        assert info["heavy"] == 1
        assert abs(new[-1] - truth) < 0.2 * abs(yhat[-1] - truth)
        kept, _ = reconcile_day(a, yhat, groups, total, 60.0, 0.02, 20.0, Settings(heavy_updates=False))
        assert kept[-1] == yhat[-1]
        assert np.sqrt(np.mean((kept[:-1] - y[:-1]) ** 2)) <= np.sqrt(np.mean((yhat[:-1] - y[:-1]) ** 2)) + 1.0   # the heavy row is accounted for, the others are not dragged


def test_without_information_nothing_moves():
    a, yhat, y, groups, total = day(0.0, None)
    p = prepare_day(a, yhat, groups)
    new, info = solve_day(p, float("nan"), 60.0, 0.02, 0.0, 30.0)
    assert not info["used"] and np.array_equal(new, yhat)
    implied = p.reference_total + p.mean_e            # the total the model expects: no shift estimated, nothing moves beyond noise
    new, info = solve_day(p, implied, 60.0, 0.02, 0.0, 30.0)
    assert abs(info["shift_s"]) < 1.0 and np.max(np.abs(new - yhat)) < 2.0
    assert shift_estimate(p, implied) == pytest.approx(0.0, abs=1e-9)


def test_month_prior():
    rng = np.random.default_rng(2)
    x = 20.0 + rng.normal(0, 10.0, 31)
    mean, sd = month_prior(x, np.full(31, 25.0))
    assert 12.0 < mean < 26.0 and 5.0 < sd < 14.0
    assert month_prior(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == (0.0, 3.0)


def two_part_day(shift_tracked: float, shift_untracked: float, share_untracked: float = 0.3, seed: int = 3):
    a, yhat, y, groups, _ = day(0.0, None, seed)
    component = (np.random.default_rng(seed).random(len(a)) < share_untracked).astype(np.int64)
    y = y + np.where(component == 1, shift_untracked, shift_tracked)
    return a, yhat, y, groups, component, float(np.maximum(a - y, 0).sum())


def test_components_with_one_common_shift_equal_the_single_shift():
    a, yhat, y, groups, component, total = two_part_day(30.0, 30.0)
    single, info = reconcile_day(a, yhat, groups, total, 60.0, 0.02, 30.0)
    p = prepare_day(a, yhat, groups, component=component, components=2)
    assert p.n_eff_k.sum() == pytest.approx(p.n_eff)
    both, info2 = solve_day(p, total, 60.0, 0.02, 0.0, 0.0, shift_cov=np.full((2, 2), 30.0 ** 2))
    assert info2["shifts_s"][0] == pytest.approx(info2["shifts_s"][1])
    assert info2["shift_s"] == pytest.approx(info["shift_s"], abs=1e-6)
    assert np.allclose(both, single, atol=1e-6)


def test_components_split_the_discrepancy_by_their_prior():
    a, yhat, y, groups, component, total = two_part_day(0.0, 60.0)
    p = prepare_day(a, yhat, groups, component=component, components=2)
    cov = np.array([[5.0 ** 2, 0.0], [0.0, 40.0 ** 2]])
    new, info = solve_day(p, total, 60.0, 0.02, 0.0, 0.0, shift_cov=cov)
    tracked, untracked = info["shifts_s"]
    assert abs(tracked) < 5.0 and untracked == pytest.approx(60.0, abs=12.0)
    assert tracked / untracked == pytest.approx(25.0 * p.n_eff_k[0] / (1600.0 * p.n_eff_k[1]), rel=1e-6)   # load_k = cov_kk n_k without correlation
    one, _ = reconcile_day(a, yhat, groups, total, 60.0, 0.02, 30.0)
    rmse = lambda v: float(np.sqrt(np.mean((v - y) ** 2)))
    assert rmse(new) < rmse(one) < rmse(yhat)
    still, info = solve_day(p, total, 60.0, 0.02, 0.0, 0.0, shift_cov=np.array([[0.0, 0.0], [0.0, 40.0 ** 2]]))
    assert info["shifts_s"][0] == 0.0 and info["shifts_s"][1] > 40.0


def test_components_carry_a_prior_mean():
    a, yhat, y, groups, component, total = two_part_day(0.0, 0.0)
    p = prepare_day(a, yhat, groups, component=component, components=2)
    implied = p.reference_total + p.mean_e
    new, info = solve_day(p, implied - p.n_eff_k[1] * 25.0, 60.0, 0.02, np.array([0.0, 25.0]), 0.0, shift_cov=np.diag([100.0, 400.0]))
    assert info["shifts_s"][0] == pytest.approx(0.0, abs=1.0) and info["shifts_s"][1] == pytest.approx(25.0, abs=1.0)
    with pytest.raises(ValueError):
        prepare_day(a, yhat, groups, component=component, components=1)
    with pytest.raises(ValueError):
        solve_day(p, implied, 60.0, 0.02, 0.0, 0.0, shift_cov=np.eye(3))
