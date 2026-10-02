import numpy as np
import pandas as pd

from prc2026_scout import count_days as cd


def samples(n=400, seed=0):
    """Fitting outcomes of an ordinary cell: labels within a few minutes of their predictions."""
    rng = np.random.default_rng(seed)
    sy = rng.uniform(500, 900, n)
    return np.full(n, 3000.0), sy, sy + rng.normal(0, 150, n)


def month(rows):
    f = pd.DataFrame(rows, columns=["takeoff", "a", "value"])
    f["takeoff"] = pd.to_datetime(f["takeoff"], utc=True)
    return f


def solve(f, n_pub, complete=None):
    groups = [(np.arange(len(f)), samples())]
    days = ["2026-07-01", "2026-07-02", "2026-07-03"]
    return cd.solve_month(f, groups, days, n_pub, set(days[:-1]) if complete is None else complete, 1e-3, 300.0, 0.02, 20.0,
                          lambda sub: [(np.arange(len(sub)), samples())], {})


def test_counts_that_agree_leave_every_value():
    # two departures just after midnight whose values put the off-block before midnight, and the counts agree with the values
    f = month([["2026-07-02 00:05:00", 3000, 700], ["2026-07-02 00:08:00", 3000, 650], ["2026-07-01 12:00:00", 3000, 700], ["2026-07-02 12:00:00", 3000, 700]])
    new, rep = solve(f, {"2026-07-01": 3, "2026-07-02": 1})
    assert rep["status"] == "ok" and rep["candidates"] == 2
    assert np.allclose(new, f["value"].to_numpy())


def test_counts_move_near_midnight_rows_across():
    # the same rows, but the published count says both left after midnight: their values fall below the time since midnight
    f = month([["2026-07-02 00:05:00", 3000, 700], ["2026-07-02 00:08:00", 3000, 650], ["2026-07-01 12:00:00", 3000, 700], ["2026-07-02 12:00:00", 3000, 700]])
    new, rep = solve(f, {"2026-07-01": 1, "2026-07-02": 3})
    assert rep["status"] == "ok"
    assert new[0] < 700 and new[1] < 650 and new[0] <= 300 + 60 and new[1] <= 480 + 60
    assert new[2] == 700 and new[3] == 700


def test_no_feasible_assignment_keeps_values():
    f = month([["2026-07-01 12:00:00", 3000, 700], ["2026-07-02 12:00:00", 3000, 700]])
    new, rep = solve(f, {"2026-07-01": 5, "2026-07-02": 1})
    assert rep["status"] == "infeasible" and np.array_equal(new, f["value"].to_numpy())


def test_count_exactness_rounds_to_the_minute_and_skips_month_ends():
    dep = pd.DataFrame({"airport": "X", "in_figure": True,
                        "block": pd.to_datetime(["2025-03-03 23:59:55", "2025-03-04 10:00:00", "2025-03-31 23:00:00"], utc=True)})
    published = pd.Series({("X", "2025-03-04"): 2.0, ("X", "2025-03-31"): 7.0}).rename_axis(["airport", "day"])
    e = cd.count_exactness(dep, published)
    assert e.loc["X", "days"] == 1 and e.loc["X", "share"] == 1.0 and bool(e.loc["X", "exact"])
