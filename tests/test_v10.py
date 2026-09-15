"""Day-density features are label-free counts; v10 items compose in the declared order; the rule is fixed."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_lirf_mixture.features import FEATURE_NAMES, population_rows
from prc2026_v10.features import DENSITY_NAMES, EXTENDED_FEATURE_NAMES, density_features
from prc2026_v10.model import ITEM_NAMES, apply_items, bagging_seeds, select_candidate

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/v10_v2.json").read_text(encoding="utf-8"))
MIXTURE = json.loads((ROOT / "configs/lirf_mixture_v1.json").read_text(encoding="utf-8"))


def movements():
    day1, day2 = pd.Timestamp("2025-03-01T12:00:00Z"), pd.Timestamp("2025-03-02T22:30:00Z")
    rows = [
        # id, phase, time, sched, adep, ades, type, aobt, lobt, stand, runway, flight, block, target
        (1.0, "DEP", day1, day1 - pd.Timedelta(hours=3), "LIRF", "EGLL", None, pd.NaT, pd.NaT, "301", "25", "BAW551", pd.NaT, 10800.0),
        (2.0, "DEP", day1 + pd.Timedelta(hours=1), day1 - pd.Timedelta(minutes=40), "LIRF", "LEBL", None, pd.NaT, pd.NaT, "302", "25", "WMTMT4SG", pd.NaT, 900.0),
        (3.0, "DEP", day1 + pd.Timedelta(hours=5), day1 - pd.Timedelta(hours=14), "LIRF", "LLBG", None, pd.NaT, pd.NaT, "303", "25", "ISR342", pd.NaT, 87300.0),
        (4.0, "DEP", day1, day1 - pd.Timedelta(hours=3), "LIRF", "EGLL", "A320", pd.NaT, pd.NaT, "301", "25", "BAW552", pd.NaT, 950.0),
        (5.0, "DEP", day1, day1 - pd.Timedelta(hours=3), "EGLL", "LIRF", None, pd.NaT, pd.NaT, "5", "27R", "BAW553", pd.NaT, 950.0),
        (6.0, "DEP", day1, day1 - pd.Timedelta(minutes=20), "LIRF", "EGLL", "A320", day1 - pd.Timedelta(minutes=15), pd.NaT, "301", "25", "BAW554", pd.NaT, 900.0),
        (7.0, "ARR", day1 - pd.Timedelta(hours=2), day1 - pd.Timedelta(hours=2), "EGLL", "LIRF", "A320", pd.NaT, pd.NaT, "301", "16L", "BAW550", day1 - pd.Timedelta(hours=1, minutes=50), 600.0),
        (8.0, "DEP", day2, day2 - pd.Timedelta(hours=2), "LIRF", "LFPG", None, pd.NaT, pd.NaT, "401", "16L", "AFR123", pd.NaT, 7200.0),
    ]
    frame = pd.DataFrame(rows, columns=[ID, PHASE, TIME, "SCHED_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
                                        "AOBT_3_flt", "LOBT_flt", "STAND_mvt", "RUNWAY_mvt", "FLIGHT_mvt", "BLOCK_TIME_UTC_mvt", TARGET])
    for name in [TIME, "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "LOBT_flt", "BLOCK_TIME_UTC_mvt"]:
        frame[name] = pd.to_datetime(frame[name], utc=True)
    return frame


def test_density_features_count_only_supplied_rows_per_local_day():
    frame = movements()
    population = population_rows(frame)
    assert population[ID].tolist() == [1.0, 2.0, 3.0, 8.0]
    out = density_features(population, frame, CONFIG).set_index(ID)
    assert list(out.columns) == DENSITY_NAMES and CONFIG["density_features"]["names"] == DENSITY_NAMES
    # 1 March (Rome): LIRF departures 1, 2, 3, 4, 6 (5 departs EGLL); no-clock ones 1, 2, 3, 4; schedule intervals 10800, 6000, 68400, 10800, 1200 -> 4/5 delayed
    assert out.loc[1.0, "day_departures"] == 5 and out.loc[1.0, "day_noclock_departures"] == 4
    assert out.loc[1.0, "day_delayed_share"] == pytest.approx(4 / 5)
    assert out.loc[1.0, "day_population_rows"] == 3 and out.loc[1.0, "day_population_mean_s"] == pytest.approx((10800 + 6000 + 68400) / 3)
    # Row 8 is on 2 March local (23:30 Rome) and alone that day
    assert out.loc[8.0, "day_population_rows"] == 1 and out.loc[8.0, "day_departures"] == 1 and out.loc[8.0, "window2h_population_rows"] == 1
    # Two-hour window: rows 1 and 2 are one hour apart; row 3 is five hours after row 1
    assert out.loc[1.0, "window2h_population_rows"] == 2 and out.loc[3.0, "window2h_population_rows"] == 1
    # Labels are never read: shuffling them leaves every feature unchanged
    shuffled = frame.copy()
    shuffled[TARGET] = shuffled[TARGET].to_numpy()[::-1]
    pd.testing.assert_frame_equal(density_features(population_rows(shuffled), shuffled, CONFIG), density_features(population, frame, CONFIG))
    assert EXTENDED_FEATURE_NAMES == FEATURE_NAMES + DENSITY_NAMES


def test_bagging_seeds_follow_the_declared_rule():
    assert bagging_seeds(CONFIG, MIXTURE) == [20260915 + k for k in range(8)]
    assert bagging_seeds({}, MIXTURE) == [20260915]
    bad = {**CONFIG, "classifier_bagging": {**CONFIG["classifier_bagging"], "seeds": [1, 2]}}
    with pytest.raises(ValueError):
        bagging_seeds(bad, MIXTURE)


def test_items_compose_in_declared_order_and_touch_only_declared_rows():
    incumbent = np.array([1000.0, 2000.0, -300.0, 5000.0, 700.0, 900.0])
    clock = np.array([True, True, True, False, False, False])
    negative = np.array([False, True, True, False, False, False])
    cat_long = np.array([1400.0, 1000.0, 100.0, np.nan, np.nan, np.nan])
    fallback = np.array([1100.0, 1300.0, 800.0, 4000.0, 800.0, 950.0])
    index, interval = np.array([3, 4]), np.array([20000.0, 50000.0])
    v9_mixture, density_mixture = np.array([9000.0, 50000.0]), np.array([12000.0, 50000.0])
    common = dict(clock=clock, cat_long=cat_long, negative_proxy=negative, fallback=fallback, population_index=index, interval=interval,
                  v9_mixture=v9_mixture, density_mixture=density_mixture, config=CONFIG, threshold=43200.0, floor=-12.0)
    values, counts = apply_items(incumbent, [], **common)
    np.testing.assert_allclose(values, [1000.0, 2000.0, -12.0, 0.25 * 5000 + 0.75 * 9000, 700.0, 900.0])
    assert counts == {"mixture_rows": 1, "population_rows": 2, "floored_rows": 1}
    values, counts = apply_items(incumbent, ITEM_NAMES, **common)
    np.testing.assert_allclose(values, [0.75 * 1000 + 0.25 * 1400, 1300.0, 800.0, 0.25 * 5000 + 0.75 * 12000, 700.0, 900.0])
    assert counts == {"mixture_rows": 1, "population_rows": 2, "cat_long_rows": 3, "routed_rows": 2, "floored_rows": 0}
    values, _ = apply_items(incumbent, ["cat_long_25"], **common)
    np.testing.assert_allclose(values[:3], [0.75 * 1000 + 0.25 * 1400, 0.75 * 2000 + 0.25 * 1000, -12.0])
    with pytest.raises(ValueError):
        apply_items(incumbent, ["undeclared"], **common)
    with pytest.raises(ValueError):
        apply_items(incumbent, [], **{**common, "negative_proxy": np.array([False, False, False, True, False, False])})


def test_selection_rule_requires_margin_and_interval_on_both_folds():
    def candidate(ratios, uppers):
        return {"folds": {fold: {"rmse_ratio_to_v9": r, "paired_days": {"rmse_difference_95pct_interval_seconds": [-1.0, u]}}
                          for fold, r, u in zip(["seasonal", "forward"], ratios, uppers)}}
    candidates = {"a": candidate([0.997, 0.996], [-0.1, -0.2]), "b": candidate([0.995, 0.999], [-0.1, -0.2]),
                  "c": candidate([0.996, 0.994], [0.02, -0.3]), "d": candidate([0.9975, 0.9975], [-0.05, -0.05])}
    assert select_candidate(candidates, ["seasonal", "forward"], 0.0025) == "a"
    assert candidates["b"]["eligible"] is False and candidates["c"]["eligible"] is False and candidates["d"]["eligible"] is True
    assert select_candidate({"b": candidates["b"]}, ["seasonal", "forward"], 0.0025) is None
    assert CONFIG["minimum_improvement_fraction_each_fold"] == 0.0025 and CONFIG["order_of_application"] == ITEM_NAMES + ["floor"]


def test_refit_declaration_binds_iterations_and_population():
    spec = CONFIG["cat_long_refit"]
    runs = [json.loads((ROOT / CONFIG["parents"][f"ladder_{fold}_cat_long_run"]["path"]).read_text(encoding="utf-8")) for fold in ["seasonal", "forward"]]
    assert spec["iterations"] == int(np.rint(np.median([run["best_iterations"] for run in runs]))) == 9929
    assert all(run["variant"] == "cat_long" and run["bundle_id"] == spec["bundle_id"] for run in runs)
    assert spec["expected_clock_fit_rows"] == 1898549 and spec["worker"] == "arcw0" and CONFIG["refit_months"] == [f"2025-{m:02}" for m in range(1, 12)]
