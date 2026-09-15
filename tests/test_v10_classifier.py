"""The bagged classifier averages its members; needs scikit-learn (development host only)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_lirf_mixture.features import classifier_features, label_modes, population_rows
from prc2026_v10.features import EXTENDED_FEATURE_NAMES, density_features
from prc2026_v10.model import fit_bagged_classifier, predict_bagged_modes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/v10_v2.json").read_text(encoding="utf-8"))
MIXTURE = json.loads((ROOT / "configs/lirf_mixture_v1.json").read_text(encoding="utf-8"))


def test_bagged_classifier_averages_members():
    rng = np.random.default_rng(0)
    n = 240
    stamps = pd.Timestamp("2025-03-01T06:00:00Z") + pd.to_timedelta(rng.integers(0, 30 * 24, n), unit="h")
    interval = rng.uniform(1000, 30000, n)
    frame = pd.DataFrame({ID: np.arange(n, dtype=float), PHASE: "DEP", TIME: stamps, "SCHED_TIME_UTC_mvt": stamps - pd.to_timedelta(interval, unit="s"),
                          "ADEP_mvt": "LIRF", "ADES_mvt": rng.choice(["EGLL", "LFPG", "LEBL"], n), "AIRCRAFT_TYPE_mvt": None, "AOBT_3_flt": pd.NaT, "LOBT_flt": pd.NaT,
                          "STAND_mvt": rng.choice(["301", "302"], n), "RUNWAY_mvt": "25", "FLIGHT_mvt": rng.choice(["AZA1", "BAW2", "RYR3"], n), "BLOCK_TIME_UTC_mvt": pd.NaT})
    fill = rng.random(n) < 0.5
    frame[TARGET] = np.where(fill, interval, 900.0)
    for name in ["AOBT_3_flt", "LOBT_flt", "BLOCK_TIME_UTC_mvt"]:
        frame[name] = pd.to_datetime(frame[name], utc=True)
    population = population_rows(frame)
    features = classifier_features(population, frame, 1200).merge(density_features(population, frame, CONFIG), on=ID)
    modes = label_modes(population[TARGET].to_numpy(dtype=float), features["schedule_interval_s"].to_numpy(), MIXTURE)
    small = {**MIXTURE, "classifier": {**MIXTURE["classifier"], "max_rounds": 60, "inner_folds": 3}}
    bag = fit_bagged_classifier(features, modes, population[TIME].dt.strftime("%Y-%m-%d").to_numpy(), small, [1, 2])
    assert bag["seeds"] == [1, 2] and len(bag["members"]) == 2 and bag["feature_names"] == EXTENDED_FEATURE_NAMES
    probabilities = predict_bagged_modes(bag, features)
    assert probabilities.shape == (n, 3) and np.allclose(probabilities.sum(axis=1), 1.0)
    member = bag["members"][0]["booster"].predict(features[EXTENDED_FEATURE_NAMES].assign(**{c: pd.Categorical(features[c].astype(str), categories=bag["members"][0]["categories"][c]) for c in bag["members"][0]["categories"]}), num_iteration=bag["members"][0]["rounds"])
    other = bag["members"][1]["booster"].predict(features[EXTENDED_FEATURE_NAMES].assign(**{c: pd.Categorical(features[c].astype(str), categories=bag["members"][1]["categories"][c]) for c in bag["members"][1]["categories"]}), num_iteration=bag["members"][1]["rounds"])
    np.testing.assert_allclose(probabilities, (member + other) / 2)
