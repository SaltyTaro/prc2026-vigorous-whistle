"""Three-class fill/normal/one-day classifier and the declared mixture prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from prc2026_lirf_mixture.features import CATEGORICAL, FEATURE_NAMES


def _matrix(features: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    frame = features[FEATURE_NAMES].copy()
    categories = categories or {}
    for name in CATEGORICAL:
        levels = categories.get(name)
        if levels is None:
            levels = sorted(frame[name].astype(str).unique().tolist())
            categories[name] = levels
        frame[name] = pd.Categorical(frame[name].astype(str), categories=levels)
    return frame, categories


def fit_classifier(features: pd.DataFrame, modes: np.ndarray, dates: np.ndarray, config: dict) -> dict:
    """Fit on fitting rows only; rounds from date-grouped inner folds so same-day rows never split."""
    spec = config["classifier"]
    frame, categories = _matrix(features)
    modes = np.asarray(modes, dtype=int)
    if len(frame) != len(modes) or len(dates) != len(modes) or set(np.unique(modes)) - {0, 1, 2}:
        raise ValueError("Classifier inputs must align and use the three declared modes.")
    params = dict(spec["params"])
    best = []
    for train, valid in GroupKFold(spec["inner_folds"]).split(frame, modes, groups=dates):
        booster = lgb.train(params, lgb.Dataset(frame.iloc[train], modes[train], categorical_feature=CATEGORICAL),
                            spec["max_rounds"], valid_sets=[lgb.Dataset(frame.iloc[valid], modes[valid], categorical_feature=CATEGORICAL)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        best.append(booster.best_iteration)
    rounds = max(50, int(np.rint(np.median(best))))
    booster = lgb.train(params, lgb.Dataset(frame, modes, categorical_feature=CATEGORICAL), rounds)
    return {"booster": booster, "categories": categories, "rounds": rounds, "inner_best_iterations": [int(b) for b in best],
            "fitting_rows": int(len(frame)), "mode_counts": {str(k): int((modes == k).sum()) for k in range(3)}}


def predict_modes(model: dict, features: pd.DataFrame) -> np.ndarray:
    frame, _ = _matrix(features, model["categories"])
    probabilities = model["booster"].predict(frame, num_iteration=model["rounds"])
    if probabilities.shape != (len(frame), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Classifier produced invalid probabilities.")
    return probabilities


def normal_component(labels: np.ndarray, interval: np.ndarray, modes: np.ndarray, config: dict) -> dict:
    """Median fitting label of normal-mode rows per declared schedule-interval band."""
    bands = [float(v) for v in config["mixture"]["normal_component_bands_seconds"]]
    normal = np.asarray(modes) == 1
    assignment = np.digitize(interval[normal], bands[1:-1], right=True)
    medians = {}
    for band in range(len(bands) - 1):
        chosen = labels[normal][assignment == band]
        medians[str(band)] = float(np.median(chosen)) if len(chosen) else None
    overall = float(np.median(labels[normal]))
    return {"bands_seconds": bands, "band_medians": medians, "overall_median": overall}


def band_values(component: dict, interval: np.ndarray) -> np.ndarray:
    bands = component["bands_seconds"]
    assignment = np.digitize(interval, bands[1:-1], right=True)
    values = np.array([component["band_medians"][str(b)] if component["band_medians"][str(b)] is not None
                       else component["overall_median"] for b in assignment], dtype=float)
    return values


def mixture_values(probabilities: np.ndarray, interval: np.ndarray, component: dict) -> np.ndarray:
    fill = probabilities[:, 0] / (probabilities[:, 0] + probabilities[:, 1])
    return fill * interval + (1 - fill) * band_values(component, interval)


def apply_candidate(incumbent: np.ndarray, population_index: np.ndarray, interval: np.ndarray, mixture: np.ndarray,
                    weight: float, threshold: float, floor: float) -> tuple[np.ndarray, dict]:
    """Blend the mixture into incumbent predictions on population rows with s <= threshold, then floor."""
    values = np.array(incumbent, dtype=float, copy=True)
    eligible = interval <= threshold
    rows = population_index[eligible]
    values[rows] = (1 - weight) * values[rows] + weight * mixture[eligible]
    floored = values < floor
    values[floored] = floor
    return values, {"population_rows": int(len(population_index)), "mixture_rows": int(eligible.sum()),
                    "floored_rows": int(floored.sum()), "changed_rows": int((values != np.asarray(incumbent, dtype=float)).sum())}
