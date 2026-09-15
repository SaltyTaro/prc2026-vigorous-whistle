"""Extended three-class classifier (v9 features plus day density) and the v10 accumulation arithmetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from prc2026_lirf_mixture.features import CATEGORICAL
from .features import EXTENDED_FEATURE_NAMES

ITEM_NAMES = ["density_classifier", "cat_long_25", "route_negative_proxy"]


def _matrix(features: pd.DataFrame, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    frame = features[EXTENDED_FEATURE_NAMES].copy()
    categories = categories or {}
    for name in CATEGORICAL:
        levels = categories.get(name)
        if levels is None:
            levels = sorted(frame[name].astype(str).unique().tolist())
            categories[name] = levels
        frame[name] = pd.Categorical(frame[name].astype(str), categories=levels)
    return frame, categories


def fit_extended_classifier(features: pd.DataFrame, modes: np.ndarray, dates: np.ndarray, mixture_config: dict) -> dict:
    """Same settings and date-grouped rounds rule as the v9 classifier, on the extended feature list."""
    spec = mixture_config["classifier"]
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
            "fitting_rows": int(len(frame)), "mode_counts": {str(k): int((modes == k).sum()) for k in range(3)},
            "feature_names": list(EXTENDED_FEATURE_NAMES)}


def predict_extended_modes(model: dict, features: pd.DataFrame) -> np.ndarray:
    frame, _ = _matrix(features, model["categories"])
    probabilities = model["booster"].predict(frame, num_iteration=model["rounds"])
    if probabilities.shape != (len(frame), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Classifier produced invalid probabilities.")
    return probabilities


def bagging_seeds(config: dict, mixture_config: dict) -> list[int]:
    """Declared member seeds: the v9 classifier seed plus k; absent declaration means a single member."""
    spec = config.get("classifier_bagging")
    if spec is None:
        return [int(mixture_config["classifier"]["params"]["seed"])]
    base = int(mixture_config["classifier"]["params"]["seed"])
    seeds = [base + k for k in range(int(spec["members"]))]
    if seeds != [int(s) for s in spec["seeds"]] or len(set(seeds)) != len(seeds):
        raise ValueError("Declared bagging seeds do not follow the seed rule.")
    return seeds


def fit_bagged_classifier(features: pd.DataFrame, modes: np.ndarray, dates: np.ndarray, mixture_config: dict, seeds: list[int]) -> dict:
    """One extended classifier per declared seed, each with its own inner rounds; probabilities are averaged."""
    members = []
    for seed in seeds:
        member_config = {**mixture_config, "classifier": {**mixture_config["classifier"], "params": {**mixture_config["classifier"]["params"], "seed": int(seed)}}}
        members.append({"seed": int(seed), **fit_extended_classifier(features, modes, dates, member_config)})
    return {"members": members, "seeds": [int(s) for s in seeds], "rounds": [m["rounds"] for m in members],
            "inner_best_iterations": [m["inner_best_iterations"] for m in members], "mode_counts": members[0]["mode_counts"],
            "fitting_rows": members[0]["fitting_rows"], "feature_names": members[0]["feature_names"]}


def predict_bagged_modes(bag: dict, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.mean([predict_extended_modes(member, features) for member in bag["members"]], axis=0)
    if probabilities.shape != (len(features), 3) or not np.isfinite(probabilities).all():
        raise ValueError("Bagged classifier produced invalid probabilities.")
    return probabilities


def apply_items(incumbent: np.ndarray, items: list[str], *, clock: np.ndarray, cat_long: np.ndarray, negative_proxy: np.ndarray,
                fallback: np.ndarray, population_index: np.ndarray, interval: np.ndarray, v9_mixture: np.ndarray,
                density_mixture: np.ndarray, config: dict, threshold: float, floor: float) -> tuple[np.ndarray, dict]:
    """Declared order: LIRF mixture (v9's or the density one), clock blend, negative-proxy routing, floor.

    `incumbent` is the frozen v7 prediction; the v9 mixture is always applied so that a candidate
    without `density_classifier` reduces exactly to v9 plus the other items.
    """
    if set(items) - set(ITEM_NAMES) or list(config["order_of_application"]) != ITEM_NAMES + ["floor"]:
        raise ValueError("Undeclared item or order of application.")
    values = np.array(incumbent, dtype=float, copy=True)
    clock = np.asarray(clock, dtype=bool)
    negative_proxy = np.asarray(negative_proxy, dtype=bool)
    if values.ndim != 1 or clock.shape != values.shape or negative_proxy.shape != values.shape or (negative_proxy & ~clock).any():
        raise ValueError("Routing masks must align with the predictions and lie on the clock route.")
    eligible = np.asarray(interval, dtype=float) <= float(threshold)
    rows = np.asarray(population_index, dtype=int)[eligible]
    if clock[rows].any():
        raise ValueError("Population rows must lie on the missing-clock route.")
    weight = float(config["items"]["density_classifier"]["mixture_weight"])
    mixture = np.asarray(density_mixture if "density_classifier" in items else v9_mixture, dtype=float)
    values[rows] = (1 - weight) * values[rows] + weight * mixture[eligible]
    counts = {"mixture_rows": int(eligible.sum()), "population_rows": int(len(population_index))}
    if "cat_long_25" in items:
        share = float(config["items"]["cat_long_25"]["weight"])
        blend = np.asarray(cat_long, dtype=float)
        if blend.shape != values.shape or not np.isfinite(blend[clock]).all():
            raise ValueError("cat_long predictions must cover every clock row.")
        values[clock] = (1 - share) * values[clock] + share * blend[clock]
        counts["cat_long_rows"] = int(clock.sum())
    if "route_negative_proxy" in items:
        replacement = np.asarray(fallback, dtype=float)
        if replacement.shape != values.shape or not np.isfinite(replacement[negative_proxy]).all():
            raise ValueError("Fallback predictions must cover every routed row.")
        values[negative_proxy] = replacement[negative_proxy]
        counts["routed_rows"] = int(negative_proxy.sum())
    floored = values < floor
    values[floored] = floor
    counts["floored_rows"] = int(floored.sum())
    if not np.isfinite(values).all():
        raise ValueError("Every prediction must be finite.")
    return values, counts


def select_candidate(candidates: dict, folds: list[str], margin: float) -> str | None:
    """Declared rule: at least `margin` better on each fold and a paired-day interval below zero on each fold."""
    for item in candidates.values():
        checks = []
        for fold in folds:
            row = item["folds"][fold]
            ratio = row["rmse_ratio_to_v9"]
            upper = row["paired_days"]["rmse_difference_95pct_interval_seconds"][1]
            checks.append(np.isfinite(ratio) and ratio <= 1 - margin and upper < 0)
        item["eligible"] = bool(all(checks))
        item["mean_relative_rmse"] = float(np.mean([item["folds"][fold]["rmse_ratio_to_v9"] for fold in folds]))
    eligible = [name for name, item in candidates.items() if item["eligible"]]
    return min(eligible, key=lambda name: (candidates[name]["mean_relative_rmse"], name)) if eligible else None
