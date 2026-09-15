"""Fit-only categorical vocabularies and strict feature/target alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import CATEGORY_SOURCES, FEATURE_NAMES, ID, NUMERIC_FEATURES, TARGET


def validate_features(frame: pd.DataFrame) -> None:
    if list(frame) != [ID, *FEATURE_NAMES]:
        raise ValueError("Model input must match the frozen feature contract exactly.")
    if frame.empty or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Empty features or duplicate/null movement IDs.")
    if np.isinf(frame[NUMERIC_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric feature.")


def aligned_target(features: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    validate_features(features)
    if len(truth) != len(features) or truth[ID].isna().any() or truth[ID].duplicated().any():
        raise ValueError("Target population must match feature rows one-to-one.")
    joined = features[[ID]].merge(truth[[ID, TARGET]], on=ID, how="left",
                                  validate="one_to_one", sort=False)
    values = joined[TARGET].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Unaligned or nonfinite training labels.")
    return values


def fit_encoder(features: pd.DataFrame, kind: str) -> dict:
    validate_features(features)
    if kind not in {"lightgbm", "catboost"}:
        raise ValueError("Unknown model kind.")
    vocabularies = {}
    if kind == "lightgbm":
        for name in CATEGORY_SOURCES:
            vocabularies[name] = sorted(features[name].astype("string")
                                       .fillna("__MISSING__").unique().tolist())
    return {"kind": kind, "feature_names": FEATURE_NAMES, "categorical_columns": list(CATEGORY_SOURCES),
            "vocabularies": vocabularies,
            "unknown_category_policy": "missing" if kind == "lightgbm" else "native_catboost",
            "fitted_on": "fitting_partition_only"}


def transform(features: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    validate_features(features)
    if encoder["feature_names"] != FEATURE_NAMES or encoder["categorical_columns"] != list(CATEGORY_SOURCES):
        raise ValueError("Saved encoder does not match the feature contract.")
    out = features[FEATURE_NAMES].copy()
    for name in CATEGORY_SOURCES:
        values = features[name].astype("string").fillna("__MISSING__")
        if encoder["kind"] == "lightgbm":
            out[name] = pd.Categorical(values, categories=encoder["vocabularies"][name])
        elif encoder["kind"] == "catboost":
            out[name] = values.astype(object)
        else:
            raise ValueError("Unknown model encoder.")
    for name in NUMERIC_FEATURES:
        out[name] = out[name].astype("float32")
    return out
