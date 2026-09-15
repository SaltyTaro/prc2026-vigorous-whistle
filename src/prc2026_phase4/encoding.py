"""Fit-only categorical vocabularies for an explicit, ID-free predictor list."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, TARGET
from .features import CATEGORICAL_COLUMNS, FEATURE_NAMES, PREVIOUS_CATEGORIES, PREVIOUS_NAMES


def contract(schema: str) -> tuple[list[str], list[str]]:
    if schema == "phase4":
        return FEATURE_NAMES, CATEGORICAL_COLUMNS
    if schema == "phase3":
        return PREVIOUS_NAMES, PREVIOUS_CATEGORIES
    raise ValueError("Unknown model feature schema.")


def validate_features(features: pd.DataFrame, schema: str = "phase4") -> tuple[list[str], list[str]]:
    names, categories = contract(schema)
    if list(features) != [ID, *names]:
        raise ValueError("Model input differs from its feature contract.")
    ids = features[ID].to_numpy(dtype=float)
    if not len(features) or not np.isfinite(ids).all() or features[ID].duplicated().any() or (ids % 1 != 0).any():
        raise ValueError("Invalid feature ID population.")
    numeric = [n for n in names if n not in categories]
    if np.isinf(features[numeric].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric feature.")
    return names, categories


def aligned_target(features: pd.DataFrame, truth: pd.DataFrame) -> np.ndarray:
    if len(features) != len(truth) or truth[ID].isna().any() or truth[ID].duplicated().any() or features[ID].duplicated().any():
        raise ValueError("Targets must align one-to-one with the feature IDs.")
    aligned = features[[ID]].merge(truth[[ID, TARGET]], on=ID, how="left", validate="one_to_one", sort=False)
    values = aligned[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Unaligned or nonfinite target.")
    return values


def fit_encoder(features: pd.DataFrame, schema: str = "phase4") -> dict:
    names, categories = validate_features(features, schema)
    vocabularies = {n: sorted(features[n].astype("string").fillna("__MISSING__").unique().tolist()) for n in categories}
    return {"kind": "lightgbm", "schema": schema, "feature_names": names, "categorical_columns": categories,
            "vocabularies": vocabularies, "unknown_category_policy": "missing", "fitted_on": "fitting_partition_only"}


def transform(features: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    names, categories = validate_features(features, encoder["schema"])
    if encoder["feature_names"] != names or encoder["categorical_columns"] != categories or encoder["kind"] != "lightgbm":
        raise ValueError("Saved encoder schema changed.")
    out = features[names].copy()
    for name in categories:
        values = features[name].astype("string").fillna("__MISSING__")
        vocabulary = encoder["vocabularies"][name]
        out[name] = pd.Categorical(values.where(values.isin(vocabulary), pd.NA), categories=vocabulary)
    for name in names:
        if name not in categories:
            out[name] = out[name].astype("float32")
    return out
