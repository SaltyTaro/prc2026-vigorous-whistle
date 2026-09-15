"""Variant-specific, fit-only categorical encoding and ID-aligned labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, TARGET
from .features import feature_contract


def validate_features(frame: pd.DataFrame, families: list[str]) -> tuple[list[str], list[str]]:
    names, categories = feature_contract(families)
    if list(frame) != [ID, *names]:
        raise ValueError("Model input differs from its declared feature contract.")
    ids = frame[ID].to_numpy(dtype=float)
    if frame.empty or not np.isfinite(ids).all() or frame[ID].duplicated().any() or (ids % 1 != 0).any():
        raise ValueError("Invalid feature ID population.")
    numeric = [n for n in names if n not in categories]
    if np.isinf(frame[numeric].to_numpy(dtype=float)).any():
        raise ValueError("Infinite numeric feature.")
    return names, categories


def aligned_target(features: pd.DataFrame, truth: pd.DataFrame, families: list[str]) -> np.ndarray:
    validate_features(features, families)
    if len(truth) != len(features) or truth[ID].isna().any() or truth[ID].duplicated().any():
        raise ValueError("Labels must match features one-to-one.")
    joined = features[[ID]].merge(truth[[ID, TARGET]], on=ID, how="left", validate="one_to_one", sort=False)
    values = joined[TARGET].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Unaligned or nonfinite training target.")
    return values


def fit_encoder(features: pd.DataFrame, families: list[str]) -> dict:
    names, categories = validate_features(features, families)
    vocabularies = {name: sorted(features[name].astype("string").fillna("__MISSING__").unique().tolist())
                    for name in categories}
    return {"kind": "lightgbm", "families": families, "feature_names": names,
            "categorical_columns": categories, "vocabularies": vocabularies,
            "unknown_category_policy": "missing", "fitted_on": "fitting_partition_only"}


def transform(features: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    names, categories = validate_features(features, encoder["families"])
    if encoder["feature_names"] != names or encoder["categorical_columns"] != categories or encoder["kind"] != "lightgbm":
        raise ValueError("Saved encoder contract mismatch.")
    out = features[names].copy()
    for name in categories:
        values = features[name].astype("string").fillna("__MISSING__")
        vocabulary = encoder["vocabularies"][name]
        # Explicit masking preserves LightGBM's unknown-as-missing behavior
        # without relying on pandas' deprecated unknown-category constructor.
        values = values.where(values.isin(vocabulary), pd.NA)
        out[name] = pd.Categorical(values, categories=vocabulary)
    for name in names:
        if name not in categories:
            out[name] = out[name].astype("float32")
    return out
