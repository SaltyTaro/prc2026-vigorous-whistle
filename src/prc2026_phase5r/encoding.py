"""Fit-only encoding with an explicit contract for each declared comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID
from .baseline import BASELINE_NAMES
from .features import CORE_NAMES, CORE_CATEGORIES, FEATURE_NAMES, CATEGORICAL_COLUMNS

MISSING = "__MISSING__"


def contract(spec: dict):
    if spec["features"] not in {"core", "ground"} or spec["target"] not in {"none", "baseline", "baseline_scaled"}:
        raise ValueError("Unknown new feature/target contract.")
    names, cats = (list(CORE_NAMES), list(CORE_CATEGORIES)) if spec["features"] == "core" else (list(FEATURE_NAMES), list(CATEGORICAL_COLUMNS))
    if spec["target"] != "none":
        names += BASELINE_NAMES
    return names, cats


def assemble(features: pd.DataFrame, spec: dict, baseline: pd.DataFrame | None = None) -> pd.DataFrame:
    names, _ = contract(spec)
    raw_names = [n for n in names if n not in BASELINE_NAMES]
    out = features[[ID, *raw_names]].reset_index(drop=True).copy()
    if spec["target"] != "none":
        if baseline is None or len(baseline) != len(features) or list(baseline) != BASELINE_NAMES:
            raise ValueError("Aligned baseline features are required.")
        out[BASELINE_NAMES] = baseline.reset_index(drop=True)
    validate_features(out, names)
    return out


def validate_features(frame: pd.DataFrame, names: list[str]):
    ids = frame[ID].to_numpy(dtype=float)
    if list(frame) != [ID, *names] or not len(frame):
        raise ValueError("Feature columns differ from the explicit model contract.")
    if not np.isfinite(ids).all() or (ids % 1 != 0).any() or frame[ID].duplicated().any():
        raise ValueError("Invalid feature IDs.")
    numeric = [n for n in names if n not in CATEGORICAL_COLUMNS]
    if np.isinf(frame[numeric].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Infinite numeric features.")


def fit_encoder(frame: pd.DataFrame, spec: dict) -> dict:
    names, cats = contract(spec)
    validate_features(frame, names)
    vocab = {name: sorted(set(frame[name].astype("string").fillna(MISSING)) | {MISSING}) for name in cats}
    return {"kind": "lightgbm", "feature_names": names, "categorical_columns": cats, "vocabularies": vocab, "spec": spec}


def transform(frame: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    names = encoder["feature_names"]
    expected, cats = contract(encoder["spec"])
    if names != expected or encoder["categorical_columns"] != cats:
        raise ValueError("Encoder contract changed.")
    validate_features(frame, names)
    out = {}
    for name in names:
        if name in cats:
            values = frame[name].astype("string").fillna(MISSING)
            values = values.where(values.isin(encoder["vocabularies"][name]), MISSING)
            out[name] = pd.Categorical(values, categories=encoder["vocabularies"][name])
        else:
            out[name] = frame[name].to_numpy(dtype="float32", na_value=np.nan)
    return pd.DataFrame(out, index=np.arange(len(frame)))
