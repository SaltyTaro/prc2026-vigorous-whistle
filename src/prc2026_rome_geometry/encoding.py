"""Matched inherited predictors plus declared numeric geometry ablations."""

import numpy as np
import pandas as pd
from prc2026.features import ID
from .common import PREVIOUS_NAMES, CATEGORIES
from .features import STATIC_NAMES, QUEUE_NAMES

MISSING = "__MISSING__"


def contract(variant):
    additions = {"control":[], "geometry":STATIC_NAMES, "geometry_queue":STATIC_NAMES + QUEUE_NAMES}
    if variant not in additions:
        raise ValueError("Undeclared Rome variant.")
    return list(PREVIOUS_NAMES) + additions[variant], list(CATEGORIES)


def assemble(features, variant):
    names, cats = contract(variant)
    data = features[[ID, *names]].reset_index(drop=True)
    ids = data[ID].to_numpy(dtype=float)
    if not len(data) or not np.isfinite(ids).all() or np.any(ids % 1) or data[ID].duplicated().any():
        raise ValueError("Invalid model feature IDs.")
    if np.isinf(data[[name for name in names if name not in cats]].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Infinite model features.")
    return data


def fit_encoder(features, variant):
    data = assemble(features, variant)
    names, cats = contract(variant)
    vocab = {name:sorted(set(data[name].astype("string").fillna(MISSING)) | {MISSING}) for name in cats}
    return {"kind":"lightgbm", "variant":variant, "feature_names":names, "categorical_columns":cats, "vocabularies":vocab}


def transform(features, encoder):
    data = assemble(features, encoder["variant"])
    names, cats = contract(encoder["variant"])
    if names != encoder["feature_names"] or cats != encoder["categorical_columns"]:
        raise ValueError("Saved predictor contract changed.")
    result = {}
    for name in names:
        if name in cats:
            values = data[name].astype("string").fillna(MISSING)
            values = values.where(values.isin(encoder["vocabularies"][name]), MISSING)
            result[name] = pd.Categorical(values, categories=encoder["vocabularies"][name])
        else:
            result[name] = data[name].to_numpy(dtype="float32", na_value=np.nan)
    return pd.DataFrame(result)
