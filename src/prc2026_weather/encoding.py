"""Explicit weather schemas and fit-only categorical vocabularies."""
import numpy as np
import pandas as pd
from prc2026.features import ID
from .features import CORE_NAMES, CORE_CATEGORIES, FEATURE_NAMES, CATEGORICAL_COLUMNS, WEATHER_NAMES

MISSING = "__MISSING__"


def contract(spec):
    if spec["features"] == "core_weather":
        return list(CORE_NAMES)+WEATHER_NAMES, list(CORE_CATEGORIES)
    if spec["features"] == "ground_weather":
        return list(FEATURE_NAMES), list(CATEGORICAL_COLUMNS)
    raise ValueError("Unknown weather model feature schema.")


def assemble(features, spec):
    names, cats = contract(spec)
    frame = features[[ID, *names]].reset_index(drop=True)
    ids = frame[ID].to_numpy(dtype=float)
    if not len(frame) or not np.isfinite(ids).all() or np.any(ids % 1) or frame[ID].duplicated().any():
        raise ValueError("Invalid model feature identities.")
    if np.isinf(frame[[n for n in names if n not in cats]].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Infinite model features.")
    return frame


def fit_encoder(features, spec):
    frame = assemble(features, spec)
    names, cats = contract(spec)
    if spec["kind"] not in {"lightgbm", "catboost"}:
        raise ValueError("Unknown model family.")
    vocab = {n: sorted(set(frame[n].astype("string").fillna(MISSING)) | {MISSING}) for n in cats}
    return {"kind": spec["kind"], "spec": spec, "feature_names": names, "categorical_columns": cats, "vocabularies": vocab}


def transform(features, encoder):
    frame = assemble(features, encoder["spec"])
    names, cats = contract(encoder["spec"])
    if names != encoder["feature_names"] or cats != encoder["categorical_columns"]:
        raise ValueError("Encoder feature contract changed.")
    out = {}
    for name in names:
        if name in cats:
            values = frame[name].astype("string").fillna(MISSING)
            values = values.where(values.isin(encoder["vocabularies"][name]), MISSING)
            out[name] = (pd.Categorical(values, categories=encoder["vocabularies"][name]) if encoder["kind"] == "lightgbm"
                         else values.to_numpy(dtype=object))
        else:
            out[name] = frame[name].to_numpy(dtype="float32", na_value=np.nan)
    return pd.DataFrame(out)
