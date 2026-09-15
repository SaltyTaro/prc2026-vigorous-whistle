"""Native mixed-model inference with strict IDs and convex ensemble weights."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json, verify_receipts
from prc2026.features import ID
from prc2026_models.experiments import load_model, predict
from prc2026_phase4.encoding import contract, fit_encoder, validate_features
from prc2026_phase4.features import prediction_offset, route_mask
from prc2026_phase4.training import node_prediction as lgb_prediction


def fit_cat_encoder(features: pd.DataFrame) -> dict:
    encoder = fit_encoder(features, "phase4")
    encoder["kind"] = "catboost"
    return encoder


def cat_transform(features: pd.DataFrame, encoder: dict) -> pd.DataFrame:
    names, categories = validate_features(features, "phase4")
    if (encoder["kind"] != "catboost" or encoder["schema"] != "phase4"
            or encoder["feature_names"] != names or encoder["categorical_columns"] != categories):
        raise ValueError("CatBoost encoder contract mismatch.")
    out = features[names].copy()
    for name in categories:
        values = features[name].astype("string").fillna("__MISSING__")
        out[name] = values.where(values.isin(encoder["vocabularies"][name]), "__MISSING__").astype(object)
    for name in names:
        if name not in categories:
            out[name] = out[name].astype("float32")
    return out


def node_prediction(run: Path, node: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    kind = node.get("model_kind", "lightgbm")
    if kind == "lightgbm":
        return lgb_prediction(run, node, features, threads)
    if kind != "catboost":
        raise ValueError("Unknown component model kind.")
    names, _ = contract(node["schema"])
    x = cat_transform(features[[ID, *names]], read_json(run / node["encoder"]))
    model = load_model(run / node["model"], "catboost")
    return predict(model, x, "catboost", threads) + prediction_offset(features, node["offset"])


def validate_weights(weights: list[float]) -> None:
    values = np.asarray(weights, dtype=float)
    if not len(values) or not np.isfinite(values).all() or (values < 0).any() or abs(values.sum()-1.0) > 1e-12:
        raise ValueError("Ensemble weights must be finite, nonnegative, and sum to one.")


def combine_predictions(predictions: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.DataFrame:
    validate_weights(list(weights.values()))
    if set(weights) - set(predictions):
        raise ValueError("Ensemble member predictions are missing.")
    first = predictions[next(iter(weights))]
    if first[ID].duplicated().any() or first[ID].isna().any():
        raise ValueError("Ensemble member IDs must be unique and nonnull.")
    result = first[[ID]].copy()
    values = np.zeros(len(result))
    for name, weight in weights.items():
        member = predictions[name]
        if len(member) != len(result) or member[ID].duplicated().any() or set(member[ID]) != set(result[ID]):
            raise ValueError("All ensemble members must cover the exact same IDs.")
        aligned = result.merge(member[[ID, "prediction"]], on=ID, how="left", validate="one_to_one", sort=False)
        numbers = aligned["prediction"].to_numpy(dtype=float)
        if not np.isfinite(numbers).all():
            raise ValueError("Nonfinite ensemble member prediction.")
        values += weight*numbers
    result["prediction"] = values
    return result


def pipeline_prediction(run: Path, pipeline: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    if pipeline.get("kind") == "mixture":
        validate_weights([m["weight"] for m in pipeline["members"]])
        values = np.zeros(len(features))
        for member in pipeline["members"]:
            values += member["weight"]*pipeline_prediction(run, member["pipeline"], features, threads)
    else:
        nodes = pipeline["nodes"]
        values = node_prediction(run, nodes[pipeline["default"]], features, threads)
        for patch in pipeline["patches"]:
            if patch["kind"] != "model" or not 0 <= patch["weight"] <= 1:
                raise ValueError("Phase-5 routes must be convex model patches.")
            mask = route_mask(features, patch["route"])
            if mask.any():
                replacement = node_prediction(run, nodes[patch["node"]], features.loc[mask], threads)
                values[mask] = (1-patch["weight"])*values[mask] + patch["weight"]*replacement
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("The model must predict every input row.")
    return values


def iter_nodes(pipeline: dict, prefix: str = ""):
    if pipeline.get("kind") == "mixture":
        for member in pipeline["members"]:
            yield from iter_nodes(member["pipeline"], prefix + member["name"] + "/")
    else:
        for name, node in pipeline["nodes"].items():
            yield prefix + name, node


def copy_pipeline(destination: Path, source: Path, prefix: str = "") -> dict:
    report = read_json(source / "run.json")
    verify_receipts(source, report["artifacts"])
    pipeline = deepcopy(read_json(source / "pipeline.json"))
    for _, node in iter_nodes(pipeline):
        for key in ["model", "encoder", "history", "importance"]:
            if key in node:
                old = node[key]
                new = prefix + old
                if Path(old).name != old or Path(new).name != new:
                    raise ValueError("Component files must be flat, relative filenames.")
                target = destination / new
                if target.exists():
                    raise ValueError("Imported model components must not overwrite artifacts.")
                shutil.copyfile(source / old, target)
                node[key] = new
    return pipeline
