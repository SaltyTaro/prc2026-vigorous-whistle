"""Portable new residual nodes composed with verified frozen native pipelines."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json, verify_receipts
from prc2026.features import ID
from prc2026_models.experiments import load_model, predict
from prc2026_phase4.features import prediction_offset
from prc2026_phase5.inference import iter_nodes, node_prediction as legacy_node, pipeline_prediction as legacy_pipeline, validate_weights
from .baseline import apply_map, center_scale
from .encoding import assemble, transform
from .features import CORE_NAMES


def copy_legacy_pipeline(destination: Path, source: Path, prefix: str) -> dict:
    """Preserve references to deduplicated native models in the frozen v4 refit."""
    verify_receipts(source, read_json(source/"run.json")["artifacts"])
    pipeline = deepcopy(read_json(source/"pipeline.json"))
    copied = {}
    for _, node in iter_nodes(pipeline):
        for key in ["model", "encoder", "history", "importance"]:
            if key not in node:
                continue
            old = node[key]
            new = prefix+"_"+old
            if Path(old).name != old or Path(new).name != new:
                raise ValueError("Imported component names must be flat relative files.")
            if old not in copied:
                if (destination/new).exists():
                    raise ValueError("Imported artifacts must not overwrite existing files.")
                shutil.copyfile(source/old, destination/new)
                copied[old] = new
            node[key] = copied[old]
    return pipeline


def route_mask(features: pd.DataFrame, route: str) -> np.ndarray:
    if route == "all":
        return np.ones(len(features), dtype=bool)
    if route == "nm_missing":
        return features["nm_missing"].eq(1).to_numpy()
    raise ValueError("Unknown new specialist route.")


def node_prediction(run: Path, node: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    baseline = apply_map(features, read_json(run/node["baseline_path"])) if node.get("baseline_path") else None
    encoded = transform(assemble(features, node["spec"], baseline), read_json(run/node["encoder_path"]))
    center, scale, _ = center_scale(baseline, node["spec"]["target"], len(features))
    values = predict(load_model(run/node["model_path"], "lightgbm"), encoded, "lightgbm", threads)
    return values*scale+center+prediction_offset(features, "lirf_missing_schedule")


def airports_prediction(run: Path, pipeline: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    old = features[[ID, *CORE_NAMES]]
    result = np.full(len(features), np.nan)
    assigned = np.zeros(len(features), dtype=bool)
    for patch in pipeline["patches"]:
        if patch["kind"] != "model" or not patch["route"].startswith("airport:"):
            raise ValueError("Expected the frozen ten airport models.")
        mask = old["airport"].eq(patch["route"].split(":", 1)[1]).to_numpy()
        if np.any(assigned & mask):
            raise ValueError("Overlapping airport routes.")
        if mask.any():
            result[mask] = legacy_node(run, pipeline["nodes"][patch["node"]], old.loc[mask].reset_index(drop=True), threads)
        assigned |= mask
    if not assigned.all() or not np.isfinite(result).all():
        raise ValueError("Airport reference must cover every row.")
    return result


def pipeline_prediction(run: Path, pipeline: dict, features: pd.DataFrame, threads: int) -> np.ndarray:
    kind = pipeline["kind"]
    if kind == "new_node":
        result = node_prediction(run, pipeline["node"], features, threads)
    elif kind == "legacy":
        result = legacy_pipeline(run, pipeline["pipeline"], features[[ID, *CORE_NAMES]], threads)
    elif kind == "legacy_airports":
        result = airports_prediction(run, pipeline["pipeline"], features, threads)
    elif kind == "mixture":
        validate_weights([m["weight"] for m in pipeline["members"]])
        result = np.zeros(len(features))
        for member in pipeline["members"]:
            result += member["weight"]*pipeline_prediction(run, member["pipeline"], features, threads)
    elif kind == "route_patch":
        result = pipeline_prediction(run, pipeline["base"], features, threads)
        mask = route_mask(features, pipeline["route"])
        weight = pipeline["weight"]
        if not 0 <= weight <= 1:
            raise ValueError("Patch must use a convex weight.")
        if mask.any():
            result[mask] = (1-weight)*result[mask]+weight*pipeline_prediction(
                run, pipeline["replacement"], features.loc[mask].reset_index(drop=True), threads)
    else:
        raise ValueError("Unknown new pipeline kind.")
    if not np.isfinite(result).all() or result.shape != (len(features),):
        raise ValueError("Every pipeline must produce a finite prediction for every requested row.")
    return result
