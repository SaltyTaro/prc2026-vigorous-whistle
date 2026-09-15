"""Native clock models with complete, portable v5 fallback predictions."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import shutil
import numpy as np

from prc2026.artifacts import read_json, verify_receipts
from prc2026_models.experiments import load_model, predict
from prc2026_phase5.inference import validate_weights
from prc2026_weather.inference import pipeline_prediction as weather_prediction
from .encoding import transform
from .features import route_mask, target_offset


def node_prediction(run: Path, node: dict, features, threads: int):
    if not route_mask(features).all():
        raise ValueError("Clock nodes must be evaluated only on their declared route.")
    encoded = transform(features, read_json(run/node["encoder_path"]))
    values = predict(load_model(run/node["model_path"], node["model_kind"]), encoded, node["model_kind"], threads)
    return values + target_offset(features, node["spec"]["target"])


def copy_file_tree(destination: Path, source: Path, tree: dict, prefix: str) -> dict:
    """Copy only explicitly referenced native files, preserving shared references."""
    copied = {}
    keys = {"model", "encoder", "history", "importance", "model_path", "encoder_path", "history_path", "importance_path"}
    result = deepcopy(tree)
    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for key, child in value.items():
                if key in keys and isinstance(child, str):
                    if Path(child).name != child or not (source/child).is_file():
                        raise ValueError("Native references must be existing flat files.")
                    if child not in copied:
                        name = prefix+child
                        if Path(name).name != name or (destination/name).exists():
                            raise ValueError("Native copies must not overwrite or escape the run.")
                        shutil.copyfile(source/child, destination/name)
                        copied[child] = name
                    value[key] = copied[child]
                else:
                    visit(child)
    visit(result)
    return result


def copy_v5_pipeline(destination: Path, source: Path) -> dict:
    verify_receipts(source, read_json(source/"run.json")["artifacts"])
    return copy_file_tree(destination, source, read_json(source/"pipeline.json"), "v5_")


def pipeline_prediction(run: Path, pipeline: dict, features, threads: int):
    kind = pipeline["kind"]
    if kind == "nmclock_node":
        values = node_prediction(run, pipeline["node"], features, threads)
    elif kind == "nmclock_airports":
        values = np.full(len(features), np.nan)
        for airport, node in pipeline["nodes"].items():
            mask = features["airport"].eq(airport).to_numpy()
            if mask.any():
                values[mask] = node_prediction(run, node, features.loc[mask].reset_index(drop=True), threads)
    elif kind == "v5":
        values = weather_prediction(run, pipeline["pipeline"], features, threads)
    elif kind == "mixture":
        validate_weights([m["weight"] for m in pipeline["members"]])
        values = np.zeros(len(features))
        for member in pipeline["members"]:
            if member["weight"]:
                values += member["weight"]*pipeline_prediction(run, member["pipeline"], features, threads)
    elif kind == "nmclock_patch":
        weight = pipeline["weight"]
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("The clock patch requires a declared convex weight.")
        mask = route_mask(features)
        if weight == 1:
            values = np.empty(len(features))
            if (~mask).any():
                values[~mask] = pipeline_prediction(run, pipeline["base"], features.loc[~mask].reset_index(drop=True), threads)
        else:
            values = pipeline_prediction(run, pipeline["base"], features, threads)
        if mask.any():
            replacement = pipeline_prediction(run, pipeline["replacement"], features.loc[mask].reset_index(drop=True), threads)
            values[mask] = replacement if weight == 1 else (1-weight)*values[mask]+weight*replacement
    else:
        raise ValueError("Unknown NM clock pipeline kind.")
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Every requested row must have a finite prediction.")
    return values


def component_pipeline(global_node: dict, airports: dict) -> dict:
    return {"kind": "mixture", "members": [
        {"weight": .5, "pipeline": {"kind": "nmclock_node", "node": global_node}},
        {"weight": .5, "pipeline": {"kind": "nmclock_airports", "nodes": airports}},
    ]}
