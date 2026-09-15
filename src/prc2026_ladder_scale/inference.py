"""Portable scaled native nodes combined with unchanged v6 components."""
from pathlib import Path
import numpy as np
from prc2026.artifacts import read_json
from prc2026_models.experiments import load_model, predict
from prc2026_nmclock import encoding as clock_encoding
from prc2026_weather import encoding as weather_encoding
from prc2026_ladder.inference import (
    compose, recipe_pipeline, route_mask, pipeline_prediction as prior_prediction,
)
from prc2026_phase5.inference import validate_weights
from .targets import offset_scale


def encoding(schema):
    if schema == "nmclock":
        return clock_encoding
    if schema == "weather":
        return weather_encoding
    raise ValueError("Unknown frozen feature schema.")


def node_variant(node):
    return {**node["spec"], "target_transform": node["target_transform"]}


def node_prediction(run: Path, node: dict, features, threads: int):
    offset, scale = offset_scale(features, node_variant(node))
    encoder = read_json(run/node["encoder_path"])
    values = predict(load_model(run/node["model_path"], "lightgbm"),
                     encoding(node["schema"]).transform(features, encoder), "lightgbm", threads)
    return offset+scale*values


def pipeline_prediction(run: Path, pipeline: dict, features, threads: int):
    kind = pipeline["kind"]
    if kind == "scaled_node":
        values = node_prediction(run, pipeline["node"], features, threads)
    elif kind == "mixture":
        validate_weights([part["weight"] for part in pipeline["members"]])
        values = np.zeros(len(features))
        for part in pipeline["members"]:
            if part["weight"]:
                values += part["weight"]*pipeline_prediction(run, part["pipeline"], features, threads)
    elif kind == "ladder_recipe":
        weight = pipeline["weight"]
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("The declared recipe must be convex.")
        if set(pipeline["routes"]) != {"clock_available", "clock_missing"}:
            raise ValueError("Both complementary covariate routes are required.")
        baseline = None if weight == 1 else pipeline_prediction(run, pipeline["base"], features, threads)
        values = np.empty(len(features))
        for route, replacement in pipeline["routes"].items():
            mask = route_mask(features, route)
            if mask.any():
                if weight != 1 and replacement == pipeline["base"]:
                    values[mask] = baseline[mask]
                else:
                    updated = pipeline_prediction(run, replacement, features.loc[mask].reset_index(drop=True), threads)
                    values[mask] = updated if weight == 1 else (1-weight)*baseline[mask]+weight*updated
    else:
        values = prior_prediction(run, pipeline, features, threads)
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Every requested row needs a finite physical prediction.")
    return values

