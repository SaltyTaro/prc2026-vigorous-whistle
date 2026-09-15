"""Native routed components and complete, portable ladder recipes."""
from pathlib import Path
import numpy as np

from prc2026_nmclock.features import route_mask as clock_available
from prc2026_nmclock.inference import (
    node_prediction as clock_prediction, pipeline_prediction as frozen_prediction,
)
from prc2026_weather.inference import node_prediction as weather_prediction
from prc2026_phase5.inference import validate_weights


def route_mask(features, route):
    available = clock_available(features)
    if route == "clock_available":
        return available
    if route == "clock_missing":
        return ~available
    raise ValueError("Unknown declared clock route.")


def node_prediction(run, node, features, threads):
    if not route_mask(features, node["spec"]["route"]).all():
        raise ValueError("A native component received rows outside its fitted route.")
    if node["schema"] == "nmclock":
        values = clock_prediction(run, node, features, threads)
    elif node["schema"] == "weather":
        values = weather_prediction(run, node, features, threads)
    else:
        raise ValueError("Unknown native component schema.")
    return values


def compose(reference, components, available, family, weight):
    """Combine aligned full populations using only declared covariate routing."""
    reference = np.asarray(reference, dtype=float)
    available = np.asarray(available)
    if reference.ndim != 1 or available.shape != reference.shape or available.dtype != bool:
        raise ValueError("Recipe inputs must be aligned vectors and boolean routes.")
    if not np.isfinite(reference).all() or not np.isfinite(weight) or not 0 < weight <= 1:
        raise ValueError("Recipe needs finite reference predictions and a convex weight.")
    if set(family) != {"clock_available", "clock_missing"}:
        raise ValueError("Both complementary clock routes are required.")
    candidate = np.empty(len(reference), dtype=float)
    for route, mask in [("clock_available", available), ("clock_missing", ~available)]:
        weights = family[route]
        validate_weights(list(weights.values()))
        candidate[mask] = 0.
        for name, fraction in weights.items():
            values = reference if name == "v6" else np.asarray(components[name], dtype=float)
            if values.shape != reference.shape or not np.isfinite(values).all():
                raise ValueError("A component does not cover the complete scoring population.")
            candidate[mask] += fraction * values[mask]
    result = candidate if weight == 1 else (1-weight)*reference + weight*candidate
    # Preserve untouched routes bit for bit, including at fractional blend weights.
    for route, mask in [("clock_available", available), ("clock_missing", ~available)]:
        if family[route] == {"v6": 1.0}:
            result[mask] = reference[mask]
    return result


def pipeline_prediction(run: Path, pipeline: dict, features, threads: int):
    kind = pipeline["kind"]
    if kind == "frozen_nmclock":
        values = frozen_prediction(run, pipeline["pipeline"], features, threads)
    elif kind == "node":
        values = node_prediction(run, pipeline["node"], features, threads)
    elif kind == "mixture":
        validate_weights([member["weight"] for member in pipeline["members"]])
        values = np.zeros(len(features))
        for member in pipeline["members"]:
            if member["weight"]:
                values += member["weight"] * pipeline_prediction(run, member["pipeline"], features, threads)
    elif kind == "ladder_recipe":
        weight = pipeline["weight"]
        if not np.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("Recipe weight must be convex.")
        if set(pipeline["routes"]) != {"clock_available", "clock_missing"}:
            raise ValueError("The recipe must cover both routes.")
        baseline = None if weight == 1 else pipeline_prediction(run, pipeline["base"], features, threads)
        values = np.empty(len(features))
        for route, replacement in pipeline["routes"].items():
            mask = route_mask(features, route)
            if mask.any():
                selected = features.loc[mask].reset_index(drop=True)
                prediction = pipeline_prediction(run, replacement, selected, threads)
                if weight == 1:
                    values[mask] = prediction
                elif replacement == pipeline["base"]:
                    values[mask] = baseline[mask]
                else:
                    values[mask] = (1-weight)*baseline[mask] + weight*prediction
    else:
        raise ValueError("Unknown ladder pipeline kind.")
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Every requested row must have a finite prediction.")
    return values


def recipe_pipeline(family, weight, sources):
    routes = {}
    for route, weights in family.items():
        routes[route] = (sources["v6"] if weights == {"v6": 1.0} else
                         {"kind": "mixture", "members": [
                             {"weight": value, "pipeline": sources[name]}
                             for name, value in weights.items()]})
    return {"kind": "ladder_recipe", "weight": weight, "base": sources["v6"], "routes": routes}
