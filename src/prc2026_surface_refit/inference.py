"""Selected timeline globals and airport models, with the exact v6 fallback."""
from copy import deepcopy
import numpy as np

from prc2026_nmclock.features import route_mask
from prc2026_nmclock.inference import pipeline_prediction as frozen_prediction
from prc2026_timeline.inference import node_prediction
from prc2026_surface_airports.inference import airport_prediction


SELECTED = "full_timeline_v6_100"
WEIGHTS = {"cat_combined": .25, "lgb_combined": .25, "surface_airports": .5}


def make_pipeline(nodes, reference):
    return {"kind": "surface_refit_v1", "selected_candidate": SELECTED,
            "new_weight": 1., "clock_available_weights": dict(WEIGHTS),
            "clock_missing": "frozen_v6", "v6": deepcopy(reference),
            "globals": deepcopy(nodes["globals"]), "airports": deepcopy(nodes["airports"])}


def pipeline_prediction(run, pipeline, features, threads):
    if (pipeline["kind"] != "surface_refit_v1" or pipeline["selected_candidate"] != SELECTED
            or pipeline["new_weight"] != 1. or pipeline["clock_available_weights"] != WEIGHTS
            or pipeline["clock_missing"] != "frozen_v6"
            or set(pipeline["globals"]) != {"cat_combined", "lgb_combined"}):
        raise ValueError("The selected full-timeline recipe must remain unchanged.")
    mask = route_mask(features)
    values = np.empty(len(features), dtype=float)
    if (~mask).any():
        values[~mask] = frozen_prediction(run, pipeline["v6"], features.loc[~mask].reset_index(drop=True), threads)
    if mask.any():
        selected = features.loc[mask].reset_index(drop=True)
        prediction = np.zeros(len(selected), dtype=float)
        for name, weight in WEIGHTS.items():
            component = (airport_prediction(run, pipeline["airports"], selected, threads) if name == "surface_airports"
                         else node_prediction(run, pipeline["globals"][name], selected, threads))
            prediction += weight * component
        values[mask] = prediction
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Every requested departure needs a finite prediction.")
    return values
