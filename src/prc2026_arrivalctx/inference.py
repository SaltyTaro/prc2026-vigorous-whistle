"""Native arrival-context experts for the missing-clock route."""
from pathlib import Path
import numpy as np
from prc2026.artifacts import read_json
from prc2026_models.experiments import load_model, predict
from prc2026_phase4.features import prediction_offset
from prc2026_nmclock.features import route_mask as clock_available
from .encoding import transform


def route_mask(features):
    return ~clock_available(features)


def node_prediction(run: Path, node: dict, features, threads: int):
    if node["schema"] != "arrivalctx" or not route_mask(features).all():
        raise ValueError("Arrival experts require their declared missing-clock route.")
    encoded = transform(features, read_json(run / node["encoder_path"]))
    values = predict(load_model(run / node["model_path"], node["model_kind"]), encoded, node["model_kind"], threads)
    values = values + prediction_offset(features, "lirf_missing_schedule")
    if not np.isfinite(values).all():
        raise ValueError("Every requested row needs a finite prediction.")
    return values
