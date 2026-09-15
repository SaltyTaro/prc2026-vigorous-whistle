"""Native weather models and a portable composition with the unchanged v4 models."""
from pathlib import Path
import numpy as np
from prc2026.artifacts import read_json
from prc2026.features import ID
from prc2026_models.experiments import load_model, predict
from prc2026_phase4.features import prediction_offset
from prc2026_phase5.inference import pipeline_prediction as legacy_prediction, validate_weights
from prc2026_phase5r.inference import airports_prediction, copy_legacy_pipeline
from .encoding import transform
from .features import CORE_NAMES


def node_prediction(run: Path, node: dict, features, threads: int):
    encoded = transform(features, read_json(run/node["encoder_path"]))
    values = predict(load_model(run/node["model_path"], node["model_kind"]), encoded, node["model_kind"], threads)
    return values+prediction_offset(features, "lirf_missing_schedule")


def pipeline_prediction(run: Path, pipeline: dict, features, threads: int):
    if pipeline["kind"] == "weather":
        values = node_prediction(run, pipeline["node"], features, threads)
    elif pipeline["kind"] == "legacy":
        values = legacy_prediction(run, pipeline["pipeline"], features[[ID, *CORE_NAMES]], threads)
    elif pipeline["kind"] == "airports":
        values = airports_prediction(run, pipeline["pipeline"], features, threads)
    elif pipeline["kind"] == "mixture":
        validate_weights([m["weight"] for m in pipeline["members"]])
        values = np.zeros(len(features))
        for member in pipeline["members"]:
            if member["weight"]:
                values += member["weight"]*pipeline_prediction(run, member["pipeline"], features, threads)
    else:
        raise ValueError("Unknown weather pipeline kind.")
    if values.shape != (len(features),) or not np.isfinite(values).all():
        raise ValueError("Every requested row needs a finite prediction.")
    return values
