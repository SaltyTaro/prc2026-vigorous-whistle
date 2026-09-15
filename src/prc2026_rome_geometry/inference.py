"""Native predictions and exact preservation outside the declared Rome route."""

import numpy as np
from prc2026.artifacts import read_json
from prc2026_models.experiments import load_model, predict
from prc2026_nmclock.features import route_mask, target_offset
from .encoding import transform


def node_prediction(run, node, features, threads=16):
    if not features.airport.eq("LIRF").all() or not route_mask(features).all():
        raise ValueError("Rome components require clock-present LIRF rows.")
    encoder = read_json(run / node["encoder_path"])
    native = load_model(run / node["model_path"], "lightgbm")
    return predict(native, transform(features, encoder), "lightgbm", threads) + target_offset(features, "nm_residual")


def compose(reference, original_airport, new_airport, eligible, weight):
    arrays = [np.asarray(value, dtype=float) for value in [reference, original_airport, new_airport]]
    eligible = np.asarray(eligible)
    if eligible.dtype != bool or arrays[0].ndim != 1 or any(value.shape != arrays[0].shape for value in arrays[1:]) or eligible.shape != arrays[0].shape:
        raise ValueError("Composition inputs must have identical shapes and a boolean route.")
    if weight not in [.5, 1.] or not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Use a declared weight and complete finite prediction vectors.")
    result = arrays[0].copy()
    result[eligible] += .5 * weight * (arrays[2][eligible] - arrays[1][eligible])
    if not np.isfinite(result).all():
        raise ValueError("A complete candidate is nonfinite.")
    return result
