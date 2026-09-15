"""Covariate-only target scales with an exact squared-error identity."""
import numpy as np
from prc2026_ladder.inference import route_mask
from prc2026_phase4.features import prediction_offset


def validate_transform(variant):
    spec = variant["target_transform"]
    if spec["weight_unit_seconds"] != 300.0:
        raise ValueError("The declared weight unit is 300 seconds.")
    if spec["kind"] == "clock_disagreement":
        if variant["route"] != "clock_available" or spec["minimum_seconds"] not in [60., 300.]:
            raise ValueError("Undeclared NM disagreement scale.")
    elif spec["kind"] == "schedule_interval":
        if variant["route"] != "clock_missing" or spec["minimum_seconds"] not in [300., 900.]:
            raise ValueError("Undeclared missing-clock schedule scale.")
    else:
        raise ValueError("Unknown target scale.")


def offset_scale(features, variant):
    validate_transform(variant)
    if not route_mask(features, variant["route"]).all():
        raise ValueError("Scaled nodes require their complete declared covariate route.")
    spec = variant["target_transform"]
    if spec["kind"] == "clock_disagreement":
        offset = features["nmclock_proxy_s"].to_numpy(dtype=float)
        source = features["nmclock_clock_difference_s"].to_numpy(dtype=float)
    else:
        offset = prediction_offset(features, "lirf_missing_schedule")
        source = features["schedule_to_movement_s"].to_numpy(dtype=float)
    if np.isinf(source).any() or not np.isfinite(offset).all():
        raise ValueError("Nonfinite clock offsets or infinite scale sources.")
    scale = np.maximum(spec["minimum_seconds"], np.where(np.isfinite(source), np.abs(source), spec["minimum_seconds"]))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Every row needs a positive finite scale.")
    return offset, scale


def target_and_weight(features, target, variant):
    offset, scale = offset_scale(features, variant)
    target = np.asarray(target, dtype=float)
    if target.shape != offset.shape or not np.isfinite(target).all():
        raise ValueError("Retain every finite target on the routed population.")
    return (target-offset)/scale, np.square(scale/variant["target_transform"]["weight_unit_seconds"])


def physical_rmse(prediction, target, offset, scale):
    return float(np.sqrt(np.mean(np.square(np.asarray(target)-offset-scale*np.asarray(prediction)))))

