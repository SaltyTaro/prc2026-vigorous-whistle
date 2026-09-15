"""Disjoint airport routing through the already verified timeline native format."""
import numpy as np
from prc2026_nmclock.features import route_mask
from prc2026_timeline.inference import node_prediction


def airport_prediction(run, nodes, features, threads):
    if not route_mask(features).all():
        raise ValueError("Airport components require the complete clock-present route.")
    values = np.empty(len(features), dtype=float)
    covered = np.zeros(len(features), dtype=bool)
    for airport, node in nodes.items():
        mask = features.airport.eq(airport).to_numpy(dtype=bool)
        if mask.any():
            values[mask] = node_prediction(run, node, features.loc[mask].reset_index(drop=True), threads)
            covered[mask] = True
    if not covered.all() or not np.isfinite(values).all():
        raise ValueError("Every clock-present row needs exactly one finite airport prediction.")
    return values
