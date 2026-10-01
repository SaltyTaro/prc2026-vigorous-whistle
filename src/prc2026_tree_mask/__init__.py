"""Static analogue, for the tree committee, of the context network's training-time masking (src/prc2026_ctx3/train.py).

The network draws each matched fitted row with probability p at every step and shows it as an unmatched row: the 21 focal-flight
ADS-B columns NaN (what every unmatched row carries) and the match flag 0, label unchanged. Its expected loss is
(1 - p) * loss(row as matched) + p * loss(row as unmatched). A tree learner fits that expectation exactly when every matched row
keeps weight 1 - p and gains a masked copy with weight p. Neighbour columns (nb_*, sfc_*) and every non-ADS-B column are untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MATCHED = "adsb_matched"
FOCAL_COLUMNS = [
    "adsb_gate", "adsb_first_point_s", "adsb_motion_s", "adsb_first_gs_kt", "adsb_first_dist_stand_m", "adsb_stop_seconds",
    "adsb_taxi_path_m", "adsb_taxi_stops", "adsb_taxi_stopped_s", "adsb_slow_start_s", "adsb_gap_before_s", "adsb_turnaround_s",
    "adsb_anchor_dist_stand_m", "adsb_last_ground_s", "adsb_points", "adsb_appear_s", "adsb_last_stand_s", "adsb_pre_motion_gap_s",
    "adsb_first_motion_s", "adsb_first_dist_stand_imp_m", "adsb_stand_imputed",
]


def augment(x: pd.DataFrame, y: np.ndarray, p: float, columns: list[str] = FOCAL_COLUMNS, matched: str = MATCHED):
    """Return (x, y, weight). With p == 0 the inputs come back unchanged and weight is None (the unweighted control path).

    Otherwise every row with matched == 1 keeps weight 1 - p and gains a copy (appended after the originals, in the same order)
    with weight p whose focal columns are NaN and whose match flag is 0; every other row has weight 1."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must lie in [0, 1).")
    missing = [c for c in [*columns, matched] if c not in x.columns]
    if missing:
        raise ValueError(f"Columns absent from the fitting frame: {missing}")
    y = np.asarray(y, dtype=float)
    if len(y) != len(x):
        raise ValueError("Label and frame lengths differ.")
    if p == 0.0:
        return x, y, None
    flag = x[matched].to_numpy(dtype=float)
    if not np.isin(flag[np.isfinite(flag)], [0.0, 1.0]).all() or not np.isfinite(flag).all():
        raise ValueError("The match flag must be 0 or 1 on every row.")
    rows = flag == 1.0
    copy = x.loc[rows].copy()
    for c in columns:
        copy[c] = np.nan
    copy[matched] = np.float32(0.0) if x[matched].dtype == np.float32 else 0.0
    copy = copy.astype(x.dtypes.to_dict())
    out = pd.concat([x, copy], ignore_index=True)
    weight = np.concatenate([np.where(rows, 1.0 - p, 1.0), np.full(int(rows.sum()), p)])
    return out, np.concatenate([y, y[rows]]), weight
