"""v19 screen features: segment timing, stand-name imputation and surface congestion.

Every quantity derives from adsb.lol surface positions (the departing aircraft's own
trace or other aircraft's traces), the supplied stand and runway names and the supplied
take-off time. No departure label or block time is read anywhere in this module.
Declared in docs/adsb_v19_screen/README.md.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from prc2026.features import ID
from prc2026_adsb.features import metres

TIMING_NAMES = ["adsb_appear_s", "adsb_last_stand_s", "adsb_pre_motion_gap_s", "adsb_first_motion_s"]
STAND_NAMES = ["adsb_first_dist_stand_imp_m", "adsb_stand_imputed"]
CONGESTION_NAMES = ["sfc_ground_10m", "sfc_moving_10m", "sfc_moving_20m", "sfc_rwy_queue_5m", "sfc_rwy_queue_10m"]
FEATURE_NAMES_V19 = TIMING_NAMES + STAND_NAMES + CONGESTION_NAMES
STAND_CODE = re.compile(r"^([A-Za-z]*)(\d+)([A-Za-z]*)$")


def segment_timing(track: pd.DataFrame, match: pd.Series, silence_seconds: float = 120.0, leave_m: float = 40.0) -> dict:
    """Timing of the matched segment relative to the motion start (see the declaration, item A)."""
    g = track[track["ground"] & (track["time"] >= match["segment_start"]) & (track["time"] <= match["takeoff"] + 30)].sort_values("time")
    t = g["time"].to_numpy()
    if len(t) < 2:
        return {}
    move = int(np.searchsorted(t, match["motion_start"]))
    move = min(move, len(t) - 1)
    # walk back from the motion start while consecutive gaps stay below the silence threshold
    start = move
    while start > 0 and t[start] - t[start - 1] < silence_seconds:
        start -= 1
    out = {"adsb_appear_s": float(match["takeoff"] - t[start]),
           "adsb_pre_motion_gap_s": float(t[move] - t[move - 1]) if move > 0 else np.nan,
           "adsb_last_stand_s": float(match["takeoff"] - t[move - 1]) if move > 0 and bool(match["gate"]) else np.nan}
    dist = metres(match["anchor_lat"], match["anchor_lon"], g["lat"].to_numpy(), g["lon"].to_numpy())
    away = np.flatnonzero(dist > leave_m)
    out["adsb_first_motion_s"] = float(match["takeoff"] - t[away[0]]) if len(away) else np.nan
    return out


def timing_features(matches: pd.DataFrame, surface: pd.DataFrame) -> pd.DataFrame:
    """One row per matched departure with the timing features; NaN where the trace is missing."""
    rows = []
    by_icao = {i: frame for i, frame in surface.groupby("icao")} if len(surface) else {}
    for _, match in matches.iterrows():
        track = by_icao.get(match["icao"])
        values = segment_timing(track, match) if track is not None else {}
        rows.append({ID: match[ID], **{name: values.get(name, np.nan) for name in TIMING_NAMES}})
    return pd.DataFrame(rows, columns=[ID, *TIMING_NAMES])


def impute_stands(stands: pd.DataFrame, departures: pd.DataFrame, max_number_gap: int = 3) -> pd.DataFrame:
    """Position for every (airport, stand) of `departures`: the learned one, else the nearest code by name, else the prefix median."""
    learned = {(r.airport, r.stand): (r.lat, r.lon) for r in stands.itertuples(index=False)}
    parsed = {}
    for r in stands.itertuples(index=False):
        m = STAND_CODE.match(str(r.stand))
        if m:
            parsed.setdefault((r.airport, m.group(1).upper()), []).append((int(m.group(2)), r.lat, r.lon))
    prefix_median = {key: (float(np.median([v[1] for v in vals])), float(np.median([v[2] for v in vals]))) for key, vals in parsed.items()}
    rows = []
    for airport, stand in departures[["ADEP_mvt", "STAND_mvt"]].drop_duplicates().itertuples(index=False):
        if not isinstance(stand, str):
            continue
        if (airport, stand) in learned:
            lat, lon = learned[(airport, stand)]
            rows.append((airport, stand, lat, lon, 0))
            continue
        m = STAND_CODE.match(stand)
        if not m:
            continue
        key, number = (airport, m.group(1).upper()), int(m.group(2))
        candidates = [(abs(n - number), lat, lon) for n, lat, lon in parsed.get(key, []) if abs(n - number) <= max_number_gap]
        if candidates:
            _, lat, lon = min(candidates)
            rows.append((airport, stand, lat, lon, 1))
        elif key in prefix_median:
            lat, lon = prefix_median[key]
            rows.append((airport, stand, lat, lon, 1))
    return pd.DataFrame(rows, columns=["airport", "stand", "lat", "lon", "imputed"])


def stand_features(matches: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Distance of the first surface point from the (learned or imputed) stand; NaN where no position exists."""
    out = matches[[ID, "airport", "stand", "first_lat", "first_lon"]].merge(positions, on=["airport", "stand"], how="left")
    out["adsb_first_dist_stand_imp_m"] = metres(out["lat"].to_numpy(), out["lon"].to_numpy(), out["first_lat"].to_numpy(), out["first_lon"].to_numpy())
    out["adsb_stand_imputed"] = out["imputed"].astype(float)
    return out[[ID, *STAND_NAMES]]


def surface_congestion(departures: pd.DataFrame, surface: pd.DataFrame, liftoff: pd.DataFrame, own_icao: pd.Series | None = None,
                       queue_radius_m: float = 1200.0, moving_kt: float = 5.0) -> pd.DataFrame:
    """Distinct other aircraft on the ground before each take-off: airport-wide and near the departure runway's roll-start point."""
    out = departures[[ID]].copy()
    for name in CONGESTION_NAMES:
        out[name] = np.nan
    if surface.empty:
        return out
    ground = surface[surface["ground"]]
    lift = {(r.airport, r.runway): (r.lat, r.lon) for r in liftoff.itertuples(index=False)}
    own = own_icao.reindex(departures.index) if own_icao is not None else pd.Series(index=departures.index, dtype=object)
    for airport, rows in departures.groupby("ADEP_mvt"):
        g = ground[ground["airport"].eq(airport)].sort_values("time")
        if g.empty:
            continue
        t = g["time"].to_numpy(); icao = g["icao"].to_numpy(); moving = (g["gs"].fillna(0).to_numpy() >= moving_kt)
        lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy()
        near_cache = {}
        for idx, r in rows.iterrows():
            key = (airport, r["RUNWAY_mvt"])
            if key in lift and key not in near_cache:
                near_cache[key] = metres(lift[key][0], lift[key][1], lat, lon) <= queue_radius_m
            near = near_cache.get(key)
            take = r["takeoff"]; mine = own.get(idx)
            lo10, lo20, lo5, hi = np.searchsorted(t, take - 600), np.searchsorted(t, take - 1200), np.searchsorted(t, take - 300), np.searchsorted(t, take, side="right")
            def distinct(a, b, mask=None):
                sel = icao[a:b] if mask is None else icao[a:b][mask[a:b]]
                if isinstance(mine, str):
                    sel = sel[sel != mine]
                return float(len(np.unique(sel)))
            out.at[idx, "sfc_ground_10m"] = distinct(lo10, hi)
            out.at[idx, "sfc_moving_10m"] = distinct(lo10, hi, moving)
            out.at[idx, "sfc_moving_20m"] = distinct(lo20, hi, moving)
            if near is not None:
                out.at[idx, "sfc_rwy_queue_5m"] = distinct(lo5, hi, near)
                out.at[idx, "sfc_rwy_queue_10m"] = distinct(lo10, hi, near)
    return out
