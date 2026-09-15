"""Departure surface segments and label-free features from adsb.lol traces.

Every quantity derives from surface positions of the departing aircraft (matched
by ICAO callsign and takeoff time) and from the supplied stand name. No
departure label or airport block time is read anywhere in this module.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME

EARTH = 6371000.0
EPOCH = pd.Timestamp(0, tz="UTC")
DEPARTURE_COLUMNS = [ID, PHASE, TIME, "ADEP_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt"]
FEATURE_NAMES = ["adsb_matched", "adsb_gate", "adsb_first_point_s", "adsb_motion_s", "adsb_first_gs_kt", "adsb_first_dist_stand_m",
                 "adsb_stop_seconds", "adsb_taxi_path_m", "adsb_taxi_stops", "adsb_taxi_stopped_s", "adsb_slow_start_s", "adsb_gap_before_s",
                 "adsb_turnaround_s", "adsb_anchor_dist_stand_m", "adsb_last_ground_s", "adsb_points"]
NEIGHBOUR_NAMES = ["nb_runway_10m_count", "nb_runway_10m_median_taxi_s", "nb_runway_10m_max_taxi_s", "nb_runway_10m_median_stopped_s",
                   "nb_airport_10m_count", "nb_airport_10m_median_taxi_s", "nb_runway_30m_count", "nb_runway_30m_median_taxi_s",
                   "nb_runway_30m_median_first_point_s", "nb_runway_prev_taxi_s", "nb_runway_next_taxi_s"]


def metres(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    return EARTH * np.sqrt(dlat ** 2 + dlon ** 2)


def load_surface(directory: Path, days: list[str]) -> pd.DataFrame:
    frames = []
    for day in days:
        path = directory / f"surface_{day}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    surface = pd.concat(frames, ignore_index=True)
    surface["callsign"] = surface["callsign"].fillna("").str.strip().str.upper()
    return surface


def _sustained_departure(lat, lon, anchor_lat, anchor_lon, start: int, radius_m: float, leave_m: float) -> int | None:
    """Index of the first point after which the aircraft never returns within radius_m of the anchor (and is beyond leave_m)."""
    distance = metres(anchor_lat, anchor_lon, lat, lon)
    inside = np.flatnonzero(distance[start:] <= radius_m)
    last_inside = start + int(inside[-1]) if len(inside) else start - 1
    away = np.flatnonzero((np.arange(len(lat)) > last_inside) & (distance > leave_m))
    return int(away[0]) if len(away) else None


def segment(track: pd.DataFrame, takeoff: float, gap_seconds: float = 600.0, stop_seconds: float = 60.0, radius_m: float = 25.0,
            leave_m: float = 40.0, stand: tuple[float, float] | None = None, taxi_speed_kt: float = 8.0) -> dict | None:
    """Surface points of the departure (after the last arrival and the last coverage gap), with the stand stop if covered.

    'gate' segments start with a stationary run (or within radius of the known stand); their motion start is the first point
    after the aircraft last left the stand vicinity for good. 'dark' segments start already moving.
    """
    window = track[(track["time"] <= takeoff + 30) & (track["time"] >= takeoff - 4 * 3600)].sort_values("time")
    airborne_before = window[(~window["ground"]) & (window["time"] < takeoff - 60)]
    arrival_last_ground = np.nan
    if len(airborne_before):
        cut = airborne_before["time"].max()
        earlier_ground = window[window["ground"] & (window["time"] < cut)]
        arrival_last_ground = float(earlier_ground["time"].max()) if len(earlier_ground) else np.nan
        window = window[window["time"] > cut]
    g = window[window["ground"]]
    if len(g) < 3:
        return None
    t_all = g["time"].to_numpy()
    gaps = np.flatnonzero(np.diff(t_all) > gap_seconds)
    seg = int(gaps[-1] + 1) if len(gaps) else 0
    if seg > 0 and np.isnan(arrival_last_ground):
        arrival_last_ground = float(t_all[seg - 1])
    t = t_all[seg:]; lat = g["lat"].to_numpy()[seg:]; lon = g["lon"].to_numpy()[seg:]; gs = g["gs"].fillna(0).to_numpy()[seg:]
    if len(t) < 3:
        return None
    b = 0
    while b + 1 < len(t) and metres(lat[0], lon[0], lat[b + 1], lon[b + 1]) <= radius_m:
        b += 1
    stationary = t[b] - t[0] >= stop_seconds and gs[0] <= 2.0
    anchor_lat, anchor_lon = float(np.mean(lat[:b + 1])), float(np.mean(lon[:b + 1]))
    at_known_stand = stand is not None and metres(stand[0], stand[1], lat[0], lon[0]) <= leave_m
    if at_known_stand and not stationary:
        anchor_lat, anchor_lon = float(stand[0]), float(stand[1])
    move = None
    if stationary or at_known_stand:
        move = _sustained_departure(lat, lon, anchor_lat, anchor_lon, 0, radius_m, leave_m)
    gate = move is not None
    if move is None:
        move = 0
    # Taxi stops after motion start: runs of consecutive points within radius_m lasting >= 30 s.
    stops, stopped, start = 0, 0.0, move
    for i in range(move + 1, len(t) + 1):
        if i == len(t) or metres(lat[start], lon[start], lat[i], lon[i]) > radius_m:
            if t[i - 1] - t[start] >= 30:
                stops += 1
                stopped += float(t[i - 1] - t[start])
            start = i
    fast = np.flatnonzero(gs[move:] >= taxi_speed_kt)
    slow_start = float(t[move + int(fast[0])] - t[move]) if len(fast) else float(t[-1] - t[move])
    path = float(np.sum(metres(lat[move:-1], lon[move:-1], lat[move + 1:], lon[move + 1:]))) if len(t) - move > 1 else 0.0
    return {"gate": gate, "segment_start": float(t[0]), "motion_start": float(t[move]), "first_gs": float(gs[0]),
            "stop_seconds": float(t[b] - t[0]) if stationary else 0.0, "anchor_lat": anchor_lat, "anchor_lon": anchor_lon,
            "first_lat": float(lat[0]), "first_lon": float(lon[0]), "taxi_path_m": path, "taxi_stops": stops, "taxi_stopped_s": stopped,
            "slow_start_s": slow_start, "gap_before_s": float(t[0] - t_all[seg - 1]) if seg > 0 else np.nan, "arrival_last_ground": arrival_last_ground,
            "last_ground": float(t[-1]), "points": int(len(t)), "at_known_stand": bool(at_known_stand),
            "first_dist_stand_m": float(metres(stand[0], stand[1], lat[0], lon[0])) if stand else np.nan,
            "anchor_dist_stand_m": float(metres(stand[0], stand[1], anchor_lat, anchor_lon)) if stand else np.nan}


def match_departures(departures: pd.DataFrame, surface: pd.DataFrame, stands: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per departure with its surface segment (or NaNs), by callsign and takeoff within the ground-to-air transition."""
    dep = departures[DEPARTURE_COLUMNS].copy()
    dep["takeoff"] = (dep[TIME] - EPOCH).dt.total_seconds()
    dep["callsign"] = dep["CALLSIGN_flt"].fillna(dep["FLIGHT_mvt"]).fillna("").str.strip().str.upper()
    stand_lookup = {}
    if stands is not None and len(stands):
        stand_lookup = {(r.airport, r.stand): (r.lat, r.lon) for r in stands.itertuples(index=False)}
    rows = []
    for airport, group in dep.groupby("ADEP_mvt"):
        s = surface[surface["airport"].eq(airport)] if len(surface) else surface
        if s.empty:
            continue
        by_callsign = {c: frame["icao"].unique() for c, frame in s[s["callsign"].ne("")].groupby("callsign")}
        by_icao = {i: frame for i, frame in s.groupby("icao")}
        for row in group.itertuples(index=False):
            icaos = by_callsign.get(row.callsign)
            if icaos is None:
                continue
            best = None
            stand = stand_lookup.get((airport, row.STAND_mvt)) if isinstance(row.STAND_mvt, str) else None
            for icao in icaos:
                track = by_icao[icao]
                before = track[(track["time"] <= row.takeoff + 30) & (track["time"] >= row.takeoff - 4 * 3600) & track["ground"]]
                after = track[(track["time"] >= row.takeoff - 30) & (track["time"] <= row.takeoff + 900) & ~track["ground"]]
                if len(before) >= 3 and len(after) >= 1:
                    info = segment(track, row.takeoff, stand=stand)
                    if info is not None and (best is None or abs(info["last_ground"] - row.takeoff) < abs(best["last_ground"] - row.takeoff)):
                        best = {**info, "icao": icao}
            if best is not None:
                rows.append({ID: getattr(row, ID), "airport": airport, "stand": row.STAND_mvt, "takeoff": row.takeoff, **best})
    return pd.DataFrame(rows)


def learn_stands(matches: pd.DataFrame, minimum: int = 3) -> pd.DataFrame:
    """Median anchor position of gate-covered segments per (airport, stand name); label-free."""
    gate = matches[matches["gate"] & matches["stand"].notna()]
    grouped = gate.groupby(["airport", "stand"]).agg(lat=("anchor_lat", "median"), lon=("anchor_lon", "median"), n=("anchor_lat", "size")).reset_index()
    return grouped[grouped["n"] >= minimum].reset_index(drop=True)


def departure_features(departures: pd.DataFrame, matches: pd.DataFrame, stands: pd.DataFrame | None = None) -> pd.DataFrame:
    """Feature table aligned to `departures` rows (NaN where no surface segment matched); stand distances from learned stand positions."""
    out = departures[[ID]].copy()
    takeoff = (departures[TIME] - EPOCH).dt.total_seconds().to_numpy()
    matches = matches.copy() if len(matches) else matches
    if len(matches) and stands is not None and len(stands):
        lookup = stands.set_index(["airport", "stand"])[["lat", "lon"]]
        keys = pd.MultiIndex.from_arrays([matches["airport"], matches["stand"].fillna("?")])
        known = lookup.reindex(keys)
        matches["first_dist_stand_m"] = metres(known["lat"].to_numpy(), known["lon"].to_numpy(), matches["first_lat"].to_numpy(), matches["first_lon"].to_numpy())
        matches["anchor_dist_stand_m"] = metres(known["lat"].to_numpy(), known["lon"].to_numpy(), matches["anchor_lat"].to_numpy(), matches["anchor_lon"].to_numpy())
    m = matches.set_index(ID).reindex(departures[ID].to_numpy()) if len(matches) else pd.DataFrame(index=departures[ID].to_numpy())
    matched = m["last_ground"].notna().to_numpy() if "last_ground" in m else np.zeros(len(out), dtype=bool)
    def col(name):
        return m[name].to_numpy(dtype=float) if name in m else np.full(len(out), np.nan)
    out["adsb_matched"] = matched.astype(float)
    out["adsb_gate"] = np.where(matched, m["gate"].fillna(False).to_numpy(dtype=float) if "gate" in m else 0.0, np.nan)
    out["adsb_first_point_s"] = takeoff - col("segment_start")
    out["adsb_motion_s"] = takeoff - col("motion_start")
    out["adsb_first_gs_kt"] = col("first_gs")
    out["adsb_first_dist_stand_m"] = col("first_dist_stand_m")
    out["adsb_stop_seconds"] = col("stop_seconds")
    out["adsb_taxi_path_m"] = col("taxi_path_m")
    out["adsb_taxi_stops"] = col("taxi_stops")
    out["adsb_taxi_stopped_s"] = col("taxi_stopped_s")
    out["adsb_slow_start_s"] = col("slow_start_s")
    out["adsb_gap_before_s"] = col("gap_before_s")
    out["adsb_turnaround_s"] = takeoff - col("arrival_last_ground")
    out["adsb_anchor_dist_stand_m"] = col("anchor_dist_stand_m")
    out["adsb_last_ground_s"] = takeoff - col("last_ground")
    out["adsb_points"] = col("points")
    return out[[ID, *FEATURE_NAMES]]


def neighbour_features(departures: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Taxi-out statistics of matched neighbours (same airport / runway, takeoff within a window), for every departure.

    Uses only other aircraft's surface segments and supplied takeoff times; the departure's own match (if any) is excluded.
    """
    out = departures[[ID]].copy()
    takeoff = (departures[TIME] - EPOCH).dt.total_seconds().to_numpy()
    airport = departures["ADEP_mvt"].to_numpy(); runway = departures["RUNWAY_mvt"].fillna("?").astype(str).to_numpy()
    values = {name: np.full(len(out), np.nan) for name in NEIGHBOUR_NAMES}
    if len(matches):
        m = matches.merge(departures[[ID, "RUNWAY_mvt"]], on=ID, how="inner")
        m["runway"] = m["RUNWAY_mvt"].fillna("?").astype(str)
        m["taxi"] = m["takeoff"] - m["motion_start"]; m["first_point"] = m["takeoff"] - m["segment_start"]
        m = m.sort_values("takeoff")
        for key_airport, group in m.groupby("airport"):
            rows_airport = np.flatnonzero(airport == key_airport)
            if len(rows_airport) == 0:
                continue
            t_all = group["takeoff"].to_numpy(); taxi_all = group["taxi"].to_numpy(); id_all = group[ID].to_numpy()
            for i in rows_airport:
                lo, hi = np.searchsorted(t_all, takeoff[i] - 600), np.searchsorted(t_all, takeoff[i] + 600)
                sel = [k for k in range(lo, hi) if id_all[k] != departures[ID].iat[i]]
                values["nb_airport_10m_count"][i] = len(sel)
                if sel:
                    values["nb_airport_10m_median_taxi_s"][i] = np.median(taxi_all[sel])
            for key_runway, sub in group.groupby("runway"):
                rows = rows_airport[runway[rows_airport] == key_runway]
                if len(rows) == 0:
                    continue
                t = sub["takeoff"].to_numpy(); taxi = sub["taxi"].to_numpy(); stopped = sub["taxi_stopped_s"].to_numpy()
                first = sub["first_point"].to_numpy(); ids = sub[ID].to_numpy()
                for i in rows:
                    own = departures[ID].iat[i]
                    for window, prefix in [(600, "nb_runway_10m"), (1800, "nb_runway_30m")]:
                        lo, hi = np.searchsorted(t, takeoff[i] - window), np.searchsorted(t, takeoff[i] + window)
                        sel = [k for k in range(lo, hi) if ids[k] != own]
                        values[prefix + "_count"][i] = len(sel)
                        if sel:
                            values[prefix + "_median_taxi_s"][i] = np.median(taxi[sel])
                            if window == 600:
                                values["nb_runway_10m_max_taxi_s"][i] = np.max(taxi[sel]); values["nb_runway_10m_median_stopped_s"][i] = np.median(stopped[sel])
                            else:
                                values["nb_runway_30m_median_first_point_s"][i] = np.median(first[sel])
                    before = [k for k in range(np.searchsorted(t, takeoff[i]) - 1, -1, -1) if ids[k] != own][:1]
                    after = [k for k in range(np.searchsorted(t, takeoff[i], side="right"), len(t)) if ids[k] != own][:1]
                    if before and takeoff[i] - t[before[0]] <= 1800:
                        values["nb_runway_prev_taxi_s"][i] = taxi[before[0]]
                    if after and t[after[0]] - takeoff[i] <= 1800:
                        values["nb_runway_next_taxi_s"][i] = taxi[after[0]]
    for name in NEIGHBOUR_NAMES:
        out[name] = values[name]
    return out
