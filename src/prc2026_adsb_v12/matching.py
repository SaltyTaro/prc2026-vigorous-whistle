"""Matcher variants for the adsb.lol surface traces (ODbL), measured against the frozen prc2026_adsb matcher.

The frozen module (src/prc2026_adsb/features.py) is a hash-bound parent of every shipped declaration and is
NOT edited. This module re-implements `segment` and `match_departures` with the census's four failure points
exposed as options, so each can be turned on and measured separately:

  min_ground_points   the frozen matcher needs three ground points before takeoff and three inside the
                      segment; the census finds 3.1-3.5% of departures held out with one or two (mostly LFPG).
  callsign_variant    12.9% (Jan) / 19.4% (Jul) of departures never find their callsign at the airport, and
                      more than half of those have a callsign with the SAME flight number and a different
                      prefix in the same day's points (the IATA/ICAO spelling difference). A variant is
                      accepted only when it is unambiguous: exactly one aircraft passes the same ground and
                      airborne gates, and, when both are known, its type equals the movement's aircraft type.
  stand_radius_m      the frozen matcher treats a first point as "at the known stand" within 40 m; 15-21% of
                      dark rows have their first point within 100 m of a stand we do know.
  extra_stands        stand positions from a source other than gate-covered segments. `learn_stands` learns
                      only from rows already classified gate, so an airport rarely caught at the gate never
                      learns its stands (LFPG has 3, LIRF 43, EHAM 187).

Every option defaults to the frozen behaviour, so `match_departures(...)` with no options reproduces it.
No label, block time or taxi time is read here.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from prc2026.features import ID, TIME
from prc2026_adsb.features import EPOCH, metres

DEPARTURE_COLUMNS = [ID, "PHASE_mvt", TIME, "ADEP_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
DIGITS = re.compile(r"(\d+)")


def flight_number(callsign: str) -> str:
    """The trailing number of a callsign: AFR1234 -> 1234, BA0871 -> 871. Empty when there is none."""
    found = DIGITS.findall(callsign)
    return found[-1].lstrip("0") if found else ""


def _sustained_departure(lat, lon, anchor_lat, anchor_lon, start: int, radius_m: float, leave_m: float) -> int | None:
    distance = metres(anchor_lat, anchor_lon, lat, lon)
    inside = np.flatnonzero(distance[start:] <= radius_m)
    last_inside = start + int(inside[-1]) if len(inside) else start - 1
    away = np.flatnonzero((np.arange(len(lat)) > last_inside) & (distance > leave_m))
    return int(away[0]) if len(away) else None


def segment(track: pd.DataFrame, takeoff: float, gap_seconds: float = 600.0, stop_seconds: float = 60.0, radius_m: float = 25.0,
            leave_m: float = 40.0, stand: tuple[float, float] | None = None, taxi_speed_kt: float = 8.0,
            min_ground_points: int = 3, stand_radius_m: float | None = None) -> dict | None:
    """As prc2026_adsb.features.segment, with the minimum point count and the known-stand radius exposed.

    `min_ground_points` = 3 and `stand_radius_m` = None (meaning leave_m) reproduce the frozen function exactly.
    With fewer points the path, stop and slow-start quantities degrade to what the points support; `points`
    records how many there were, so a model can discount a thin segment.
    """
    at_stand_radius = leave_m if stand_radius_m is None else float(stand_radius_m)
    window = track[(track["time"] <= takeoff + 30) & (track["time"] >= takeoff - 4 * 3600)].sort_values("time")
    airborne_before = window[(~window["ground"]) & (window["time"] < takeoff - 60)]
    arrival_last_ground = np.nan
    if len(airborne_before):
        cut = airborne_before["time"].max()
        earlier_ground = window[window["ground"] & (window["time"] < cut)]
        arrival_last_ground = float(earlier_ground["time"].max()) if len(earlier_ground) else np.nan
        window = window[window["time"] > cut]
    g = window[window["ground"]]
    if len(g) < min_ground_points:
        return None
    t_all = g["time"].to_numpy()
    gaps = np.flatnonzero(np.diff(t_all) > gap_seconds)
    seg = int(gaps[-1] + 1) if len(gaps) else 0
    if seg > 0 and np.isnan(arrival_last_ground):
        arrival_last_ground = float(t_all[seg - 1])
    t = t_all[seg:]; lat = g["lat"].to_numpy()[seg:]; lon = g["lon"].to_numpy()[seg:]; gs = g["gs"].fillna(0).to_numpy()[seg:]
    if len(t) < min_ground_points:
        return None
    b = 0
    while b + 1 < len(t) and metres(lat[0], lon[0], lat[b + 1], lon[b + 1]) <= radius_m:
        b += 1
    stationary = t[b] - t[0] >= stop_seconds and gs[0] <= 2.0
    anchor_lat, anchor_lon = float(np.mean(lat[:b + 1])), float(np.mean(lon[:b + 1]))
    at_known_stand = stand is not None and metres(stand[0], stand[1], lat[0], lon[0]) <= at_stand_radius
    if at_known_stand and not stationary:
        anchor_lat, anchor_lon = float(stand[0]), float(stand[1])
    move = None
    if stationary or at_known_stand:
        move = _sustained_departure(lat, lon, anchor_lat, anchor_lon, 0, radius_m, leave_m)
    gate = move is not None
    if move is None:
        move = 0
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


def match_departures(departures: pd.DataFrame, surface: pd.DataFrame, stands: pd.DataFrame | None = None,
                     min_ground_points: int = 3, callsign_variant: bool = False, stand_radius_m: float | None = None,
                     extra_stands: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per matched departure, with `match_kind` recording how the aircraft was identified.

    Defaults reproduce the frozen matcher. `extra_stands` is appended to `stands` for names the learned table
    does not carry (it never overrides a learned position).
    """
    columns = [c for c in DEPARTURE_COLUMNS if c in departures.columns]
    dep = departures[columns].copy()
    dep["takeoff"] = (dep[TIME] - EPOCH).dt.total_seconds()
    dep["callsign"] = dep["CALLSIGN_flt"].fillna(dep["FLIGHT_mvt"]).fillna("").str.strip().str.upper()
    types = dep["AIRCRAFT_TYPE_mvt"].fillna("").str.strip().str.upper() if "AIRCRAFT_TYPE_mvt" in dep else pd.Series([""] * len(dep), index=dep.index)
    dep["aircraft_type"] = types
    stand_lookup = {}
    for table in [extra_stands, stands]:  # learned positions take precedence
        if table is not None and len(table):
            stand_lookup.update({(r.airport, r.stand): (r.lat, r.lon) for r in table.itertuples(index=False)})
    rows = []
    for airport, group in dep.groupby("ADEP_mvt"):
        s = surface[surface["airport"].eq(airport)] if len(surface) else surface
        if s.empty:
            continue
        by_callsign = {c: frame["icao"].unique() for c, frame in s[s["callsign"].ne("")].groupby("callsign")}
        by_icao = {i: frame for i, frame in s.groupby("icao")}
        by_number = {}
        if callsign_variant:
            for c in by_callsign:
                number = flight_number(c)
                if number:
                    by_number.setdefault(number, []).append(c)
        icao_type = {i: (frame["type"].dropna().iloc[0].strip().upper() if frame["type"].notna().any() else "") for i, frame in by_icao.items()}
        for row in group.itertuples(index=False):
            stand = stand_lookup.get((airport, row.STAND_mvt)) if isinstance(row.STAND_mvt, str) else None
            icaos, kind = by_callsign.get(row.callsign), "callsign"
            if icaos is None:
                if not callsign_variant:
                    continue
                number = flight_number(row.callsign)
                candidates = [c for c in by_number.get(number, []) if c != row.callsign] if number else []
                if not candidates:
                    continue
                icaos, kind = np.unique(np.concatenate([by_callsign[c] for c in candidates])), "variant"
            best, accepted = None, 0
            for icao in icaos:
                track = by_icao[icao]
                before = track[(track["time"] <= row.takeoff + 30) & (track["time"] >= row.takeoff - 4 * 3600) & track["ground"]]
                after = track[(track["time"] >= row.takeoff - 30) & (track["time"] <= row.takeoff + 900) & ~track["ground"]]
                if len(before) < min_ground_points or len(after) < 1:
                    continue
                if kind == "variant":
                    # A variant is accepted only when the aircraft type agrees (when both are known).
                    observed = icao_type.get(icao, "")
                    if observed and row.aircraft_type and observed != row.aircraft_type:
                        continue
                info = segment(track, row.takeoff, stand=stand, min_ground_points=min_ground_points, stand_radius_m=stand_radius_m)
                if info is None:
                    continue
                accepted += 1
                if best is None or abs(info["last_ground"] - row.takeoff) < abs(best["last_ground"] - row.takeoff):
                    best = {**info, "icao": icao}
            if kind == "variant" and accepted != 1:
                continue  # ambiguous or unsupported: leave the row unmatched rather than guess
            if best is not None:
                rows.append({ID: getattr(row, ID), "airport": airport, "stand": row.STAND_mvt, "takeoff": row.takeoff,
                             "match_kind": kind, **best})
    return pd.DataFrame(rows)
