"""Compare adsb.lol surface pushback times with the airport block time on labeled training days.

For each departure of the given days, finds the aircraft's surface track at the
airport (callsign match, takeoff within the track's ground-to-air transition),
locates the last stand stop before takeoff and the first movement after it,
and reports per-airport offsets between that movement time and BLOCK_TIME.
December is refused. Read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME

SURFACE = ROOT / "data/external/adsblol/surface"
COLUMNS = [ID, PHASE, TIME, "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ADEP_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "AOBT_3_flt", "LOBT_flt", "STAND_mvt", "RUNWAY_mvt", TARGET]
EARTH = 6371000.0


def metres(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    return EARTH * np.sqrt(dlat ** 2 + dlon ** 2)


def pushback(track: pd.DataFrame, takeoff: float, gap_seconds: float = 600.0, stop_seconds: float = 60.0, radius_m: float = 25.0, leave_m: float = 30.0) -> dict | None:
    """Departure surface segment (after the last coverage gap before takeoff); stand stop and first movement if the gate was covered.

    regime 'gate': the segment starts with a stationary run at the stand -> motion_start is the first point away from it.
    regime 'dark': the segment starts already moving (transponder or coverage absent at the gate) -> first_point is an upper bound.
    """
    window = track[(track["time"] <= takeoff + 30) & (track["time"] >= takeoff - 4 * 3600)].sort_values("time")
    airborne_before = window[(~window["ground"]) & (window["time"] < takeoff - 60)]
    if len(airborne_before):
        window = window[window["time"] > airborne_before["time"].max()]
    g = window[window["ground"]]
    if len(g) < 3:
        return None
    t = g["time"].to_numpy(); lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy(); gs = g["gs"].fillna(0).to_numpy()
    gaps = np.flatnonzero(np.diff(t) > gap_seconds)
    seg = int(gaps[-1] + 1) if len(gaps) else 0
    t, lat, lon, gs = t[seg:], lat[seg:], lon[seg:], gs[seg:]
    if len(t) < 3:
        return None
    # Stationary run at the segment start: points within radius_m of the first point.
    b = 0
    while b + 1 < len(t) and metres(lat[0], lon[0], lat[b + 1], lon[b + 1]) <= radius_m:
        b += 1
    stationary = t[b] - t[0] >= stop_seconds and gs[0] <= 2.0
    anchor_lat, anchor_lon = float(np.mean(lat[:b + 1])), float(np.mean(lon[:b + 1]))
    away = np.flatnonzero(metres(anchor_lat, anchor_lon, lat, lon) > leave_m)
    if stationary:
        if len(away) == 0:
            return None
        move = int(away[0]); last = move - 1
        motion_start, regime = float(t[move]), "gate"
    else:
        move, last = 0, 0
        motion_start, regime = float(t[0]), "dark"
    return {"stand_lat": anchor_lat, "stand_lon": anchor_lon, "regime": regime, "segment_start": float(t[0]), "stop_end": float(t[last]),
            "motion_start": motion_start, "motion_gap_s": float(t[move] - t[last]), "first_ground": float(g["time"].iloc[0]), "last_ground": float(t[-1]),
            "ground_points": int(len(t)), "stop_seconds": float(t[b] - t[0]), "first_gs": float(gs[0]), "arrival_seen": bool(len(airborne_before)),
            "gap_before_segment_s": float(t[0] - g["time"].to_numpy()[seg - 1]) if seg > 0 else np.nan,
            "taxi_path_m": float(np.sum(metres(lat[move:-1], lon[move:-1], lat[move + 1:], lon[move + 1:]))) if len(t) - move > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", required=True, help="YYYYMMDD of extracted surface files with labels available")
    parser.add_argument("--report", type=Path, default=Path("docs/opensky/adsb_pushback_audit.json"))
    args = parser.parse_args()
    if any(d[4:6] == "12" and d[:4] == "2025" for d in args.days):
        raise SystemExit("December remains reserved.")
    results, matched_rows = {}, []
    for day in args.days:
        y, m = int(day[:4]), int(day[4:6])
        raw = pd.read_parquet(ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet", columns=COLUMNS)
        dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y%m%d").eq(day)].reset_index(drop=True)
        surface = pd.read_parquet(SURFACE / f"surface_{day}.parquet")
        surface["callsign"] = surface["callsign"].fillna("").str.strip().str.upper()
        epoch = pd.Timestamp(0, tz="UTC")
        dep["takeoff"] = (dep[TIME] - epoch).dt.total_seconds()
        dep["block"] = (dep["BLOCK_TIME_UTC_mvt"] - epoch).dt.total_seconds()
        dep["callsign"] = dep["CALLSIGN_flt"].fillna(dep["FLIGHT_mvt"]).fillna("").str.strip().str.upper()
        for airport, group in dep.groupby("ADEP_mvt"):
            s = surface[surface["airport"].eq(airport)]
            if s.empty:
                results.setdefault(airport, []).append({"day": day, "departures": int(len(group)), "surface_rows": 0, "matched": 0})
                continue
            by_callsign = {c: frame for c, frame in s[s["callsign"].ne("")].groupby("callsign")}
            by_icao = {i: frame for i, frame in s.groupby("icao")}
            found = 0
            for row in group.itertuples(index=False):
                cand = by_callsign.get(row.callsign)
                if cand is None:
                    continue
                # aircraft whose track has ground points before takeoff and airborne points after it
                icaos = cand["icao"].unique()
                best = None
                for icao in icaos:
                    track = by_icao[icao]
                    before = track[(track["time"] <= row.takeoff + 30) & (track["time"] >= row.takeoff - 4 * 3600) & track["ground"]]
                    after = track[(track["time"] >= row.takeoff - 30) & (track["time"] <= row.takeoff + 900) & ~track["ground"]]
                    if len(before) >= 3 and len(after) >= 1:
                        info = pushback(track, row.takeoff)
                        if info is not None and (best is None or abs(info["last_ground"] - row.takeoff) < abs(best["last_ground"] - row.takeoff)):
                            best = {**info, "icao": icao}
                if best is None:
                    continue
                found += 1
                matched_rows.append({ID: row.MVT_ID_mvt, "day": day, "airport": airport, "callsign": row.callsign, "stand": row.STAND_mvt, "runway": row.RUNWAY_mvt,
                                     "takeoff": row.takeoff, "block": row.block, "label": float(getattr(row, TARGET)),
                                     "aobt": (row.AOBT_3_flt - epoch).total_seconds() if pd.notna(row.AOBT_3_flt) else np.nan, **best})
            results.setdefault(airport, []).append({"day": day, "departures": int(len(group)), "surface_rows": int(len(s)), "matched": found, "match_rate": found / len(group)})
    matched = pd.DataFrame(matched_rows)
    report = {"created_at_utc": utc_now(), "days": args.days, "coverage": results, "airports": {}}
    if len(matched):
        matched["offset_motion_s"] = matched["motion_start"] - matched["block"]      # positive: motion seen after the block time
        matched["offset_stop_end_s"] = matched["stop_end"] - matched["block"]
        matched["offset_first_ground_s"] = matched["first_ground"] - matched["block"]
        matched["taxi_adsb_s"] = matched["takeoff"] - matched["motion_start"]
        matched["takeoff_minus_last_ground_s"] = matched["takeoff"] - matched["last_ground"]
        matched["proxy_error_s"] = (matched["takeoff"] - matched["aobt"]) - matched["label"]
        for airport, frame in matched.groupby("airport"):
            o = frame["offset_motion_s"].to_numpy(); e = frame["taxi_adsb_s"].to_numpy() - frame["label"].to_numpy()
            p = frame["proxy_error_s"].dropna().to_numpy()
            regimes = {}
            for regime, part in frame.groupby("regime"):
                oo = part["offset_motion_s"].to_numpy(); ee = part["taxi_adsb_s"].to_numpy() - part["label"].to_numpy()
                regimes[regime] = {"rows": int(len(part)), "offset_quantiles_s": {str(q): float(np.quantile(oo, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
                                   "offset_within_60s": float(np.mean(np.abs(oo) <= 60)), "offset_std_s": float(np.std(oo)),
                                   "offset_iqr_s": float(np.quantile(oo, 0.75) - np.quantile(oo, 0.25)),
                                   "rmse_label_vs_adsb_taxi_centered_s": float(np.sqrt(np.mean((ee - np.median(ee)) ** 2)))}
            report["airports"][airport] = {
                "matched": int(len(frame)), "regimes": regimes,
                "motion_offset_quantiles_s": {str(q): float(np.quantile(o, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
                "motion_offset_within_60s": float(np.mean(np.abs(o) <= 60)), "motion_offset_within_180s": float(np.mean(np.abs(o) <= 180)),
                "stop_end_offset_median_s": float(frame["offset_stop_end_s"].median()), "first_ground_offset_median_s": float(frame["offset_first_ground_s"].median()),
                "rmse_adsb_taxi_vs_label_s": float(np.sqrt(np.mean(e ** 2))), "rmse_adsb_taxi_vs_label_centered_s": float(np.sqrt(np.mean((e - np.median(e)) ** 2))),
                "mae_adsb_taxi_vs_label_s": float(np.mean(np.abs(e))), "rmse_nm_proxy_vs_label_centered_s": float(np.sqrt(np.mean((p - np.median(p)) ** 2))) if len(p) else None,
                "takeoff_minus_last_ground_quantiles_s": {str(q): float(np.quantile(frame["takeoff_minus_last_ground_s"], q)) for q in [0.05, 0.5, 0.95]},
                "motion_gap_median_s": float(frame["motion_gap_s"].median())}
            print(airport, "matched", len(frame), "nm proxy centered rmse", round(report["airports"][airport]["rmse_nm_proxy_vs_label_centered_s"] or float("nan"), 1))
            for regime, r in regimes.items():
                print(f"   {regime:5s} rows {r['rows']:4d} offset q {({k: round(v) for k, v in r['offset_quantiles_s'].items()})} within60 {r['offset_within_60s']:.3f} iqr {r['offset_iqr_s']:.0f} std {r['offset_std_s']:.0f} | label vs adsb taxi centered rmse {r['rmse_label_vs_adsb_taxi_centered_s']:.1f}")
        matched.to_parquet(ROOT / "docs/opensky" / "adsb_pushback_matches.parquet", index=False)
    print("coverage", json.dumps({a: [{k: r[k] for k in r if k in ["day", "departures", "matched"]} for r in rows] for a, rows in results.items()}))
    write_json(ROOT / args.report, report)


if __name__ == "__main__":
    main()
