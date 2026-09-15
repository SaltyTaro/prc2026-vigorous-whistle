"""Measure whether OpenSky first-seen times carry off-block information beyond the NM clocks.

Joins downloaded OpenSky departure records (callsign + time window) to training
departures of the requested months, compares firstSeen with the airport block
time, and reports per-airport match rates, offset distributions and a simple
residual comparison against the NM proxy. December is refused. Read-only.
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

FLIGHTS = ROOT / "data/external/opensky/flights"
COLUMNS = [ID, PHASE, TIME, "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "ADEP_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "AOBT_3_flt", "LOBT_flt", "STAND_mvt", TARGET]


def load_flights(airport: str) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in (FLIGHTS / airport).glob("departure_*.json") if not p.name.endswith(".meta.json")):
        for record in json.loads(path.read_text(encoding="utf-8")):
            rows.append(record)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).drop_duplicates()
    frame["callsign"] = frame["callsign"].fillna("").str.strip().str.upper()
    return frame


def match(departures: pd.DataFrame, flights: pd.DataFrame, before_hours: float = 6.0, after_minutes: float = 5.0) -> pd.DataFrame:
    """For each departure, the OpenSky flight with the same callsign whose firstSeen is the latest one not after takeoff + 5 min."""
    dep = departures.copy()
    dep["callsign"] = dep["CALLSIGN_flt"].fillna(dep["FLIGHT_mvt"]).fillna("").str.strip().str.upper()
    dep["takeoff"] = (dep[TIME] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    fl = flights[["callsign", "icao24", "firstSeen", "lastSeen", "estDepartureAirport", "estArrivalAirport",
                  "estDepartureAirportHorizDistance", "estDepartureAirportVertDistance", "departureAirportCandidatesCount"]].copy()
    fl = fl.sort_values("firstSeen")
    merged = dep.merge(fl, on="callsign", how="inner")
    merged = merged[(merged["firstSeen"] >= merged["takeoff"] - before_hours * 3600) & (merged["firstSeen"] <= merged["takeoff"] + after_minutes * 60)]
    merged = merged.sort_values(["MVT_ID_mvt", "firstSeen"]).groupby(ID, as_index=False).tail(1)
    return dep.merge(merged.drop(columns=[c for c in dep.columns if c not in [ID]]), on=ID, how="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", required=True, help="Training months, e.g. 2025-03")
    parser.add_argument("--airports", nargs="+", default=None)
    parser.add_argument("--report", type=Path, default=Path("docs/opensky/flights_audit.json"))
    args = parser.parse_args()
    if any(m.endswith("-12") for m in args.months):
        raise SystemExit("December remains reserved.")
    frames = []
    for month in args.months:
        y, m = int(month[:4]), int(month[5:7])
        path = ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"
        frame = pd.read_parquet(path, columns=COLUMNS)
        frames.append(frame[frame[PHASE].eq("DEP")])
    departures = pd.concat(frames, ignore_index=True)
    airports = args.airports or sorted(p.name for p in FLIGHTS.iterdir() if p.is_dir())
    report = {"created_at_utc": utc_now(), "months": args.months, "airports": {}}
    for airport in airports:
        flights = load_flights(airport)
        dep = departures[departures["ADEP_mvt"].eq(airport)].reset_index(drop=True)
        if flights.empty or dep.empty:
            report["airports"][airport] = {"flight_records": int(len(flights)), "departures": int(len(dep)), "matched": 0}
            continue
        lo, hi = dep[TIME].min().timestamp() - 6 * 3600, dep[TIME].max().timestamp() + 600
        flights = flights[(flights["firstSeen"] >= lo) & (flights["firstSeen"] <= hi)]
        joined = match(dep, flights)
        ok = joined["firstSeen"].notna()
        label = joined[TARGET].to_numpy(dtype=float)
        block = (joined["BLOCK_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(dtype=float)
        first = joined["firstSeen"].to_numpy(dtype=float)
        takeoff = joined["takeoff"].to_numpy(dtype=float)
        aobt = (joined[TIME] - joined["AOBT_3_flt"]).dt.total_seconds().to_numpy(dtype=float)
        lobt = (joined[TIME] - joined["LOBT_flt"]).dt.total_seconds().to_numpy(dtype=float)
        proxy = np.where(np.isfinite(aobt), aobt, lobt)
        gap = takeoff - first                       # takeoff minus first seen
        offset = first - block                      # first seen minus airport block time (negative: seen before off-block)
        near = ok & (joined["estDepartureAirportHorizDistance"].fillna(1e9).to_numpy() < 6000)
        both = ok & np.isfinite(proxy)
        def stats(mask):
            if mask.sum() < 20:
                return None
            o = offset[mask]
            return {"rows": int(mask.sum()), "offset_quantiles_s": {str(q): float(np.quantile(o, q)) for q in [0.05, 0.25, 0.5, 0.75, 0.95]},
                    "offset_std_s": float(np.std(o)), "offset_within_120s_share": float(np.mean(np.abs(o) <= 120)),
                    "offset_within_300s_share": float(np.mean(np.abs(o) <= 300))}
        comparison = None
        if both.sum() >= 100:
            g, p, l = gap[both], proxy[both], label[both]
            g_c = g - np.median(g - l); p_c = p - np.median(p - l)
            # Best affine combination of the two clocks, in-sample (descriptive only)
            X = np.c_[np.ones(both.sum()), g, p]
            beta, *_ = np.linalg.lstsq(X, l, rcond=None)
            fitted = X @ beta
            comparison = {"rows": int(both.sum()),
                          "rmse_label_vs_firstseen_gap_centered": float(np.sqrt(np.mean((l - g_c) ** 2))),
                          "rmse_label_vs_nm_proxy_centered": float(np.sqrt(np.mean((l - p_c) ** 2))),
                          "rmse_affine_both_in_sample": float(np.sqrt(np.mean((l - fitted) ** 2))),
                          "mae_label_vs_firstseen_gap_centered": float(np.mean(np.abs(l - g_c))),
                          "mae_label_vs_nm_proxy_centered": float(np.mean(np.abs(l - p_c))),
                          "corr_firstseen_gap_minus_proxy_with_label_minus_proxy": float(np.corrcoef(g - p, l - p)[0, 1])}
        report["airports"][airport] = {"flight_records": int(len(flights)), "departures": int(len(dep)), "matched": int(ok.sum()),
                                       "match_rate": float(ok.mean()), "matched_first_seen_near_airport": int(near.sum()),
                                       "all_matched": stats(ok), "near_airport": stats(near), "comparison_with_nm_proxy": comparison}
        print(airport, json.dumps({k: report["airports"][airport][k] for k in ["departures", "matched", "match_rate", "matched_first_seen_near_airport"]}),
              "near-airport offset quantiles", None if not stats(near) else {k: round(v) for k, v in stats(near)["offset_quantiles_s"].items()},
              "comparison", None if not comparison else {k: round(v, 3) for k, v in comparison.items() if k != "rows"})
    write_json(ROOT / args.report, report)


if __name__ == "__main__":
    main()
