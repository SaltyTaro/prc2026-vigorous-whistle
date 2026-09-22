"""Read-only census of WHY departures fail to reach an adsb.lol surface segment (or reach one without a stand observation).

For every departure of the requested months this replays the matcher's own steps
(prc2026_adsb.features.match_departures) and records the first step that rejects the row:

  no_archive_day            the day's surface file is absent                    (download gap)
  airport_absent            the archive holds no points for that airport         (no receiver coverage)
  no_callsign               the movement record carries no callsign              (nothing to join on)
  callsign_absent           the callsign never appears at that airport that day  (variant flag records a near-miss)
  no_ground_points          the aircraft is seen, but never on the ground there
  few_ground_points         fewer than the three ground points the matcher needs
  no_airborne_after         no airborne point in the 15 minutes after takeoff
  segment_rejected          the gates pass but segment() returns nothing
  dark                      matched, but the aircraft was never observed standing
  gate                      matched with the stand stop observed

No label, block time or taxi time is read anywhere: the inputs are the movement identifiers,
clocks and stand names of DEPARTURE_COLUMNS, the surface archive and the learned stand table.
Nothing is fitted and nothing is written outside the report.
"""
from __future__ import annotations

import argparse
import calendar
import multiprocessing as mp
import re
import sys
from pathlib import Path


import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import environment, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TIME  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS, EPOCH, load_surface, segment  # noqa: E402

SURFACE = ROOT / "data/external/adsblol/surface"
STANDS = ROOT / "data/external/adsblol/matches_v11/stands.parquet"
DIGITS = re.compile(r"(\d+)")

REASONS = ["no_archive_day", "airport_absent", "no_callsign", "callsign_absent", "no_ground_points",
           "few_ground_points", "no_airborne_after", "segment_rejected", "dark", "gate"]


def raw_path(month: str) -> Path:
    y, m = int(month[:4]), int(month[5:7])
    if y == 2026:
        return ROOT / "data/raw/ranking.parquet"
    return ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def suffix(callsign: str) -> str:
    """The trailing number of a callsign: AFR1234 -> 1234. Used only to flag a near-miss, never to match."""
    found = DIGITS.findall(callsign)
    return found[-1].lstrip("0") if found else ""


def census_day(job):
    day, dep, stands = job
    surface = load_surface(SURFACE, [day])
    rows = []
    if surface.empty:
        for row in dep.itertuples(index=False):
            rows.append({ID: getattr(row, ID), "airport": row.ADEP_mvt, "day": day, "reason": "no_archive_day"})
        return pd.DataFrame(rows), []
    stand_lookup = {(r.airport, r.stand): (r.lat, r.lon) for r in stands.itertuples(index=False)} if stands is not None and len(stands) else {}
    dep = dep.copy()
    dep["takeoff"] = (dep[TIME] - EPOCH).dt.total_seconds()
    dep["callsign"] = dep["CALLSIGN_flt"].fillna(dep["FLIGHT_mvt"]).fillna("").str.strip().str.upper()
    day_stats = []
    for airport, group in dep.groupby("ADEP_mvt"):
        s = surface[surface["airport"].eq(airport)]
        if s.empty:
            for row in group.itertuples(index=False):
                rows.append({ID: getattr(row, ID), "airport": airport, "day": day, "reason": "airport_absent"})
            continue
        ground = s[s["ground"]]
        named = s[s["callsign"].ne("")]
        by_callsign = {c: frame["icao"].unique() for c, frame in named.groupby("callsign")}
        by_icao = {i: frame for i, frame in s.groupby("icao")}
        by_suffix = {}
        for c in by_callsign:
            by_suffix.setdefault(suffix(c), []).append(c)
        # Callsign coverage of the day's ground points: points whose callsign is blank are invisible to the index.
        ground_named = ground[ground["callsign"].ne("")]
        day_stats.append({"day": day, "airport": airport, "points": int(len(s)), "ground_points": int(len(ground)),
                          "ground_points_without_callsign": int(len(ground) - len(ground_named)),
                          "icao_on_ground": int(ground["icao"].nunique()),
                          "icao_on_ground_without_any_callsign": int(ground["icao"].nunique() - ground_named["icao"].nunique()),
                          "departures": int(len(group))})
        for row in group.itertuples(index=False):
            record = {ID: getattr(row, ID), "airport": airport, "day": day}
            call = row.callsign
            if not call:
                rows.append({**record, "reason": "no_callsign"})
                continue
            icaos = by_callsign.get(call)
            if icaos is None:
                variants = [c for c in by_suffix.get(suffix(call), []) if c != call]
                rows.append({**record, "reason": "callsign_absent", "variant_present": bool(variants),
                             "variant_example": variants[0] if variants else None})
                continue
            stand = stand_lookup.get((airport, row.STAND_mvt)) if isinstance(row.STAND_mvt, str) else None
            best, best_before, best_after = None, 0, 0
            for icao in icaos:
                track = by_icao[icao]
                before = track[(track["time"] <= row.takeoff + 30) & (track["time"] >= row.takeoff - 4 * 3600) & track["ground"]]
                after = track[(track["time"] >= row.takeoff - 30) & (track["time"] <= row.takeoff + 900) & ~track["ground"]]
                best_before, best_after = max(best_before, len(before)), max(best_after, len(after))
                if len(before) >= 3 and len(after) >= 1:
                    info = segment(track, row.takeoff, stand=stand)
                    if info is not None and (best is None or abs(info["last_ground"] - row.takeoff) < abs(best["last_ground"] - row.takeoff)):
                        best = info
            if best is not None:
                rows.append({**record, "reason": "gate" if best["gate"] else "dark", "stand_known": stand is not None,
                             "points": best["points"], "first_gs": best["first_gs"],
                             "first_dist_stand_m": best["first_dist_stand_m"], "gap_before_s": best["gap_before_s"]})
            elif best_before == 0:
                rows.append({**record, "reason": "no_ground_points"})
            elif best_before < 3:
                rows.append({**record, "reason": "few_ground_points", "points": int(best_before)})
            elif best_after == 0:
                rows.append({**record, "reason": "no_airborne_after", "points": int(best_before)})
            else:
                rows.append({**record, "reason": "segment_rejected", "points": int(best_before)})
    return pd.DataFrame(rows), day_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("docs/adsb_coverage/census.json"))
    parser.add_argument("--rows-output", type=Path, default=Path("runs/adsb_coverage"))
    args = parser.parse_args()
    stands = pd.read_parquet(STANDS)
    report = {"created_at_utc": utc_now(), "audit_used": False, "labels_read": False,
              "inputs": {"stands": sha256_file(STANDS), "surface_days_present": len(list(SURFACE.glob("surface_*.parquet")))},
              "months": {}}
    (ROOT / args.rows_output).mkdir(parents=True, exist_ok=True)
    for month in args.months:
        y, m = int(month[:4]), int(month[5:7])
        raw = pd.read_parquet(raw_path(month), columns=DEPARTURE_COLUMNS)
        dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
        dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
        jobs = [(f"{y}{m:02}{d:02}", dep[dep["day"].eq(f"{y}{m:02}{d:02}")].drop(columns=["day"]).reset_index(drop=True), stands)
                for d in range(1, calendar.monthrange(y, m)[1] + 1)]
        with mp.Pool(args.workers) as pool:
            parts = pool.map(census_day, jobs)
        table = pd.concat([p for p, _ in parts if len(p)], ignore_index=True)
        stats = pd.DataFrame([s for _, ss in parts for s in ss])
        table.to_parquet(ROOT / args.rows_output / f"census_{month}.parquet", index=False, compression="zstd")
        counts = table["reason"].value_counts()
        by_airport = table.pivot_table(index="airport", columns="reason", values=ID, aggfunc="count").fillna(0).astype(int)
        for reason in REASONS:
            if reason not in by_airport:
                by_airport[reason] = 0
        by_airport = by_airport[REASONS]
        variant = table[table["reason"].eq("callsign_absent")]
        entry = {
            "departures": int(len(dep)), "rows_censused": int(len(table)),
            "reasons": {r: int(counts.get(r, 0)) for r in REASONS},
            "reasons_pct": {r: round(100 * float(counts.get(r, 0)) / len(table), 2) for r in REASONS},
            "by_airport": {a: {r: int(v) for r, v in by_airport.loc[a].items()} for a in by_airport.index},
            "callsign_absent_with_variant_present": int(variant["variant_present"].fillna(False).sum()) if len(variant) else 0,
            "ground_points_without_callsign": {a: {"ground_points": int(g["ground_points"].sum()),
                                                   "without_callsign": int(g["ground_points_without_callsign"].sum()),
                                                   "icao_on_ground": int(g["icao_on_ground"].sum()),
                                                   "icao_without_any_callsign": int(g["icao_on_ground_without_any_callsign"].sum())}
                                               for a, g in stats.groupby("airport")} if len(stats) else {},
            "days_without_archive": sorted(set(table.loc[table["reason"].eq("no_archive_day"), "day"])),
        }
        dark = table[table["reason"].eq("dark")]
        if len(dark):
            entry["dark_detail"] = {"rows": int(len(dark)), "stand_known_pct": round(100 * float(dark["stand_known"].fillna(False).mean()), 2),
                                    "median_points": float(dark["points"].median()),
                                    "median_first_gs_kt": float(dark["first_gs"].median()),
                                    "first_point_within_100m_of_known_stand_pct": round(100 * float((dark["first_dist_stand_m"] <= 100).fillna(False).mean()), 2),
                                    "median_first_dist_stand_m": float(dark["first_dist_stand_m"].median(skipna=True))}
        report["months"][month] = entry
        print(month, entry["reasons_pct"], flush=True)
    write_json(ROOT / args.output, report)
    report_env = environment(ROOT)
    print("written", args.output, report_env["python"])


if __name__ == "__main__":
    main()
