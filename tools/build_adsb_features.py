"""Match departures to adsb.lol surface segments for whole months (day-parallel) and write label-free features.

Reads the raw movement files (departure identifiers, clocks and stand names only;
no labels), the extracted surface parquet per day, writes one matches parquet
per month under data/external/adsblol/matches/ and learns stand positions from
gate-covered segments of the requested months. December is refused.
"""
from __future__ import annotations

import argparse
import calendar
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.features import ID, PHASE, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, learn_stands, load_surface, match_departures

SURFACE = ROOT / "data/external/adsblol/surface"
OUTPUT = ROOT / "data/external/adsblol/matches"


def raw_path(month: str) -> Path:
    y, m = int(month[:4]), int(month[5:7])
    if y == 2026:
        return ROOT / "data/raw/ranking.parquet"
    return ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def one_day(args):
    day, departures, stands = args
    surface = load_surface(SURFACE, [day])
    if surface.empty or departures.empty:
        return pd.DataFrame()
    matches = match_departures(departures, surface, stands)
    matches["day"] = day
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--stands", action="store_true", help="Second pass: use learned stand positions (stands.parquet) and write matches2_*.parquet")
    args = parser.parse_args()
    if "2025-12" in args.months:
        raise SystemExit("December remains reserved.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stands = pd.read_parquet(OUTPUT / "stands.parquet") if args.stands else None
    prefix = "matches2" if args.stands else "matches"
    for month in args.months:
        target = OUTPUT / f"{prefix}_{month}.parquet"
        if target.exists():
            print(month, "exists")
            continue
        y, m = int(month[:4]), int(month[5:7])
        raw = pd.read_parquet(raw_path(month), columns=DEPARTURE_COLUMNS)
        dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
        dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
        jobs = [(f"{y}{m:02}{d:02}", dep[dep["day"].eq(f"{y}{m:02}{d:02}")].drop(columns=["day"]).reset_index(drop=True), stands) for d in range(1, calendar.monthrange(y, m)[1] + 1)]
        with mp.Pool(args.workers) as pool:
            parts = pool.map(one_day, jobs)
        matches = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame()
        matches.to_parquet(target, index=False, compression="zstd")
        print(month, "departures", len(dep), "matched", len(matches), "gate share", round(float(matches["gate"].mean()), 3) if len(matches) else None, flush=True)
    if not args.stands:
        frames = [pd.read_parquet(p) for p in sorted(OUTPUT.glob("matches_2025-*.parquet"))]
        if frames:
            learned = learn_stands(pd.concat(frames, ignore_index=True))
            learned.to_parquet(OUTPUT / "stands.parquet", index=False)
            print("stands learned from 2025 months:", len(learned))


if __name__ == "__main__":
    main()
