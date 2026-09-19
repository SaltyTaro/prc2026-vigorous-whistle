"""Pass-2 ADS-B matches for December 2025 with a frozen stand table (the December guard of tools/build_adsb_features.py is a v22 parent, so this is a separate tool).

Same per-day matching as tools/build_adsb_features.py::one_day (prc2026_adsb.features.load_surface / match_departures), label-free.
Usage: build_december_matches.py --matches-dir data/external/adsblol/matches      (stands.parquet of that directory; writes matches2_2025-12.parquet there)
"""
from __future__ import annotations

import argparse
import calendar
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file
from prc2026.features import ID, PHASE, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, load_surface, match_departures

SURFACE = ROOT / "data/external/adsblol/surface"
MONTH, RAW = "2025-12", "training_2025-12-01_2026-01-01.parquet"


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
    parser.add_argument("--matches-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = ROOT / args.matches_dir
    target = out / f"matches2_{MONTH}.parquet"
    if target.exists():
        raise SystemExit(f"Preserve the existing {target}")
    stands = pd.read_parquet(out / "stands.parquet")
    raw = pd.read_parquet(ROOT / "data/raw" / RAW, columns=DEPARTURE_COLUMNS)
    dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(MONTH)].reset_index(drop=True)
    dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
    days = [f"202512{d:02}" for d in range(1, calendar.monthrange(2025, 12)[1] + 1)]
    jobs = [(day, dep[dep["day"].eq(day)].drop(columns=["day"]).reset_index(drop=True), stands) for day in days]
    with mp.Pool(args.workers) as pool:
        parts = pool.map(one_day, jobs)
    matches = pd.concat([p for p in parts if len(p)], ignore_index=True)
    matches.to_parquet(target, index=False, compression="zstd")
    record = {"month": MONTH, "departures": int(len(dep)), "matched": int(len(matches)), "matched_share": float(len(matches) / len(dep)),
              "gate_share": float(matches["gate"].mean()), "days_with_surface": int(sum((SURFACE / f"surface_{d}.parquet").exists() for d in days)),
              "stands": int(len(stands)), "stands_sha256": sha256_file(out / "stands.parquet"), "output": target.relative_to(ROOT).as_posix(), "output_sha256": sha256_file(target),
              "surface_sha256": {d: sha256_file(SURFACE / f"surface_{d}.parquet") for d in days if (SURFACE / f"surface_{d}.parquet").exists()},
              "by_airport_matched_share": {a: float(matches["airport"].eq(a).sum() / dep["ADEP_mvt"].eq(a).sum()) for a in sorted(dep["ADEP_mvt"].unique())} if "airport" in matches else None}
    (out / f"matches2_{MONTH}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ["month", "departures", "matched", "matched_share", "gate_share", "days_with_surface", "by_airport_matched_share"]}))


if __name__ == "__main__":
    main()
