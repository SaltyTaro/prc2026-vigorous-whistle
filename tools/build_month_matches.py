"""Pass-2 ADS-B matches for one 2025 month with a frozen stand table (generalises tools/build_december_matches.py, a v22/v23 parent).

Same per-day matching as tools/build_adsb_features.py::one_day (prc2026_adsb.features.load_surface / match_departures), label-free.
The residual stage's snapshot directory data/external/adsblol/matches_v11 (1,111 stands) holds matches2 tables only for January, July,
November and December 2025 and the ranking months; the other eight months are needed once the stage trains on twelve months.
Usage: build_month_matches.py --month 2025-02 --matches-dir data/external/adsblol/matches_v11
Writes matches2_<month>.parquet and matches2_<month>.json (receipt) there; refuses to overwrite.
"""
from __future__ import annotations

import argparse
import calendar
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now
from prc2026.features import PHASE, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, load_surface, match_departures

SURFACE = ROOT / "data/external/adsblol/surface"


def raw_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


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
    parser.add_argument("--month", required=True, help="2025-MM")
    parser.add_argument("--matches-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    month = args.month
    if month[:4] != "2025":
        raise SystemExit("Training months of 2025 only; the ranking months have their tables.")
    started = time.time()
    out = ROOT / args.matches_dir
    target = out / f"matches2_{month}.parquet"
    if target.exists():
        raise SystemExit(f"Preserve the existing {target}")
    stands = pd.read_parquet(out / "stands.parquet")
    raw = pd.read_parquet(ROOT / "data/raw" / raw_file(month), columns=DEPARTURE_COLUMNS)
    dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
    dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
    y, m = int(month[:4]), int(month[5:7])
    days = [f"{y}{m:02}{d:02}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]
    jobs = [(day, dep[dep["day"].eq(day)].drop(columns=["day"]).reset_index(drop=True), stands) for day in days]
    with mp.Pool(args.workers) as pool:
        parts = pool.map(one_day, jobs)
    matches = pd.concat([p for p in parts if len(p)], ignore_index=True)
    matches.to_parquet(target, index=False, compression="zstd")
    record = {"created_at_utc": utc_now(), "month": month, "departures": int(len(dep)), "matched": int(len(matches)), "matched_share": float(len(matches) / len(dep)),
              "gate_share": float(matches["gate"].mean()), "days_with_surface": int(sum((SURFACE / f"surface_{d}.parquet").exists() for d in days)),
              "stands": int(len(stands)), "stands_sha256": sha256_file(out / "stands.parquet"), "output": target.relative_to(ROOT).as_posix(), "output_sha256": sha256_file(target),
              "surface_sha256": {d: sha256_file(SURFACE / f"surface_{d}.parquet") for d in days if (SURFACE / f"surface_{d}.parquet").exists()},
              "by_airport_matched_share": {a: float(matches["airport"].eq(a).sum() / dep["ADEP_mvt"].eq(a).sum()) for a in sorted(dep["ADEP_mvt"].unique())} if "airport" in matches else None,
              "code_sha256": sha256_file(Path(__file__)), "features_code_sha256": sha256_file(ROOT / "src/prc2026_adsb/features.py"), "seconds": round(time.time() - started, 1)}
    (out / f"matches2_{month}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print({k: record[k] for k in ["month", "departures", "matched", "matched_share", "gate_share", "days_with_surface", "seconds"]})


if __name__ == "__main__":
    main()
