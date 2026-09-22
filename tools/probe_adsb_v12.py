"""Measure what each matcher fix recovers, against the frozen matcher, on sampled days. Read-only and label-free.

For every requested day this runs the frozen matcher (prc2026_adsb.features.match_departures) and each
configuration of the variant matcher (prc2026_adsb_v12.matching), and reports how many departures move
between the three classes (unmatched / dark / gate). The `base` configuration must reproduce the frozen
matcher row for row; the run fails if it does not.

No label, block time or taxi time is read, nothing is fitted and nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TIME  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS as FROZEN_COLUMNS, load_surface  # noqa: E402
from prc2026_adsb.features import match_departures as frozen_match  # noqa: E402
from prc2026_adsb_v12.matching import DEPARTURE_COLUMNS as V12_COLUMNS  # noqa: E402
from prc2026_adsb_v12.matching import match_departures as v12_match  # noqa: E402

SURFACE = ROOT / "data/external/adsblol/surface"
STANDS = ROOT / "data/external/adsblol/matches_v11/stands.parquet"

CONFIGS = {
    "base": {},
    "minpts1": {"min_ground_points": 1},
    "variant": {"callsign_variant": True},
    "radius100": {"stand_radius_m": 100.0},
    "all": {"min_ground_points": 1, "callsign_variant": True, "stand_radius_m": 100.0},
}


def raw_path(month: str) -> Path:
    y, m = int(month[:4]), int(month[5:7])
    if y == 2026:
        return ROOT / "data/raw/ranking.parquet"
    return ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def classify(departures: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    out = pd.Series("unmatched", index=departures[ID].to_numpy())
    if len(matches):
        m = matches.set_index(ID)
        out.loc[m.index] = m["gate"].map({True: "gate", False: "dark"}).to_numpy()
    return out


def one_day(job):
    day, dep, stands = job
    surface = load_surface(SURFACE, [day])
    if surface.empty or dep.empty:
        return day, {}, None
    frozen = frozen_match(dep[[c for c in FROZEN_COLUMNS if c in dep.columns]], surface, stands)
    result = {"frozen": classify(dep, frozen)}
    for name, options in CONFIGS.items():
        result[name] = classify(dep, v12_match(dep, surface, stands, **options))
    base = v12_match(dep, surface, stands)
    same = len(base) == len(frozen) and (
        base.sort_values(ID)[["gate", "motion_start", "segment_start", "points"]].reset_index(drop=True)
        .equals(frozen.sort_values(ID)[["gate", "motion_start", "segment_start", "points"]].reset_index(drop=True)))
    return day, {k: v.value_counts().to_dict() for k, v in result.items()}, bool(same)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    parser.add_argument("--days", nargs="+", required=True, help="days of the month, e.g. 03 09 15 21 27")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("docs/adsb_coverage/matcher_probe.json"))
    args = parser.parse_args()
    y, m = int(args.month[:4]), int(args.month[5:7])
    stands = pd.read_parquet(STANDS)
    raw = pd.read_parquet(raw_path(args.month), columns=sorted(set(FROZEN_COLUMNS) | set(V12_COLUMNS)))
    dep = raw[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq(args.month)].reset_index(drop=True)
    dep["day"] = dep[TIME].dt.strftime("%Y%m%d")
    jobs = [(f"{y}{m:02}{int(d):02}", dep[dep["day"].eq(f"{y}{m:02}{int(d):02}")].drop(columns=["day"]).reset_index(drop=True), stands)
            for d in args.days]
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        parts = pool.map(one_day, jobs)
    totals, reproduces = {}, []
    for day, counts, same in parts:
        reproduces.append(same)
        for name, classes in counts.items():
            entry = totals.setdefault(name, {"unmatched": 0, "dark": 0, "gate": 0})
            for k, v in classes.items():
                entry[k] += int(v)
    if not all(r is True for r in reproduces if r is not None):
        raise SystemExit(f"The base configuration does not reproduce the frozen matcher: {reproduces}")
    rows = sum(totals["frozen"].values())
    report = {"created_at_utc": utc_now(), "month": args.month, "days": args.days, "departures": rows,
              "labels_read": False, "audit_used": False,
              "base_reproduces_frozen_matcher": True,
              "inputs": {"stands": sha256_file(STANDS)},
              "configurations": {name: {"counts": c, "pct": {k: round(100 * v / rows, 2) for k, v in c.items()},
                                        "matched_pct": round(100 * (c["dark"] + c["gate"]) / rows, 2),
                                        "options": CONFIGS.get(name, "frozen matcher")}
                                 for name, c in totals.items()}}
    for name, c in totals.items():
        if name == "frozen":
            continue
        f = totals["frozen"]
        report["configurations"][name]["vs_frozen"] = {
            "matched_rows_gained": (c["dark"] + c["gate"]) - (f["dark"] + f["gate"]),
            "gate_rows_gained": c["gate"] - f["gate"],
            "matched_pct_gained": round(100 * ((c["dark"] + c["gate"]) - (f["dark"] + f["gate"])) / rows, 2),
            "gate_pct_gained": round(100 * (c["gate"] - f["gate"]) / rows, 2)}
    write_json(ROOT / args.output, report)
    print(json.dumps({n: {"matched_pct": v["matched_pct"], "gate_pct": v["pct"].get("gate"),
                          "gained": v.get("vs_frozen", {}).get("matched_pct_gained")} for n, v in report["configurations"].items()}, indent=1))


if __name__ == "__main__":
    main()
