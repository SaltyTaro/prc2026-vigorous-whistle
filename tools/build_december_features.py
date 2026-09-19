"""Build one month's 335-column timeline-layout features with the frozen functions (December guards lifted); --verify reproduces a frozen month.

Usage: build_december_features.py --month 2025-11 --verify        (exact reproduction of the frozen timeline bundle's November rows)
       build_december_features.py --month 2025-12 --weather-inventory docs/december/weather_inventory.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json
from prc2026_december.features import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/december"))
    parser.add_argument("--weather-inventory", type=Path, action="append", default=[], help="Additional IEM inventory (e.g. the December pull) joined to the frozen thirty objects")
    parser.add_argument("--verify", action="store_true", help="Require exact equality with the frozen timeline bundle's rows of this month")
    args = parser.parse_args()
    record = build(ROOT, args.month, ROOT / args.output, [read_json(ROOT / p) for p in args.weather_inventory], verify=args.verify)
    print(json.dumps({k: record[k] for k in ["month", "rows", "columns", "ids_sha256", "build_id", *(["reproduces_frozen_timeline_rows_exactly"] if args.verify else [])]}))


if __name__ == "__main__":
    main()
