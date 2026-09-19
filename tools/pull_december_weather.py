"""Cache the ten stations' IEM observations for December 2025 (public domain) into a separate directory; the frozen 2025 inventory is untouched.

Same request, parsing and receipt code as tools/pull_prc_weather.py (imported); output data/external/iem_december/ and
docs/december/weather_inventory.json in the frozen inventory's format so the December feature build can join it.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from pull_prc_weather import LICENSE_URL, STATIONS, ProviderCooldown, fetch, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "data/external/iem_december"
    license_path = ROOT / "docs/competition/weather_20260914/iem_usage.html"
    results, failures = [], []
    for station in STATIONS:
        try:
            result = fetch(station, "december", "2025-12-01", "2026-01-01", directory, args.offline)
            results.append(result)
            print(json.dumps({k: result[k] for k in ["station", "period", "rows_in_period", "observed_days", "expected_days"]}), flush=True)
        except Exception as exc:
            failures.append({"station": station, "period": "december", "error": str(exc)})
            print(json.dumps(failures[-1]), flush=True)
            if isinstance(exc, ProviderCooldown):
                break
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "provider": "Iowa Environmental Mesonet",
              "license": {"source_url": LICENSE_URL, "statement": "IEM website materials are public domain; attribution appreciated",
                          "saved_path": license_path.relative_to(ROOT).as_posix(), "sha256": sha(license_path.read_bytes())},
              "purpose": "December 2025 covariates for the opened December month (docs/december/authorization_20260919.json); no labels read",
              "objects": sorted(results, key=lambda item: item["station"]), "failures": failures}
    output = ROOT / "docs/december/weather_inventory.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": output.relative_to(ROOT).as_posix(), "successful_objects": len(results), "failed_objects": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
