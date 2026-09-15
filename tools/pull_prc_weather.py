"""Cache openly licensed IEM airport surface observations; never read labels.

This is an input-coverage audit for a later declared experiment. None of these
weather observations are predictors in the frozen Phase-5R initial batch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
PERIODS = [("train", "2025-01-01", "2025-12-01"), ("ranking_january", "2026-01-01", "2026-02-01"),
           ("ranking_july", "2026-07-01", "2026-08-01")]
VARIABLES = ["tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "alti", "mslp", "vsby", "gust",
             "skyc1", "skyc2", "skyc3", "skyc4", "skyl1", "skyl2", "skyl3", "skyl4", "wxcodes"]
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
LICENSE_URL = "https://mesonet.agron.iastate.edu/disclaimer.php"
USER_AGENT = "PRC2026-vigorous-whistle public weather research; bounded historical GET requests"


class ProviderCooldown(RuntimeError):
    """Stop new requests when the provider needs a longer cooldown."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(station, period, start, end, directory, offline):
    begin, finish = pd.Timestamp(start), pd.Timestamp(end)
    parameters = [("station", station), *[("data", name) for name in VARIABLES],
                  ("year1", str(begin.year)), ("month1", str(begin.month)), ("day1", str(begin.day)),
                  ("year2", str(finish.year)), ("month2", str(finish.month)), ("day2", str(finish.day)),
                  ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"), ("elev", "no"),
                  ("missing", "M"), ("trace", "T"), ("direct", "no"), ("report_type", "3"), ("report_type", "4")]
    filename = directory/f"{station}_{period}.csv"
    meta_path = filename.with_suffix(".json")
    if filename.is_file() and meta_path.is_file():
        data = filename.read_bytes()
        receipt = json.loads(meta_path.read_text())
        if receipt["sha256"] != sha(data):
            raise ValueError("Cached observation hash mismatch.")
        if any(receipt[key] != value for key, value in [("station", station), ("period", period),
                                                        ("start_inclusive", start), ("end_exclusive", end)]):
            raise ValueError("Cached observation identity differs from the requested station/period.")
    else:
        if offline:
            raise ValueError("Requested weather cache object is absent.")
        # The provider throttles concurrent requests. Keep one request in
        # flight, with a gap and bounded Retry-After handling.
        time.sleep(2)
        for attempt in range(3):
            response = requests.get(BASE_URL, params=parameters, timeout=(15, 90), headers={"User-Agent": USER_AGENT})
            if response.status_code != 429 or attempt == 2:
                break
            retry = response.headers.get("Retry-After", "15")
            delay = float(retry) if retry.isdigit() else 15.0
            if delay > 60:
                raise ProviderCooldown(f"Provider requests a {delay:g}-second cooldown; stop and resume later.")
            time.sleep(max(2.0, delay))
        if response.status_code == 429:
            raise ProviderCooldown("Provider still throttles requests after bounded backoff; stop and resume later.")
        response.raise_for_status()
        data = response.content
        if len(data) > 30_000_000 or not response.text.lstrip().startswith("station,valid"):
            raise ValueError("Provider did not return the expected bounded observation CSV.")
        receipt = {"url": response.url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "sha256": sha(data),
                   "size_bytes": len(data), "station": station, "period": period, "start_inclusive": start,
                   "end_exclusive": end, "license_url": LICENSE_URL, "path": filename.relative_to(ROOT).as_posix()}
        directory.mkdir(parents=True, exist_ok=True)
        filename.write_bytes(data)
        meta_path.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    frame = pd.read_csv(StringIO(data.decode("utf-8-sig")), na_values=["M"])
    if "valid" not in frame or not frame["station"].eq(station).all():
        raise ValueError("Observation station/schema mismatch.")
    times = pd.to_datetime(frame["valid"], utc=True, errors="raise")
    wanted = times.ge(pd.Timestamp(start, tz="UTC")) & times.lt(pd.Timestamp(end, tz="UTC"))
    frame, times = frame.loc[wanted], times.loc[wanted]
    return {**receipt, "rows_in_period": len(frame), "unique_observation_times": int(times.nunique()),
            "first_time_utc": None if times.empty else times.min().isoformat(),
            "last_time_utc": None if times.empty else times.max().isoformat(),
            "observed_days": int(times.dt.floor("D").nunique()),
            "expected_days": int((finish-begin).days),
            "nonnull_fraction": {name: float(frame[name].notna().mean()) if len(frame) and name in frame else 0.0 for name in VARIABLES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Check one day at each station before downloading the full periods.")
    args = parser.parse_args()
    directory = ROOT/"data/external/iem"
    docs = ROOT/"docs/competition/weather_20260914"
    docs.mkdir(parents=True, exist_ok=True)
    license_path = docs/"iem_usage.html"
    if not license_path.exists():
        if args.offline:
            raise ValueError("IEM usage statement is absent.")
        response = requests.get(LICENSE_URL, timeout=(15, 45), headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        if "public domain" not in response.text.lower():
            raise ValueError("Review the provider's changed usage statement before data acquisition.")
        license_path.write_bytes(response.content)
    periods = [("probe", "2025-01-15", "2025-01-16")] if args.probe else PERIODS
    results, failures = [], []
    tasks = [(station, period, start, end) for station in STATIONS for period, start, end in periods]
    for station, period, start, end in tasks:
        try:
            result = fetch(station, period, start, end, directory, args.offline)
            results.append(result)
            print(json.dumps({k: result[k] for k in ["station", "period", "rows_in_period", "observed_days", "expected_days"]}), flush=True)
        except Exception as exc:
            failure = {"station": station, "period": period, "error": str(exc)}
            failures.append(failure)
            print(json.dumps(failure), flush=True)
            if isinstance(exc, ProviderCooldown):
                break
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "provider": "Iowa Environmental Mesonet",
              "license": {"source_url": LICENSE_URL, "statement": "IEM website materials are public domain; attribution appreciated",
                          "saved_path": license_path.relative_to(ROOT).as_posix(), "sha256": sha(license_path.read_bytes())},
              "purpose": "Public covariate coverage only; no labels read, no weather model trained or scored",
              "used_by_phase5r_initial_models": False, "audit_labels_used": False,
              "objects": sorted(results, key=lambda item: (item["station"], item["period"])), "failures": failures}
    output = docs/("probe.json" if args.probe else "inventory.json")
    output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"report": output.relative_to(ROOT).as_posix(), "successful_objects": len(results), "failed_objects": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
