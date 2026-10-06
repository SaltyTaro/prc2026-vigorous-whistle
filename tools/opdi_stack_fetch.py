"""Lever 7 of the final-week campaign (docs/opdi_stack/README.md): fetch the OPDI v0.0.2 flight-event files that cover the four 2025
development months (plus their neighbouring days) and the two ranking months, keep only the surface / take-off events at the ten
challenge airports, and delete each raw download once its filtered extract and receipt are written.

Source: EUROCONTROL Open Performance Data Initiative (opdi.aero), flight events v0.0.2, derived from OpenSky Network state vectors;
open data with attribution.

  python tools/opdi_stack_fetch.py            fetch + filter every missing window, one at a time (sequential, polite)
Outputs: data/external/opdi/stack_v1/events_<start>_<end>.parquet (+ .receipt.json); docs/opdi_stack/sources.json.
The 18 September probe file (data/external/opdi/flight_events/flight_events_20251101_20251111.parquet) is read in place, never changed.
No label is read. Nothing is uploaded.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/flight_events_{}_{}.parquet"
OUT = ROOT / "data/external/opdi/stack_v1"
TMP = ROOT / "data/external/opdi/raw_tmp"
PROBE = ROOT / "data/external/opdi/flight_events"
SOURCES = ROOT / "docs/opdi_stack/sources.json"
NEED = [(date(2024, 12, 31), date(2025, 2, 1)), (date(2025, 6, 29), date(2025, 8, 1)), (date(2025, 10, 30), date(2026, 2, 1)),
        (date(2026, 6, 30), date(2026, 8, 1))]
SURFACE = ["entry-parking_position", "exit-parking_position", "entry-apron", "exit-apron", "entry-taxiway", "exit-taxiway",
           "entry-runway", "exit-runway", "entry-deicing_pad", "exit-deicing_pad", "entry-hangar", "exit-hangar", "take-off", "first_seen"]
AIRPORTS = {"EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4700, -0.4543), "EHAM": (52.3086, 4.7639),
            "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626), "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389),
            "LSZH": (47.4647, 8.5492), "LTFM": (41.2753, 28.7519)}
BOX = (0.10, 0.15)


def windows() -> list[tuple[date, date]]:
    out, d = [], date(2022, 1, 1)
    while d < date(2026, 8, 10):
        e = d + timedelta(days=10)
        if any(d < b and e > a for a, b in NEED):
            out.append((d, e))
        d = e
    return out


def in_boxes(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    m = np.zeros(len(lat), bool)
    for la, lo in AIRPORTS.values():
        m |= (np.abs(lat - la) <= BOX[0]) & (np.abs(lon - lo) <= BOX[1])
    return m


def filter_file(src: Path) -> pd.DataFrame:
    f = pq.ParquetFile(src)
    keep = pa.array(SURFACE)
    parts = []
    for b in f.iter_batches(batch_size=500_000, columns=["flight_id", "type", "event_time", "longitude", "latitude", "altitude", "info"]):
        b = b.filter(pc.is_in(b.column("type"), value_set=keep))
        if b.num_rows == 0:
            continue
        lat, lon = b.column("latitude").to_numpy(zero_copy_only=False), b.column("longitude").to_numpy(zero_copy_only=False)
        b = b.filter(pa.array(in_boxes(np.nan_to_num(lat, nan=999.0), np.nan_to_num(lon, nan=999.0))))
        if b.num_rows:
            parts.append(b.to_pandas())
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["flight_id", "type", "event_time", "longitude", "latitude", "altitude", "info"])
    info = t["info"].fillna("").map(lambda s: json.loads(s) if s.startswith("{") else {})
    t["osm_airport"] = info.map(lambda d: d.get("osm_airport"))
    t["osm_ref"] = info.map(lambda d: d.get("osm_ref"))
    t["icao24"] = info.map(lambda d: d.get("osn_icao24"))
    t["callsign"] = info.map(lambda d: (d.get("osn_flight_id") or "").strip() or None)
    t["in_use_s"] = info.map(lambda d: d.get("opdi_time_in_use_s"))
    t["event_time"] = pd.to_datetime(t["event_time"], utc=True)
    t["flight_id"] = t["flight_id"].astype("uint64")
    return t.drop(columns=["info"])


def fetch(url: str, dest: Path) -> tuple[str, int]:
    h, n = hashlib.sha256(), 0
    req = urllib.request.Request(url, headers={"User-Agent": "prc2026-vigorous-whistle/0.1 (research; contact via the PRC Data Challenge)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
                    fh.write(chunk)
                    n += len(chunk)
            return h.hexdigest(), n
        except Exception as ex:  # bounded retry, then fail loudly
            print("retry", attempt, url, ex, flush=True)
            h, n = hashlib.sha256(), 0
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    src = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else {"source": "EUROCONTROL OPDI flight events v0.0.2 (opdi.aero)", "files": {}}
    for a, b in windows():
        name = f"flight_events_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        out = OUT / f"events_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        if out.exists() and name in src["files"]:
            continue
        url = URL.format(f"{a:%Y%m%d}", f"{b:%Y%m%d}")
        probe = PROBE / name
        t0 = time.time()
        if probe.exists():
            raw, keep_raw = probe, True
            sha = hashlib.sha256(raw.read_bytes()).hexdigest()
            size = raw.stat().st_size
        else:
            raw, keep_raw = TMP / name, False
            sha, size = fetch(url, raw)
        t = filter_file(raw)
        t.to_parquet(out, index=False)
        days = t["event_time"].dt.strftime("%Y-%m-%d").value_counts().sort_index()
        rec = {"url": url, "bytes": size, "sha256": sha, "fetched_at_utc": pd.Timestamp.utcnow().isoformat(), "seconds": round(time.time() - t0, 1),
               "read_from": "18 September probe copy" if keep_raw else "download", "filtered_rows": int(len(t)),
               "filtered_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "events_by_day": {k: int(v) for k, v in days.items()},
               "by_airport": {k: int(v) for k, v in t["osm_airport"].value_counts().items()}}
        src["files"][name] = rec
        SOURCES.write_text(json.dumps(src, indent=1, sort_keys=True), encoding="utf-8")
        if not keep_raw:
            raw.unlink()
        print(name, size, len(t), round(time.time() - t0, 1), "s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
