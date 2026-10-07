"""Lever 9 (docs/ctx_opdi/README.md): fetch the OPDI v0.0.2 flight-event windows that the context-network folds need and lever 7 did not
download (all of 2025 plus January / July 2026), filtered exactly as lever 7 (tools/opdi_stack_fetch.filter_file: surface / take-off
events in boxes around the ten airports). Windows already filtered by lever 7 (data/external/opdi/stack_v1/) are read in place, never
rewritten. Each raw download is deleted once its extract and receipt are written. Sequential, one request at a time.

Source: EUROCONTROL Open Performance Data Initiative (opdi.aero), flight events v0.0.2, derived from OpenSky Network state vectors; open data
with attribution. No label is read.

  python tools/ctx_opdi_fetch.py      -> data/external/opdi/ctx_v1/events_<start>_<end>.parquet, docs/ctx_opdi/sources.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import opdi_stack_fetch as F  # noqa: E402

OUT = ROOT / "data/external/opdi/ctx_v1"
LEVER7 = ROOT / "data/external/opdi/stack_v1"
TMP = ROOT / "data/external/opdi/raw_tmp_ctx"
SOURCES = ROOT / "docs/ctx_opdi/sources.json"
NEED = [(date(2024, 12, 31), date(2026, 2, 1)), (date(2026, 6, 30), date(2026, 8, 1))]


def windows() -> list[tuple[date, date]]:
    out, d = [], date(2022, 1, 1)
    while d < date(2026, 8, 10):
        e = d + timedelta(days=10)
        if any(d < b and e > a for a, b in NEED):
            out.append((d, e))
        d = e
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    SOURCES.parent.mkdir(parents=True, exist_ok=True)
    src = json.loads(SOURCES.read_text(encoding="utf-8")) if SOURCES.exists() else {
        "source": "EUROCONTROL OPDI flight events v0.0.2 (opdi.aero), derived from OpenSky Network state vectors; open data with attribution",
        "filter": "tools/opdi_stack_fetch.filter_file (lever 7), unchanged", "files": {}, "read_in_place_from_lever7": {}}
    for a, b in windows():
        name = f"flight_events_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        stem = f"events_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        if (LEVER7 / stem).exists():
            src["read_in_place_from_lever7"][stem] = hashlib.sha256((LEVER7 / stem).read_bytes()).hexdigest()
            continue
        out = OUT / stem
        if out.exists() and name in src["files"]:
            continue
        url = F.URL.format(f"{a:%Y%m%d}", f"{b:%Y%m%d}")
        t0 = time.time()
        raw = TMP / name
        sha, size = F.fetch(url, raw)
        t = F.filter_file(raw)
        t.to_parquet(out, index=False)
        days = t["event_time"].dt.strftime("%Y-%m-%d").value_counts().sort_index()
        src["files"][name] = {"url": url, "bytes": size, "sha256": sha, "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "seconds": round(time.time() - t0, 1), "filtered_rows": int(len(t)),
                              "filtered_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                              "events_by_day": {k: int(v) for k, v in days.items()},
                              "by_airport": {str(k): int(v) for k, v in t["osm_airport"].value_counts().items()}}
        SOURCES.write_text(json.dumps(src, indent=1, sort_keys=True), encoding="utf-8")
        raw.unlink()
        print(name, size, len(t), round(time.time() - t0, 1), "s", flush=True)
    SOURCES.write_text(json.dumps(src, indent=1, sort_keys=True), encoding="utf-8")
    print("done", len(src["files"]), "downloaded,", len(src["read_in_place_from_lever7"]), "read in place", flush=True)


if __name__ == "__main__":
    sys.exit(main())
