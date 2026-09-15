"""Extract airport surface traces for one day from the adsb.lol globe_history archive (ODbL).

Streams the day's tar parts from GitHub, decodes every readsb trace, keeps only
positions inside the ten competition airports' boxes that are on the ground or
below 3,000 ft, and writes them as one parquet per day plus a manifest with the
archive part hashes. Nothing else is stored. Runs on an Azure worker.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing as mp
import subprocess
import sys
import tarfile
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BOXES = {  # generous aerodrome boxes: lat_min, lat_max, lon_min, lon_max
    "EDDF": (49.97, 50.09, 8.47, 8.66), "EDDM": (48.31, 48.40, 11.72, 11.86), "EGLL": (51.43, 51.51, -0.52, -0.39),
    "EHAM": (52.26, 52.41, 4.68, 4.83), "LEBL": (41.26, 41.33, 2.04, 2.13), "LEMD": (40.43, 40.55, -3.62, -3.51),
    "LFPG": (48.97, 49.05, 2.50, 2.62), "LIRF": (41.76, 41.85, 12.20, 12.30), "LSZH": (47.42, 47.50, 8.50, 8.60),
    "LTFM": (41.22, 41.32, 28.68, 28.80)}
COLUMNS = ["icao", "registration", "type", "airport", "time", "lat", "lon", "ground", "alt_baro", "gs", "track", "callsign", "squawk", "on_ground_flag"]


def airport_of(lat: float, lon: float) -> str | None:
    for name, (a, b, c, d) in BOXES.items():
        if a <= lat <= b and c <= lon <= d:
            return name
    return None


def decode(name: str, payload: bytes) -> list[list] | None:
    """Rows inside the aerodrome boxes; None when the member's gzip stream is corrupt (skipped and counted)."""
    if payload[:2] == b"\x1f\x8b":
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError, zlib.error):
            return None
    try:
        record = json.loads(payload)
    except Exception:
        return []
    base = record.get("timestamp")
    trace = record.get("trace")
    if base is None or not trace:
        return []
    icao, reg, typ = record.get("icao"), record.get("r"), record.get("t")
    out, callsign, squawk = [], None, None
    for point in trace:
        try:
            dt, lat, lon, alt = point[0], point[1], point[2], point[3]
        except (IndexError, TypeError):
            continue
        extra = point[8] if len(point) > 8 and isinstance(point[8], dict) else None
        if extra:
            callsign = extra.get("flight", callsign)
            squawk = extra.get("squawk", squawk)
        if lat is None or lon is None:
            continue
        airport = airport_of(lat, lon)
        if airport is None:
            continue
        ground = alt == "ground"
        altitude = None if ground or alt is None else float(alt)
        if not ground and (altitude is None or altitude > 3000):
            continue
        flags = point[6] if len(point) > 6 else 0
        out.append([icao, reg, typ, airport, base + dt, lat, lon, ground, altitude, point[4] if len(point) > 4 else None,
                    point[5] if len(point) > 5 else None, callsign, squawk, bool(flags & 1) if isinstance(flags, int) else None])
    return out


def worker(batch: list[tuple[str, bytes]]) -> tuple[list[list], list[str]]:
    rows, corrupt = [], []
    for name, payload in batch:
        decoded = decode(name, payload)
        if decoded is None:
            corrupt.append(name)
        else:
            rows.extend(decoded)
    return rows, corrupt


def release_assets(day: str) -> list[dict]:
    year = day[:4]
    tag = f"v{day}-planes-readsb-prod-0"
    url = f"https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"
    payload = subprocess.run(["curl", "-sL", "-H", "Accept: application/vnd.github+json", url], check=True, capture_output=True).stdout
    release = json.loads(payload)
    if "assets" not in release:
        raise RuntimeError(f"No release for {tag}: {release.get('message')}")
    assets = sorted(release["assets"], key=lambda a: a["name"])
    return [{"name": a["name"], "size": a["size"], "url": a["browser_download_url"]} for a in assets if ".tar" in a["name"]]


class HashingStream:
    """Sequential reader over the concatenated tar parts that also hashes each part."""

    def __init__(self, assets: list[dict]):
        self.assets, self.index, self.process, self.hashes, self.digest = assets, -1, None, [], None
        self._next()

    def _next(self):
        if self.process is not None:
            self.process.wait()
            self.hashes.append({"name": self.assets[self.index]["name"], "sha256": self.digest.hexdigest()})
        self.index += 1
        if self.index >= len(self.assets):
            self.process = None
            return
        self.digest = hashlib.sha256()
        self.process = subprocess.Popen(["curl", "-sL", self.assets[self.index]["url"]], stdout=subprocess.PIPE)

    def read(self, size: int = -1) -> bytes:
        chunks, remaining = [], size
        while self.process is not None and (remaining > 0 or size < 0):
            chunk = self.process.stdout.read(remaining if size > 0 else 1 << 20)
            if not chunk:
                self._next()
                continue
            self.digest.update(chunk)
            chunks.append(chunk)
            if size > 0:
                remaining -= len(chunk)
        return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", required=True, help="YYYY.MM.DD")
    parser.add_argument("--output", type=Path, default=Path("data/external/adsblol/surface"))
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f"surface_{args.day.replace('.', '')}.parquet"
    manifest_path = target.with_suffix(".json")
    if target.exists() and manifest_path.exists():
        print(f"{target} exists; skipping")
        return
    started = time.time()
    assets = release_assets(args.day)
    stream = HashingStream(assets)
    rows, corrupt, members, batch, pending = [], [], 0, [], []

    def collect(job) -> None:
        got_rows, got_corrupt = job.get()
        rows.extend(got_rows)
        corrupt.extend(got_corrupt)

    with mp.Pool(args.workers) as pool, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if not member.isfile() or "/traces/" not in member.name:
                continue
            payload = archive.extractfile(member).read()
            batch.append((member.name, payload))
            members += 1
            if len(batch) >= args.batch:
                pending.append(pool.apply_async(worker, (batch,)))
                batch = []
            if len(pending) >= args.workers * 4:
                collect(pending.pop(0))
        if batch:
            pending.append(pool.apply_async(worker, (batch,)))
        for job in pending:
            collect(job)
    stream.read()  # drain to finish hashing
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame["time"] = frame["time"].astype(float)
    frame = frame.sort_values(["airport", "icao", "time"]).reset_index(drop=True)
    frame.to_parquet(target, index=False, compression="zstd")
    manifest = {"day": args.day, "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": "adsb.lol globe_history (ODbL)",
                "release": f"v{args.day}-planes-readsb-prod-0", "assets": assets, "part_sha256": stream.hashes, "trace_members": members, "corrupt_members": corrupt,
                "rows": int(len(frame)), "aircraft": int(frame["icao"].nunique()) if len(frame) else 0,
                "rows_by_airport": {k: int(v) for k, v in frame["airport"].value_counts().sort_index().items()} if len(frame) else {},
                "ground_rows_by_airport": {k: int(v) for k, v in frame[frame["ground"]]["airport"].value_counts().sort_index().items()} if len(frame) else {},
                "seconds": round(time.time() - started, 1), "boxes": BOXES}
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ["day", "trace_members", "corrupt_members", "rows", "aircraft", "ground_rows_by_airport", "seconds"]}))


if __name__ == "__main__":
    main()
