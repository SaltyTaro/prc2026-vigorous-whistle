"""Probe: does adsb.lol's `mlatonly` release add surface positions the prod replica lacks at the ten airports? (ODbL, read-only)

For each given day, streams the `v<day>-planes-readsb-mlatonly-0` release through tools/adsb_extract_day.py's decoder (same airport
boxes, ground or below 3,000 ft) and compares it with the stored prod surface file of the same day
(data/external/adsblol/surface/surface_<day>.parquet): per airport, aircraft and ground points in each, and aircraft (ICAO) with ground
points in mlatonly but none in prod. Writes docs/adsb_coverage/mlatonly_probe.json; nothing is fitted or scored.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from adsb_extract_day import COLUMNS, HashingStream, worker  # noqa: E402


def assets_for(day: str, kind: str) -> list[dict]:
    tag = f"v{day}-planes-readsb-{kind}"
    url = f"https://api.github.com/repos/adsblol/globe_history_{day[:4]}/releases/tags/{tag}"
    release = json.loads(subprocess.run(["curl", "-sL", "-H", "Accept: application/vnd.github+json", url], check=True, capture_output=True).stdout)
    if "assets" not in release:
        raise RuntimeError(f"No release for {tag}: {release.get('message')}")
    return [{"name": a["name"], "size": a["size"], "url": a["browser_download_url"]} for a in sorted(release["assets"], key=lambda a: a["name"]) if ".tar" in a["name"]]


def extract(day: str, kind: str, workers: int) -> tuple[pd.DataFrame, dict]:
    assets = assets_for(day, kind)
    stream = HashingStream(assets)
    rows, members, batch, pending = [], 0, [], []
    with mp.Pool(workers) as pool, tarfile.open(fileobj=stream, mode="r|") as archive:
        for member in archive:
            if not member.isfile() or "/traces/" not in member.name:
                continue
            batch.append((member.name, archive.extractfile(member).read()))
            members += 1
            if len(batch) >= 64:
                pending.append(pool.apply_async(worker, (batch,)))
                batch = []
            if len(pending) >= workers * 4:
                rows.extend(pending.pop(0).get()[0])
        if batch:
            pending.append(pool.apply_async(worker, (batch,)))
        for job in pending:
            rows.extend(job.get()[0])
    stream.read()
    return pd.DataFrame(rows, columns=COLUMNS), {"tag": f"v{day}-planes-readsb-{kind}", "assets": assets, "part_sha256": stream.hashes, "trace_members": members}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", default=["2026.01.15", "2026.07.15"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_coverage/mlatonly_probe.json"))
    args = parser.parse_args()
    report = {"note": "read-only probe; ODbL source adsb.lol globe_history", "days": {}}
    for day in args.days:
        started = time.time()
        mlat, meta = extract(day, "mlatonly-0", args.workers)
        prod = pd.read_parquet(ROOT / "data/external/adsblol/surface" / f"surface_{day.replace('.', '')}.parquet")
        entry = {"mlatonly": meta, "airports": {}}
        for airport in sorted(set(prod["airport"]) | set(mlat["airport"]) | {"LFPG", "LTFM", "LIRF", "EGLL"}):
            p, m = prod[prod["airport"].eq(airport)], mlat[mlat["airport"].eq(airport)]
            pg, mg = set(p.loc[p["ground"], "icao"]), set(m.loc[m["ground"], "icao"])
            entry["airports"][airport] = {"prod_aircraft": int(p["icao"].nunique()), "prod_ground_aircraft": len(pg), "prod_ground_points": int(p["ground"].sum()),
                                          "mlat_aircraft": int(m["icao"].nunique()), "mlat_ground_aircraft": len(mg), "mlat_ground_points": int(m["ground"].sum()),
                                          "mlat_ground_aircraft_not_in_prod_ground": len(mg - pg)}
        entry["seconds"] = round(time.time() - started, 1)
        report["days"][day] = entry
        print(day, json.dumps(entry["airports"]), entry["seconds"], flush=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
