"""Cache X-Plane Scenery Gateway airport data (GPL-2.0-or-later) for the ten challenge airports; never reads labels.

For each airport this stores the Gateway's airport record and two dated scenery versions: the latest version
approved before 2025-01-01 ("pre_data", older than every movement in the supplied data) and the version the
Gateway recommends at fetch time ("recommended"). Each version's zip blob is decoded and saved as it was served;
`apt.dat` stand positions (row code 1300) are parsed from the contained ICAO.dat file by
`tools/probe_xplane_stands.py`. Requests are sequential with a fixed gap and bounded retries.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.artifacts import utc_now, write_json

ROOT = Path(__file__).resolve().parents[1]
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
API = "https://gateway.x-plane.com/apiv1"
LICENSE = ("X-Plane Scenery Gateway airport data: GNU General Public License, version 2 or (at your option) "
           "any later version; copyright the Gateway contributors; the copyright notice must be kept on redistribution.")
USER_AGENT = "prc2026-vigorous-whistle/0.1 (public research; sequential bounded requests)"
CUTOFF = "2025-01-01T00:00:00.000Z"
OUT = ROOT / "data/external/xplane_gateway"


def get(url: str) -> bytes:
    for attempt in range(4):
        time.sleep(2)
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=90)
        if response.status_code == 200:
            return response.content
        if response.status_code not in (429, 500, 502, 503, 504):
            response.raise_for_status()
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Gateway request did not succeed: {url}")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pull(icao: str) -> dict:
    record_bytes = get(f"{API}/airport/{icao}")
    (OUT / f"{icao}_airport.json").write_bytes(record_bytes)
    airport = json.loads(record_bytes)["airport"]
    approved = [s for s in airport["scenery"] if s["Status"] == "Approved" and s["dateApproved"]]
    before = [s for s in approved if s["dateApproved"] < CUTOFF]
    chosen = {"recommended": airport["recommendedSceneryId"]}
    if before:
        chosen["pre_data"] = max(before, key=lambda s: s["dateApproved"])["sceneryId"]
    versions = {}
    for role, scenery_id in chosen.items():
        if scenery_id in [v["scenery_id"] for v in versions.values()]:
            versions[role] = next(dict(v) for v in versions.values() if v["scenery_id"] == scenery_id)
            continue
        payload = json.loads(get(f"{API}/scenery/{scenery_id}"))["scenery"]
        blob = base64.b64decode(payload.pop("masterZipBlob"))
        zip_path = OUT / f"{icao}_{scenery_id}.zip"
        zip_path.write_bytes(blob)
        meta_path = OUT / f"{icao}_{scenery_id}.json"
        write_json(meta_path, payload)
        versions[role] = {"scenery_id": scenery_id, "date_approved": payload["dateApproved"],
                          "user": payload["userName"], "zip": zip_path.relative_to(ROOT).as_posix(),
                          "zip_sha256": sha(blob), "metadata_sha256": sha(meta_path.read_bytes())}
    return {"icao": icao, "record_sha256": sha(record_bytes), "approved_versions": len(approved), "versions": versions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", nargs="*", default=AIRPORTS)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/xplane_stands/sources.json")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    airports = [pull(icao) for icao in args.airports]
    write_json(args.manifest, {"fetched_started_utc": started, "fetched_finished_utc": utc_now(), "api": API,
                               "license": LICENSE, "pre_data_cutoff_utc": CUTOFF, "airports": airports})
    print(json.dumps(airports, indent=1))


if __name__ == "__main__":
    main()
