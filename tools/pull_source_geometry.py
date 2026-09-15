"""Read bounded public OSM snapshots into a new, immutable audit cache."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json

CONFIG_PATH = ROOT / "configs/source_geometry_audit_v1.json"
CACHE = ROOT / "data/external/osm/source_geometry_v1"
RECEIPTS = ROOT / "docs/source_geometry/sources"


def retrieve(name, addresses, *, query=None, requested_snapshot=None):
    config = read_json(CONFIG_PATH)["geometry"]
    path, receipt_path = CACHE / name, RECEIPTS / (name + ".receipt.json")
    if path.exists() or receipt_path.exists():
        if not path.exists() or not receipt_path.exists():
            raise ValueError("Incomplete source pair; preserve it for inspection.")
        saved = read_json(receipt_path)
        if sha256_file(path) != saved["sha256"] or path.stat().st_size != saved["size_bytes"]:
            raise ValueError("The saved source changed.")
        print(json.dumps({"cached": name, "sha256": saved["sha256"]}), flush=True)
        return saved
    failures = []
    for address in addresses:
        url = address + "?" + urllib.parse.urlencode({"data": query}) if query else address
        started = utc_now()
        print(json.dumps({"reading": name, "provider": address, "started_at_utc": started}), flush=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "PRC2026-vigorous-whistle-research/1.0"})
            with urllib.request.urlopen(request, timeout=config["max_request_timeout_seconds"]) as response:
                payload = response.read(config["max_response_bytes"] + 1)
                final_url = response.url
                status = response.status
                content_type = response.headers.get("Content-Type")
            if len(payload) > config["max_response_bytes"]:
                raise ValueError("Public response exceeds the declared size limit.")
            metadata = {}
            if query:
                parsed = json.loads(payload)
                if "elements" not in parsed or parsed.get("remark"):
                    raise ValueError("Incomplete Overpass response: " + str(parsed.get("remark", "missing elements")))
                metadata = {"osm_metadata": parsed.get("osm3s", {}), "element_count": len(parsed["elements"])}
            CACHE.mkdir(parents=True, exist_ok=True)
            RECEIPTS.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(payload)
            report = {
                "started_at_utc": started, "retrieved_at_utc": utc_now(),
                "url": url, "final_url": final_url, "http_status": status,
                "content_type": content_type, "path": path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload),
                "query": query, "requested_snapshot": requested_snapshot,
                "declaration_sha256": sha256_file(CONFIG_PATH),
                "tool_sha256": sha256_file(Path(__file__)), "previous_attempts": failures,
                "license": "ODbL-1.0 for map data; see archived OSM copyright and ODbL legal text",
                **metadata,
            }
            write_json(receipt_path, report)
            print(json.dumps({"saved": name, "bytes": len(payload), **metadata}), flush=True)
            return report
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
            failure = {"started_at_utc": started, "finished_at_utc": utc_now(),
                       "url": url, "error_type": type(exc).__name__, "error": str(exc)[:1200]}
            failures.append(failure)
            RECEIPTS.mkdir(parents=True, exist_ok=True)
            stamp = failure["finished_at_utc"].replace(":", "").replace("+", "_")
            write_json(RECEIPTS / f"{name}.failed_{stamp}.json", failure)
            print(json.dumps({"failed": name, "provider": address, "error": str(exc)[:180]}), flush=True)
            time.sleep(config["delay_between_requests_seconds"])
    raise RuntimeError(f"All declared sources failed for {name}; attempts preserved.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", action="store_true")
    parser.add_argument("--historical", choices=["LFPG", "LIRF"])
    parser.add_argument("--licenses", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    config = read_json(CONFIG_PATH)
    geo = config["geometry"]
    if args.offline:
        receipts = sorted(RECEIPTS.glob("*.receipt.json"))
        if not receipts:
            raise ValueError("No source receipts exist.")
        for path in receipts:
            item = read_json(path)
            source = ROOT / item["path"]
            if sha256_file(source) != item["sha256"] or source.stat().st_size != item["size_bytes"]:
                raise ValueError("Changed source: " + str(source))
        print(json.dumps({"verified_sources": len(receipts)}))
        return
    tasks = [(airport, "current") for airport in config["airports"]] if args.current else []
    if args.historical:
        tasks += [(args.historical, stamp) for stamp in geo["requested_snapshots"] if stamp != "current"]
    for airport, stamp in tasks:
        date = "" if stamp == "current" else f'[date:"{stamp}"]'
        bbox = ",".join(map(str, geo["bbox_south_west_north_east"][airport]))
        values = "|".join(geo["aeroway_tags"])
        query = f'[out:json][timeout:30][maxsize:268435456]{date};nwr["aeroway"~"^({values})$"]({bbox});out meta geom;'
        label = "current" if stamp == "current" else stamp[:10].replace("-", "")
        retrieve(f"{airport}_{label}.json", geo["endpoints"], query=query, requested_snapshot=stamp)
        time.sleep(geo["delay_between_requests_seconds"])
    if args.licenses:
        for name, address in zip(["osm_copyright.html", "odbl_1_0.html"], geo["license_urls"]):
            retrieve(name, [address])
            time.sleep(geo["delay_between_requests_seconds"])
    if not tasks and not args.licenses:
        parser.error("Choose --current, --historical, --licenses or --offline.")


if __name__ == "__main__":
    main()
